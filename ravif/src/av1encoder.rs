use imgref::Img;
use rav1e::prelude::*;
use rgb::RGBA8;

/// See [`Config`]
#[derive(Debug, Copy, Clone)]
pub enum ColorSpace {
    YCbCr,
    RGB,
}

/// Encoder configuration struct
///
/// See [`encode_rgba`](crate::encode_rgba)
#[derive(Debug, Copy, Clone)]
pub struct EncConfig {
    /// 0-100 scale
    pub quality: u8,
    /// 0-100 scale
    pub alpha_quality: u8,
    /// rav1e preset 1 (slow) 10 (fast but crappy)
    pub speed: u8,
    /// True if RGBA input has already been premultiplied. It inserts appropriate metadata. Warning: decoding of this is not supported by libavif yet.
    pub premultiplied_alpha: bool,
    /// Which pixel format to use in AVIF file. RGB tends to give larger files.
    pub color_space: ColorSpace,
    /// How many threads should be used (0 = match core count)
    pub threads: usize,    
}

/// Make a new AVIF image from RGBA pixels
///
/// Make the `Img` for the `buffer` like this:
///
/// ```rust,ignore
/// Img::new(&pixels_rgba[..], width, height)
/// ```
///
/// If you have pixels as `u8` slice, then:
///
/// ```rust,ignore
/// use rgb::ComponentSlice;
/// let pixels_rgba = pixels.as_rgba();
/// ```
///
/// If all pixels are opaque, alpha channel will be left out automatically.
///
/// It's highly recommended to apply [`cleared_alpha`](crate::cleared_alpha) first.
pub fn encode_rgba(buffer: Img<&[RGBA8]>, config: &EncConfig) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let width = buffer.width();
    let height = buffer.height();
    let mut y_plane = Vec::with_capacity(width*height);
    let mut u_plane = Vec::with_capacity(width*height);
    let mut v_plane = Vec::with_capacity(width*height);
    let mut a_plane = Vec::with_capacity(width*height);
    for px in buffer.pixels() {
        let (y,u,v) = match config.color_space {
            ColorSpace::YCbCr => {
                let y  = 0.2126 * px.r as f32 + 0.7152 * px.g as f32 + 0.0722 * px.b as f32;
                let cb = (px.b as f32 - y) * (0.5/(1.-0.0722));
                let cr = (px.r as f32 - y) * (0.5/(1.-0.2126));

                ((y * (235.-16.)/255. + 16_f32).round().max(0.).min(255.) as u8,
                ((cb + 128.) * (240.-16.)/255. + 16_f32).round().max(0.).min(255.) as u8,
                ((cr + 128.) * (240.-16.)/255. + 16_f32).round().max(0.).min(255.) as u8)
            },
            ColorSpace::RGB => {
                (px.g, px.b, px.r)
            },
        };
        y_plane.push(y);
        u_plane.push(u);
        v_plane.push(v);
        a_plane.push(px.a);
    }

    // quality setting
    let quantizer = quality_to_quantizer(config.quality);
    let alpha_quantizer = quality_to_quantizer(config.alpha_quality);
    let use_alpha = a_plane.iter().copied().any(|b| b != 255);

    let (color_pixel_range, matrix_coefficients) = match config.color_space {
        ColorSpace::YCbCr => (PixelRange::Limited, MatrixCoefficients::BT709),
        ColorSpace::RGB => (PixelRange::Full, MatrixCoefficients::Identity),
    };

    let color_description = Some(ColorDescription {
        transfer_characteristics: TransferCharacteristics::SRGB,
        color_primaries: ColorPrimaries::BT709, // sRGB-compatible
        matrix_coefficients,
    });
    // Firefox 81 doesn't support Full yet, but doesn't support alpha either
    let (color, alpha) = rayon::join(
        || encode_to_av1(width, height, &[&y_plane, &u_plane, &v_plane], quantizer, config.speed, config.threads, color_pixel_range, ChromaSampling::Cs444, color_description),
        || if use_alpha {
            Some(encode_to_av1(width, height, &[&a_plane], alpha_quantizer, config.speed, config.threads, PixelRange::Full, ChromaSampling::Cs400, None))
          } else {
            None
        });
    let (color, alpha) = (color?, alpha.transpose()?);

    let out = avif_serialize::Aviffy::new()
        .premultiplied_alpha(config.premultiplied_alpha)
        .to_vec(&color, alpha.as_deref(), width as u32, height as u32, 8);
    let color_size = color.len();
    let alpha_size = alpha.as_ref().map_or(0, |a| a.len());

    Ok((out, color_size, alpha_size))
}

fn quality_to_quantizer(quality: u8) -> usize {
    ((1.-(quality as f32)/100.) * 255.).round().max(0.).min(255.) as usize
}

fn encode_to_av1(width: usize, height: usize, planes: &[&[u8]], quantizer: usize, speed: u8, threads: usize, pixel_range: PixelRange, chroma_sampling: ChromaSampling, color_description: Option<ColorDescription>) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // AV1 needs all the CPU power you can give it,
    // except when it'd create inefficiently tiny tiles
    let cpus = if threads > 0 { threads } else { num_cpus::get() };
    let tiles = cpus.min((width * height) / (128 * 128));

    let cfg = Config::new()
        .with_threads(cpus.into())
        .with_encoder_config(EncoderConfig {
        width,
        height,
        time_base: Rational::new(1, 1),
        sample_aspect_ratio: Rational::new(1, 1),
        bit_depth: 8,
        chroma_sampling,
        chroma_sample_position: if chroma_sampling == ChromaSampling::Cs400 {
            ChromaSamplePosition::Unknown
        } else {
            ChromaSamplePosition::Colocated
        },
        pixel_range,
        color_description,
        mastering_display: None,
        content_light: None,
        enable_timing_info: false,
        still_picture: true,
        error_resilient: false,
        switch_frame_interval: 0,
        min_key_frame_interval: 0,
        max_key_frame_interval: 0,
        reservoir_frame_delay: None,
        low_latency: false,
        quantizer,
        min_quantizer: quantizer as _,
        bitrate: 0,
        tune: Tune::Psychovisual,
        tile_cols: 0,
        tile_rows: 0,
        tiles,
        rdo_lookahead_frames: 1,
        speed_settings: SpeedSettings::from_preset(speed.into()),
    });

    let mut ctx: Context<u8> = cfg.new_context()?;
    let mut frame = ctx.new_frame();

    for (dst, src) in frame.planes.iter_mut().zip(planes) {
        dst.copy_from_raw_u8(src, width, 1);
    }

    ctx.send_frame(frame)?;
    ctx.flush();

    let mut out = Vec::new();
    loop {
        match ctx.receive_packet() {
            Ok(mut packet) => match packet.frame_type {
                FrameType::KEY => {
                    out.append(&mut packet.data);
                }
                _ => continue,
            },
            Err(EncoderStatus::Encoded) |
            Err(EncoderStatus::LimitReached) => break,
            Err(err) => Err(err)?,
        }
    }
    Ok(out)
}
