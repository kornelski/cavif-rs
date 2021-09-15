use imgref::Img;
use rav1e::prelude::*;
use rgb::RGB8;
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
    pub quality: f32,
    /// 0-100 scale
    pub alpha_quality: f32,
    /// rav1e preset 1 (slow) 10 (fast but crappy)
    pub speed: u8,
    /// True if RGBA input has already been premultiplied. It inserts appropriate metadata.
    pub premultiplied_alpha: bool,
    /// Which pixel format to use in AVIF file. RGB tends to give larger files.
    pub color_space: ColorSpace,
    /// How many threads should be used (0 = match core count)
    pub threads: usize,
}

/// Make a new AVIF image from RGBA pixels (non-premultiplied, alpha last)
///
/// Make the `Img` for the `buffer` like this:
///
/// ```rust,ignore
/// Img::new(&pixels_rgba[..], width, height)
/// ```
///
/// If you have pixels as `u8` slice, then first do:
///
/// ```rust,ignore
/// use rgb::ComponentSlice;
/// let pixels_rgba = pixels_u8.as_rgba();
/// ```
///
/// If all pixels are opaque, alpha channel will be left out automatically.
///
/// It's highly recommended to apply [`cleared_alpha`](crate::cleared_alpha) first.
///
/// returns AVIF file, size of color metadata, size of alpha metadata overhead
pub fn encode_rgba(buffer: Img<&[RGBA8]>, config: &EncConfig) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let width = buffer.width();
    let height = buffer.height();
    if buffer.buf().len() < width * height {
        return Err("Too few pixels".into());
    }
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

                (y.round() as u8, (cb + 128.).round() as u8, (cr + 128.).round() as u8)
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

    let use_alpha = a_plane.iter().copied().any(|b| b != 255);
    let color_pixel_range = PixelRange::Full;

    encode_raw_planes(width, height, &y_plane, &u_plane, &v_plane, if use_alpha { Some(&a_plane) } else { None }, color_pixel_range, config)
}

/// Make a new AVIF image from RGB pixels
///
/// Make the `Img` for the `buffer` like this:
///
/// ```rust,ignore
/// Img::new(&pixels_rgb[..], width, height)
/// ```
///
/// If you have pixels as `u8` slice, then first do:
///
/// ```rust,ignore
/// use rgb::ComponentSlice;
/// let pixels_rgba = pixels_u8.as_rgb();
/// ```
///
/// returns AVIF file, size of color metadata
pub fn encode_rgb(buffer: Img<&[RGB8]>, config: &EncConfig) -> Result<(Vec<u8>, usize), Box<dyn std::error::Error + Send + Sync>> {
    let width = buffer.width();
    let height = buffer.height();
    if buffer.buf().len() < width * height {
        return Err("Too few pixels".into());
    }
    let mut y_plane = Vec::with_capacity(width*height);
    let mut u_plane = Vec::with_capacity(width*height);
    let mut v_plane = Vec::with_capacity(width*height);
    for px in buffer.pixels() {
        let (y,u,v) = match config.color_space {
            ColorSpace::YCbCr => {
                let y  = 0.2126 * px.r as f32 + 0.7152 * px.g as f32 + 0.0722 * px.b as f32;
                let cb = (px.b as f32 - y) * (0.5/(1.-0.0722));
                let cr = (px.r as f32 - y) * (0.5/(1.-0.2126));

                (y.round() as u8, (cb + 128.).round() as u8, (cr + 128.).round() as u8)
            },
            ColorSpace::RGB => {
                (px.g, px.b, px.r)
            },
        };
        y_plane.push(y);
        u_plane.push(u);
        v_plane.push(v);
    }

    let color_pixel_range = PixelRange::Full;

    let (avif, heif_bloat, _) = encode_raw_planes(width, height, &y_plane, &u_plane, &v_plane, None, color_pixel_range, config)?;
    Ok((avif, heif_bloat))
}

/// If config.color_space is ColorSpace::YCbCr, then it takes 8-bit BT709 color space.
///
/// Alpha always uses full range. Chroma subsampling is not supported, and it's a bad idea for AVIF anyway.
///
/// returns AVIF file, size of color metadata, size of alpha metadata overhead
pub fn encode_raw_planes(width: usize, height: usize, y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], a_plane: Option<&[u8]>, color_pixel_range: PixelRange, config: &EncConfig) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    if y_plane.len() < width * height {
        return Err("Too few pixels".into());
    }

    // quality setting
    let quantizer = quality_to_quantizer(config.quality);
    let alpha_quantizer = quality_to_quantizer(config.alpha_quality);

    let matrix_coefficients = match config.color_space {
        ColorSpace::YCbCr => MatrixCoefficients::BT709,
        ColorSpace::RGB => MatrixCoefficients::Identity,
    };

    let color_description = Some(ColorDescription {
        transfer_characteristics: TransferCharacteristics::SRGB,
        color_primaries: ColorPrimaries::BT709, // sRGB-compatible
        matrix_coefficients,
    });

    let threads = if config.threads > 0 { config.threads } else { num_cpus::get() };

    // Firefox 81 doesn't support Full yet, but doesn't support alpha either
    let (color, alpha) = rayon::join(
        || encode_to_av1(&Av1EncodeConfig {
                width,
                height,
                planes: &[&y_plane, &u_plane, &v_plane],
                quantizer,
                speed: SpeedTweaks::from_my_preset(config.speed, config.quality as _),
                threads,
                pixel_range: color_pixel_range,
                chroma_sampling: ChromaSampling::Cs444,
                color_description,
            }),
        || if let Some(a_plane) = a_plane {
            Some(encode_to_av1(&Av1EncodeConfig {
                width,
                height,
                planes: &[&a_plane],
                quantizer: alpha_quantizer,
                speed: SpeedTweaks::from_my_preset(config.speed, config.alpha_quality as _),
                threads,
                pixel_range: PixelRange::Full,
                chroma_sampling: ChromaSampling::Cs400,
                color_description: None,
            }))
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

fn quality_to_quantizer(quality: f32) -> usize {
    ((1.-quality/100.) * 255.).round().max(0.).min(255.) as usize
}


#[derive(Debug, Copy, Clone)]
pub struct SpeedTweaks {
    pub speed_preset: u8,

    pub fast_deblock: Option<bool>,
    pub reduced_tx_set: Option<bool>,
    pub tx_domain_distortion: Option<bool>,
    pub tx_domain_rate: Option<bool>,
    pub encode_bottomup: Option<bool>,
    pub rdo_tx_decision: Option<bool>,
    pub cdef: Option<bool>,
    /// loop restoration filter
    pub lrf: Option<bool>,
    pub non_square_partition: Option<bool>,
    pub sgr_complexity_full: Option<bool>,
    pub use_satd_subpel: Option<bool>,
    pub inter_tx_split: Option<bool>,
    pub fine_directional_intra: Option<bool>,
    pub complex_prediction_modes: Option<bool>,
    pub partition_range: Option<(u8, u8)>,
    pub min_tile_size: u16,
}

impl SpeedTweaks {
    pub fn from_my_preset(speed: u8, quality: u8) -> Self {
        let low_quality = quality < 60;

        Self {
            speed_preset: speed,

            partition_range: Some(match speed {
                0 => (4, 64),
                1 if low_quality => (4, 64),
                2 if low_quality => (4, 32),
                1..=4 => (4, 16),
                5..=8 => (8, 16),
                _ => (16, 16),
            }),

            complex_prediction_modes: Some(speed <= 1), // 2x-3x slower, 2% better
            sgr_complexity_full: Some(speed <= 2), // 15% slower, barely improves anything -/+1%

            encode_bottomup: Some(speed <= 2), // may be costly (+60%), may even backfire

            // big blocks disabled at 3
            non_square_partition: Some(speed <= 3), // doesn't seem to do anything?

            // these two are together?
            rdo_tx_decision: Some(speed <= 4),
            reduced_tx_set: Some(speed == 4 || speed >= 9), // It interacts with tx_domain_distortion too?

            // 4px blocks disabled at 5

            fine_directional_intra: Some(speed <= 6),
            fast_deblock: Some(speed >= 7), // mixed bag?

            // 8px blocks disabled at 8
            lrf: Some(low_quality && speed <= 8), // hardly any help for hi-q images. recovers some q at low quality
            cdef: Some(low_quality && speed <= 9), // hardly any help for hi-q images. recovers some q at low quality

            inter_tx_split: Some(speed >= 9), // mixed bag even when it works, and it backfires if not used together with reduced_tx_set
            tx_domain_rate: Some(speed >= 10), // 20% faster, but also 10% larger files!

            tx_domain_distortion: None, // very mixed bag, sometimes helps speed sometimes it doesn't
            use_satd_subpel: Some(false), // doesn't make sense
            min_tile_size: match speed {
                0 => 4096,
                1 => 2048,
                2 => 1024,
                3 => 512,
                4 => 256,
                _ => 128,
            },
        }
    }

    pub(crate) fn speed_settings(&self) -> SpeedSettings {

        let mut speed_settings = SpeedSettings::from_preset(self.speed_preset.into());

        speed_settings.multiref = false;
        speed_settings.no_scene_detection = true;
        speed_settings.include_near_mvs = false;

        if let Some(v) = self.fast_deblock { speed_settings.fast_deblock = v; }
        if let Some(v) = self.reduced_tx_set { speed_settings.reduced_tx_set = v; }
        if let Some(v) = self.tx_domain_distortion { speed_settings.tx_domain_distortion = v; }
        if let Some(v) = self.tx_domain_rate { speed_settings.tx_domain_rate = v; }
        if let Some(v) = self.encode_bottomup { speed_settings.encode_bottomup = v; }
        if let Some(v) = self.rdo_tx_decision { speed_settings.rdo_tx_decision = v; }
        if let Some(v) = self.cdef { speed_settings.cdef = v; }
        if let Some(v) = self.lrf { speed_settings.lrf = v; }
        if let Some(v) = self.inter_tx_split { speed_settings.enable_inter_tx_split = v; }
        if let Some(v) = self.non_square_partition { speed_settings.non_square_partition = v; }
        if let Some(v) = self.sgr_complexity_full { speed_settings.sgr_complexity = if v { SGRComplexityLevel::Full } else { SGRComplexityLevel::Reduced } };
        if let Some(v) = self.use_satd_subpel { speed_settings.use_satd_subpel = v; }
        if let Some(v) = self.fine_directional_intra { speed_settings.fine_directional_intra = v; }
        if let Some(v) = self.complex_prediction_modes { speed_settings.prediction_modes = if v { PredictionModesSetting::ComplexAll } else { PredictionModesSetting::Simple} };
        if let Some((min, max)) = self.partition_range {
            assert!(min <= max);
            fn sz(s: u8) -> BlockSize {
                match s {
                    4 => BlockSize::BLOCK_4X4,
                    8 => BlockSize::BLOCK_8X8,
                    16 => BlockSize::BLOCK_16X16,
                    32 => BlockSize::BLOCK_32X32,
                    64 => BlockSize::BLOCK_64X64,
                    128 => BlockSize::BLOCK_128X128,
                    _ => panic!("bad size {}", s),
                }
            }
            speed_settings.partition_range = PartitionRange::new(sz(min), sz(max));
        }

        speed_settings
    }
}

pub(crate) struct Av1EncodeConfig<'a> {
    pub width: usize,
    pub height: usize,
    pub planes: &'a [&'a [u8]],
    pub quantizer: usize,
    pub speed: SpeedTweaks,
    pub threads: usize,
    pub pixel_range: PixelRange,
    pub chroma_sampling: ChromaSampling,
    pub color_description: Option<ColorDescription>,
}

fn encode_to_av1(p: &Av1EncodeConfig<'_>) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // AV1 needs all the CPU power you can give it,
    // except when it'd create inefficiently tiny tiles
    let tiles = p.threads.min((p.width * p.height) / (p.speed.min_tile_size as usize).pow(2));
    let bit_depth = 8;

    let speed_settings = p.speed.speed_settings();
    let cfg = Config::new()
        .with_threads(p.threads.into())
        .with_encoder_config(EncoderConfig {
        width: p.width,
        height: p.height,
        time_base: Rational::new(1, 1),
        sample_aspect_ratio: Rational::new(1, 1),
        bit_depth,
        chroma_sampling: p.chroma_sampling,
        chroma_sample_position: ChromaSamplePosition::Unknown,
        pixel_range: p.pixel_range,
        color_description: p.color_description,
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
        quantizer: p.quantizer,
        min_quantizer: p.quantizer as _,
        bitrate: 0,
        tune: Tune::Psychovisual,
        tile_cols: 0,
        tile_rows: 0,
        tiles,
        rdo_lookahead_frames: 1,
        speed_settings,
    });

    let mut ctx: Context<u8> = cfg.new_context()?;
    let mut frame = ctx.new_frame();

    for (dst, src) in frame.planes.iter_mut().zip(p.planes) {
        dst.copy_from_raw_u8(src, p.width, (bit_depth+7)/8);
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
