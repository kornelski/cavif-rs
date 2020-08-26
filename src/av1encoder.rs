use rgb::RGBA8;
use rav1e::prelude::*;

pub fn encode_rgba(width: usize, height: usize, buffer: &[RGBA8], quality: u8, speed: u8) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let mut y_plane = Vec::with_capacity(width*height);
    let mut u_plane = Vec::with_capacity(width*height);
    let mut v_plane = Vec::with_capacity(width*height);
    let mut a_plane = Vec::with_capacity(width*height);
    for px in buffer.iter().copied() {
        let y  = 0.2126 * px.r as f32 + 0.7152 * px.g as f32 + 0.0722 * px.b as f32;
        let cb = (px.b as f32 - y) * (1./1.8556);
        let cr = (px.r as f32 - y) * (1./1.5748);

        y_plane.push((y * (235.-16.)/255. + 16_f32).round().max(0.).min(255.) as u8);
        u_plane.push(((cb + 128.) * (240.-16.)/255. + 16_f32).round().max(0.).min(255.) as u8);
        v_plane.push(((cr + 128.) * (240.-16.)/255. + 16_f32).round().max(0.).min(255.) as u8);
        a_plane.push(px.a);
    }

    // quality setting
    let quantizer = ((1.-(quality as f32)/100.) * 255.).round().max(0.).min(255.) as usize;
    let use_alpha = a_plane.iter().copied().any(|b| b != 255);

    let color_description = Some(ColorDescription {
        transfer_characteristics: TransferCharacteristics::SRGB,
        color_primaries: ColorPrimaries::BT709, // sRGB-compatible
        matrix_coefficients: MatrixCoefficients::BT709,
    });
    // Firefox 81 doesn't support Full yet, but doesn't support alpha either
    let (color, alpha) = rayon::join(
        || encode_to_av1(width, height, &[&y_plane, &u_plane, &v_plane], quantizer, speed, PixelRange::Limited, ChromaSampling::Cs444, color_description),
        || if use_alpha {
            Some(encode_to_av1(width, height, &[&a_plane], quantizer, speed, PixelRange::Full, ChromaSampling::Cs400, None))
          } else {
            None
        });
    let (color, alpha) = (color?, alpha.transpose()?);

    let out = avif_serialize::serialize_to_vec(&color, alpha.as_deref(), width as u32, height as u32, 8);
    let color_size = color.len();
    let alpha_size = alpha.as_ref().map_or(0, |a| a.len());

    Ok((out, color_size, alpha_size))
}

fn encode_to_av1(width: usize, height: usize, planes: &[&[u8]], quantizer: usize, speed: u8, pixel_range: PixelRange, chroma_sampling: ChromaSampling, color_description: Option<ColorDescription>) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // AV1 needs all the CPU power you can give it,
    // except when it'd create inefficiently tiny tiles
    let tiles = num_cpus::get().min((width*height) / (128*128));

    let cfg = Config::new()
        .with_threads(num_cpus::get())
        .with_encoder_config(EncoderConfig {
        width,
        height,
        time_base: Rational::new(1, 1),
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
            Ok(mut packet) => {
                match packet.frame_type {
                    FrameType::KEY => {
                        out.append(&mut packet.data);
                    },
                    _ => continue,
                }
            },
            Err(EncoderStatus::Encoded) |
            Err(EncoderStatus::LimitReached) => break,
            Err(err) => Err(err)?,
        }
    }
    Ok(out)
}
