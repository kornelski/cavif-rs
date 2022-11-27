#![allow(deprecated)]
use crate::dirtyalpha::blurred_dirty_alpha;
use crate::error::Error;
use imgref::Img;
use imgref::ImgVec;
use rav1e::prelude::*;
use rgb::RGB8;
use rgb::RGBA8;

/// For [`Encoder::with_internal_color_space`]
#[derive(Debug, Copy, Clone)]
pub enum ColorSpace {
    /// Standard color space for photographic content. Usually the best choice.
    /// This library always uses full-resolution color (4:4:4).
    YCbCr,
    /// RGB channels are encoded without colorspace transformation.
    /// Usually results in larger file sizes, and is less compatible than `YCbCr`.
    /// Use only if the content really makes use of RGB, e.g. anaglyph images or RGB subpixel anti-aliasing.
    RGB,
}

/// Handling of color channels in transparent images. For [`Encoder::with_alpha_color_mode`]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum AlphaColorMode {
    /// Use unassociated alpha channel and leave color channels unchanged, even if there's redundant color data in transparent areas.
    UnassociatedDirty,
    /// Use unassociated alpha channel, but set color channels of transparent areas to a solid color to eliminate invisible data and improve compression.
    UnassociatedClean,
    /// Store color channels of transparent images in premultiplied form.
    /// This requires support for premultiplied alpha in AVIF decoders.
    ///
    /// It may reduce file sizes due to clearing of fully-transparent pixels, but
    /// may also increase file sizes due to creation of new edges in the color channels.
    ///
    /// Note that this is only internal detail for the AVIF file.
    /// It does not change meaning of `RGBA` in this library — it's always unassociated.
    Premultiplied,
}

/// The image file + extra info
#[non_exhaustive]
#[derive(Clone)]
pub struct EncodedImage {
    /// AVIF (HEIF+AV1) encoded image data
    pub avif_file: Vec<u8>,
    /// FIY: number of bytes of AV1 payload used for the color
    pub color_byte_size: usize,
    /// FIY: number of bytes of AV1 payload used for the alpha channel
    pub alpha_byte_size: usize,
}

/// Encoder config builder
#[derive(Debug, Clone)]
pub struct Encoder {
    config: EncConfig,
    /// `false` uses 8 bits, `true` uses 10 bits
    high_bit_depth: bool,
    /// [`AlphaColorMode`]
    alpha_color_mode: AlphaColorMode,
}

/// Builder methods
impl Encoder {
    /// Start here
    pub fn new() -> Self {
        Self {
            config: EncConfig {
                quality: 80.,
                alpha_quality: 80.,
                speed: 5,
                premultiplied_alpha: false,
                color_space: ColorSpace::YCbCr,
                threads: None,
            },
            high_bit_depth: false,
            alpha_color_mode: AlphaColorMode::UnassociatedClean,
        }
    }

    /// Quality 1..=100. Panics if out of range.
    #[inline(always)]
    #[track_caller]
    pub fn with_quality(mut self, quality: f32) -> Self {
        assert!(quality >= 1. && quality <= 100.);
        self.config.quality = quality;
        self
    }

    /// Quality for the alpha channel only. 1..=100. Panics if out of range.
    #[inline(always)]
    #[track_caller]
    pub fn with_alpha_quality(mut self, quality: f32) -> Self {
        assert!(quality >= 1. && quality <= 100.);
        self.config.alpha_quality = quality;
        self
    }

    /// 1..=10. 1 = very very slow, but max compression.
    /// 10 = quick, but larger file sizes and lower quality.
    #[inline(always)]
    #[track_caller]
    pub fn with_speed(mut self, speed: u8) -> Self {
        assert!(speed >= 1 && speed <= 10);
        self.config.speed = speed;
        self
    }

    /// Changes how color channels are stored in the image. The default is YCbCr.
    ///
    /// Note that this is only internal detail for the AVIF file, and doesn't
    /// change color space of inputs to encode functions.
    #[inline(always)]
    pub fn with_internal_color_space(mut self, color_space: ColorSpace) -> Self {
        self.config.color_space = color_space;
        self
    }

    /// Store color channels using 10-bit depth instead of the default 8-bit.
    #[inline(always)]
    pub fn with_internal_10_bit_depth(mut self, high_bit_depth: bool) -> Self {
        self.high_bit_depth = high_bit_depth;
        self
    }

    /// Configures `rayon` thread pool size.
    /// The default `None` is to use all threads in the default `rayon` thread pool.
    #[inline(always)]
    #[track_caller]
    pub fn with_num_threads(mut self, num_threads: Option<usize>) -> Self {
        assert!(num_threads.map_or(true, |n| n > 0));
        self.config.threads = num_threads;
        self
    }

    /// Configure handling of color channels in transparent images
    #[inline(always)]
    pub fn with_alpha_color_mode(mut self, mode: AlphaColorMode) -> Self {
        self.alpha_color_mode = mode;
        self.config.premultiplied_alpha = mode == AlphaColorMode::Premultiplied;
        self
    }
}

/// Use [`Encoder::new`] instead.
#[deprecated(note = "use Encoder::new()")]
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
    /// How many threads should be used (0 = match core count), None - use global rayon thread pool
    pub threads: Option<usize>,
}

/// Once done with config, call one of the `encode_*` functions
impl Encoder {

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
pub fn encode_rgba(&self, in_buffer: Img<&[RGBA8]>) -> Result<EncodedImage, Error> {
    let tmp;
    let buffer = match self.alpha_color_mode {
        AlphaColorMode::UnassociatedDirty => in_buffer,
        AlphaColorMode::UnassociatedClean => {
            if let Some(new) = blurred_dirty_alpha(in_buffer) {
                tmp = new;
                tmp.as_ref()
            } else {
                in_buffer
            }
        },
        AlphaColorMode::Premultiplied => {
            let prem = in_buffer.pixels()
                .filter(|px| px.a != 255)
                .map(|px| if px.a == 0 { RGBA8::default() } else { RGBA8::new(
                    (px.r as u16 * 255 / px.a as u16) as u8,
                    (px.r as u16 * 255 / px.a as u16) as u8,
                    (px.r as u16 * 255 / px.a as u16) as u8,
                    px.a,
                )})
                .collect();
            tmp = ImgVec::new(prem, in_buffer.width(), in_buffer.height());
            tmp.as_ref()
        },
    };

    let width = buffer.width();
    let height = buffer.height();
    let mut y_plane = Vec::with_capacity(width*height);
    let mut u_plane = Vec::with_capacity(width*height);
    let mut v_plane = Vec::with_capacity(width*height);
    let mut a_plane = Vec::with_capacity(width*height);
    for px in buffer.pixels() {
        let (y,u,v) = match self.config.color_space {
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

    self.encode_raw_planes_8_bit(width, height, &y_plane, &u_plane, &v_plane, if use_alpha { Some(&a_plane) } else { None }, color_pixel_range)
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
pub fn encode_rgb(&self, buffer: Img<&[RGB8]>) -> Result<EncodedImage, Error> {
    let width = buffer.width();
    let height = buffer.height();
    let mut y_plane = Vec::with_capacity(width*height);
    let mut u_plane = Vec::with_capacity(width*height);
    let mut v_plane = Vec::with_capacity(width*height);
    for px in buffer.pixels() {
        let (y,u,v) = match self.config.color_space {
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

    self.encode_raw_planes_8_bit(width, height, &y_plane, &u_plane, &v_plane, None, color_pixel_range)
}

/// If config.color_space is ColorSpace::YCbCr, then it takes 8-bit BT.709 color space.
///
/// Alpha always uses full range. Chroma subsampling is not supported, and it's a bad idea for AVIF anyway.
///
/// returns AVIF file, size of color metadata, size of alpha metadata overhead
pub fn encode_raw_planes_8_bit(&self, width: usize, height: usize, y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], a_plane: Option<&[u8]>, color_pixel_range: PixelRange) -> Result<EncodedImage, Error> {
    let config = &self.config;

    if y_plane.len() < width * height {
        return Err(Error::TooFewPixels);
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

    let threads = config.threads.map(|threads| {
        if threads > 0 { threads } else { num_cpus::get() }
    });

    // Firefox 81 doesn't support Full yet, but doesn't support alpha either

    let encode_color = move || encode_to_av1(&Av1EncodeConfig {
        width,
        height,
        planes: &[y_plane, u_plane, v_plane],
        quantizer,
        speed: SpeedTweaks::from_my_preset(config.speed, config.quality as _),
        threads,
        pixel_range: color_pixel_range,
        chroma_sampling: ChromaSampling::Cs444,
        color_description,
    });
    let encode_alpha = move || a_plane.map(|a| encode_to_av1(&Av1EncodeConfig {
        width,
        height,
        planes: &[a],
        quantizer: alpha_quantizer,
        speed: SpeedTweaks::from_my_preset(config.speed, config.alpha_quality as _),
        threads,
        pixel_range: PixelRange::Full,
        chroma_sampling: ChromaSampling::Cs400,
        color_description: None,
    }));
    #[cfg(all(target_arch="wasm32", not(target_feature = "atomics")))]
    let (color, alpha) = (encode_color(), encode_alpha());
    #[cfg(not(all(target_arch="wasm32", not(target_feature = "atomics"))))]
    let (color, alpha) = rayon::join(encode_color, encode_alpha);
    let (color, alpha) = (color?, alpha.transpose()?);

    let avif_file = avif_serialize::Aviffy::new()
        .premultiplied_alpha(config.premultiplied_alpha)
        .to_vec(&color, alpha.as_deref(), width as u32, height as u32, 8);
    let color_byte_size = color.len();
    let alpha_byte_size = alpha.as_ref().map_or(0, |a| a.len());

    Ok(EncodedImage {
        avif_file, color_byte_size, alpha_byte_size,
    })
}
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
        let low_quality = quality < 55;
        let high_quality = quality > 80;
        let max_block_size = if high_quality { 16 } else { 64 };

        Self {
            speed_preset: speed,

            partition_range: Some(match speed {
                0 => (4, 64.min(max_block_size)),
                1 if low_quality => (4, 64.min(max_block_size)),
                2 if low_quality => (4, 32.min(max_block_size)),
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
            rdo_tx_decision: Some(speed <= 4 && !high_quality), // it tends to blur subtle textures
            reduced_tx_set: Some(speed == 4 || speed >= 9), // It interacts with tx_domain_distortion too?

            // 4px blocks disabled at 5

            fine_directional_intra: Some(speed <= 6),
            fast_deblock: Some(speed >= 7 && !high_quality), // mixed bag?

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
            } * if high_quality { 2 } else { 1 },
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
    /// 0 means num_cpus
    pub threads: Option<usize>,
    pub pixel_range: PixelRange,
    pub chroma_sampling: ChromaSampling,
    pub color_description: Option<ColorDescription>,
}

fn encode_to_av1(p: &Av1EncodeConfig<'_>) -> Result<Vec<u8>, Error> {
    // AV1 needs all the CPU power you can give it,
    // except when it'd create inefficiently tiny tiles
    let tiles = {
        let threads = p.threads.unwrap_or_else(rayon::current_num_threads);
        threads.min((p.width * p.height) / (p.speed.min_tile_size as usize).pow(2))
    };
    let bit_depth = 8;

    let speed_settings = p.speed.speed_settings();
    let mut cfg = Config::new()
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

    if let Some(threads) = p.threads {
        cfg = cfg.with_threads(threads);
    }

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

#[deprecated(note = "use Encoder::new().encode_rgba(…)")]
#[cold]
pub fn encode_rgba(buffer: Img<&[RGBA8]>, config: &EncConfig) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let res = Encoder { config: *config, high_bit_depth: false, alpha_color_mode: AlphaColorMode::UnassociatedDirty }.encode_rgba(buffer)?;
    Ok((res.avif_file, res.color_byte_size, res.alpha_byte_size))
}

#[deprecated(note = "use Encoder::new().encode_rgb(…)")]
#[cold]
pub fn encode_rgb(buffer: Img<&[RGB8]>, config: &EncConfig) -> Result<(Vec<u8>, usize), Box<dyn std::error::Error + Send + Sync>> {
    let res = Encoder { config: *config, high_bit_depth: false, alpha_color_mode: AlphaColorMode::UnassociatedDirty }.encode_rgb(buffer)?;
    Ok((res.avif_file, res.color_byte_size))
}

#[deprecated(note = "use Encoder::new().encode_raw_planes(…)")]
#[cold]
pub fn encode_raw_planes(width: usize, height: usize, y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], a_plane: Option<&[u8]>, color_pixel_range: PixelRange, config: &EncConfig) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let res = Encoder { config: *config, high_bit_depth: false, alpha_color_mode: AlphaColorMode::UnassociatedDirty }.encode_raw_planes_8_bit(width, height, y_plane, u_plane, v_plane, a_plane, color_pixel_range)?;
    Ok((res.avif_file, res.color_byte_size, res.alpha_byte_size))
}
