//! Wide gamut and HDR signalling: `colr`, `clli`, `mdcv`, AV1 color config and metadata OBUs,
//! and sample round-trips through an AV1 decoder.

use ravif::*;
use re_rav1d::dav1d::{self, pixel, PlanarImageComponent};

const BT2020_PRIMARIES: [(f32, f32); 3] = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)];
const D65: (f32, f32) = (0.3127, 0.3290);

fn encoder() -> Encoder<'static> {
    Encoder::new().with_quality(90.).with_alpha_quality(90.).with_speed(10).with_num_threads(Some(1))
}

// ---------------------------------------------------------------------------
// Minimal ISOBMFF walk
// ---------------------------------------------------------------------------

fn be16(d: &[u8]) -> u16 {
    u16::from_be_bytes([d[0], d[1]])
}

fn be32(d: &[u8]) -> u32 {
    u32::from_be_bytes([d[0], d[1], d[2], d[3]])
}

fn children(mut data: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    while data.len() >= 8 {
        let size = be32(data) as usize;
        let fourcc = data[4..8].try_into().unwrap();
        let (header, size) = match size {
            0 => (8, data.len()),
            1 => (16, u64::from_be_bytes(data[8..16].try_into().unwrap()) as usize),
            _ => (8, size),
        };
        out.push((fourcc, &data[header..size]));
        data = &data[size..];
    }
    out
}

fn child<'a>(data: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    children(data).into_iter().find(|(f, _)| f == fourcc).map(|(_, d)| d)
}

/// Properties associated with the primary item, as `(fourcc, payload)`
fn primary_item_properties(file: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    let meta = &child(file, b"meta").expect("meta")[4..];
    let primary_id = u32::from(be16(&child(meta, b"pitm").expect("pitm")[4..]));
    let iprp = child(meta, b"iprp").expect("iprp");
    let ipco = children(child(iprp, b"ipco").expect("ipco"));
    let ipma = child(iprp, b"ipma").expect("ipma");

    let version = ipma[0];
    let large_index = ipma[3] & 1 != 0;
    let mut pos = 4;
    let entry_count = be32(&ipma[pos..]);
    pos += 4;
    for _ in 0..entry_count {
        let item_id = if version < 1 {
            pos += 2;
            u32::from(be16(&ipma[pos - 2..]))
        } else {
            pos += 4;
            be32(&ipma[pos - 4..])
        };
        let count = ipma[pos];
        pos += 1;
        let mut props = Vec::new();
        for _ in 0..count {
            let index = if large_index {
                pos += 2;
                usize::from(be16(&ipma[pos - 2..]) & 0x7FFF)
            } else {
                pos += 1;
                usize::from(ipma[pos - 1] & 0x7F)
            };
            let (fourcc, payload) = ipco[index - 1];
            props.push((fourcc, payload.to_vec()));
        }
        if item_id == primary_id {
            return props;
        }
    }
    panic!("primary item has no properties");
}

fn property<'a>(props: &'a [([u8; 4], Vec<u8>)], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    props.iter().find(|(f, _)| f == fourcc).map(|(_, d)| d.as_slice())
}

/// `(primaries, transfer, matrix, full_range)`
fn nclx(props: &[([u8; 4], Vec<u8>)]) -> Option<(u16, u16, u16, bool)> {
    let colr = property(props, b"colr")?;
    assert_eq!(&colr[..4], b"nclx");
    Some((be16(&colr[4..]), be16(&colr[6..]), be16(&colr[8..]), colr[10] & 0x80 != 0))
}

/// `(seq_profile, bit_depth)`
fn av1c(props: &[([u8; 4], Vec<u8>)]) -> (u8, u8) {
    let av1c = property(props, b"av1C").expect("av1C");
    let high_bitdepth = av1c[2] & 0x40 != 0;
    let twelve_bit = av1c[2] & 0x20 != 0;
    (av1c[1] >> 5, if twelve_bit { 12 } else if high_bitdepth { 10 } else { 8 })
}

fn pixi_depth(props: &[([u8; 4], Vec<u8>)]) -> u8 {
    let pixi = property(props, b"pixi").expect("pixi");
    assert_eq!(pixi[4], 3);
    pixi[5]
}

// ---------------------------------------------------------------------------
// Minimal AV1 OBU walk
// ---------------------------------------------------------------------------

fn leb128(data: &[u8], pos: &mut usize) -> usize {
    let mut value = 0;
    for i in 0..8 {
        let byte = data[*pos];
        *pos += 1;
        value |= usize::from(byte & 0x7F) << (i * 7);
        if byte & 0x80 == 0 {
            break;
        }
    }
    value
}

/// Payloads of metadata OBUs, keyed by metadata type (1 = HDR CLL, 2 = HDR MDCV)
fn metadata_obus(av1: &[u8]) -> Vec<(usize, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < av1.len() {
        let header = av1[pos];
        pos += 1;
        if header & 0x04 != 0 {
            pos += 1; // extension header
        }
        assert!(header & 0x02 != 0, "OBUs must have a size field");
        let size = leb128(av1, &mut pos);
        let payload = &av1[pos..pos + size];
        pos += size;
        if (header >> 3) & 0x0F == 5 {
            let mut p = 0;
            let metadata_type = leb128(payload, &mut p);
            out.push((metadata_type, payload[p..].to_vec()));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

fn decode(av1: &[u8]) -> dav1d::Picture {
    let mut settings = dav1d::Settings::new();
    settings.set_n_threads(1);
    settings.set_max_frame_delay(1);
    let mut decoder = dav1d::Decoder::with_settings(&settings).unwrap();
    decoder.send_data(av1.to_vec(), None, None, None).unwrap();
    match decoder.get_picture() {
        Ok(picture) => picture,
        Err(dav1d::Error::Again) => {
            decoder.flush();
            decoder.get_picture().unwrap()
        },
        Err(err) => panic!("{err:?}"),
    }
}

/// Mean sample value of a plane
fn plane_mean(picture: &dav1d::Picture, component: PlanarImageComponent) -> f64 {
    let plane = picture.plane(component);
    let stride = picture.stride(component) as usize;
    let (width, height) = (picture.width() as usize, picture.height() as usize);
    let samples: Vec<f64> = (0..height)
        .flat_map(|y| {
            let row = &plane[y * stride..];
            (0..width).map(move |x| if picture.bit_depth() > 8 { f64::from(u16::from_ne_bytes([row[2 * x], row[2 * x + 1]])) } else { f64::from(row[x]) })
        })
        .collect();
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Inverse of the H.273 YCbCr quantization, back to RGB code values
fn ycbcr_to_rgb(picture: &dav1d::Picture, [kr, kg, kb]: [f64; 3]) -> [f64; 3] {
    let depth = picture.bit_depth() as i32;
    let max = f64::from((1 << depth) - 1);
    let s = f64::from(1 << (depth - 8));
    let [y, cb, cr] = [PlanarImageComponent::Y, PlanarImageComponent::U, PlanarImageComponent::V].map(|c| plane_mean(picture, c));
    let (y, cb, cr) = match picture.color_range() {
        pixel::YUVRange::Full => (y, cb - f64::from(1 << (depth - 1)), cr - f64::from(1 << (depth - 1))),
        pixel::YUVRange::Limited => ((y - 16. * s) / (219. * s) * max, (cb - 128. * s) / (224. * s) * max, (cr - 128. * s) / (224. * s) * max),
    };
    let r = y + 2. * (1. - kr) * cr;
    let b = y + 2. * (1. - kb) * cb;
    let g = (y - kr * r - kb * b) / kg;
    [r, g, b]
}

fn assert_close(actual: [f64; 3], expected: [u16; 3], tolerance: f64) {
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - f64::from(e)).abs() <= tolerance, "decoded {actual:?}, expected {expected:?} +/- {tolerance}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn bt2100_pq_10_bit_with_hdr_metadata() {
    let rgb = RGB16::new(620, 480, 300);
    let img = Img::new(vec![rgb; 32 * 24], 32, 24);
    let encoded = encoder()
        .with_color_description(ColorDescription::BT2100_PQ)
        .with_content_light_level(ContentLightLevel { max_content_light_level: 1000, max_frame_average_light_level: 400 })
        .with_mastering_display(MasteringDisplay {
            primaries: BT2020_PRIMARIES,
            white_point: D65,
            max_luminance: 1000.,
            min_luminance: 0.0001,
        })
        .encode_rgb16(img.as_ref(), 10)
        .unwrap();
    assert_eq!(encoded.alpha_byte_size, 0);

    let props = primary_item_properties(&encoded.avif_file);
    assert_eq!(nclx(&props), Some((9, 16, 9, true)));
    assert_eq!(av1c(&props), (1, 10));
    assert_eq!(pixi_depth(&props), 10);

    let clli = property(&props, b"clli").expect("clli");
    assert_eq!((be16(clli), be16(&clli[2..])), (1000, 400));

    // ST 2086 order (G, B, R), 0.00002 chromaticity units, 0.0001 cd/m^2 luminance units
    let mdcv = property(&props, b"mdcv").expect("mdcv");
    let mdcv_u16: Vec<u16> = mdcv[..16].chunks(2).map(be16).collect();
    assert_eq!(mdcv_u16, [8500, 39850, 6550, 2300, 35400, 14600, 15635, 16450]);
    assert_eq!((be32(&mdcv[16..]), be32(&mdcv[20..])), (10_000_000, 1));

    let parsed = avif_parse::read_avif(&mut encoded.avif_file.as_slice()).unwrap();
    let metadata = metadata_obus(&parsed.primary_item);
    let cll = &metadata.iter().find(|(t, _)| *t == 1).expect("HDR CLL metadata OBU").1;
    assert_eq!((be16(cll), be16(&cll[2..])), (1000, 400));
    // AV1 order (R, G, B), 0.16 fixed-point chromaticities, 24.8 max and 18.14 min luminance
    let mdcv = &metadata.iter().find(|(t, _)| *t == 2).expect("HDR MDCV metadata OBU").1;
    let fixed = |v: f64| (v * 65536.).round() as u16;
    let mdcv_u16: Vec<u16> = mdcv[..16].chunks(2).map(be16).collect();
    assert_eq!(mdcv_u16, [fixed(0.708), fixed(0.292), fixed(0.170), fixed(0.797), fixed(0.131), fixed(0.046), fixed(0.3127), fixed(0.3290)]);
    assert_eq!((be32(&mdcv[16..]), be32(&mdcv[20..])), (1000 << 8, 2));

    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.bit_depth(), 10);
    assert_eq!(picture.pixel_layout(), dav1d::PixelLayout::I444);
    assert_eq!(picture.color_primaries(), pixel::ColorPrimaries::BT2020);
    assert_eq!(picture.transfer_characteristic(), pixel::TransferCharacteristic::PerceptualQuantizer);
    assert_eq!(picture.matrix_coefficients(), pixel::MatrixCoefficients::BT2020NonConstantLuminance);
    assert_eq!(picture.color_range(), pixel::YUVRange::Full);
    assert_close(ycbcr_to_rgb(&picture, [0.2627, 0.6780, 0.0593]), [rgb.r, rgb.g, rgb.b], 3.);
}

#[test]
fn bt2100_hlg_12_bit_with_alpha() {
    let px = RGBA16::new(2900, 1800, 700, 3000);
    let img = Img::new(vec![px; 24 * 16], 24, 16);
    let encoded = encoder()
        .with_color_description(ColorDescription::BT2100_HLG)
        .with_alpha_color_mode(AlphaColorMode::UnassociatedDirty)
        .encode_rgba16(img.as_ref(), 12)
        .unwrap();
    assert!(encoded.alpha_byte_size > 0);

    let props = primary_item_properties(&encoded.avif_file);
    assert_eq!(nclx(&props), Some((9, 18, 9, true)));
    assert_eq!(av1c(&props), (2, 12));
    assert_eq!(pixi_depth(&props), 12);
    assert!(property(&props, b"clli").is_none());
    assert!(property(&props, b"mdcv").is_none());

    let parsed = avif_parse::read_avif(&mut encoded.avif_file.as_slice()).unwrap();
    assert!(metadata_obus(&parsed.primary_item).is_empty());

    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.bit_depth(), 12);
    assert_eq!(picture.transfer_characteristic(), pixel::TransferCharacteristic::HybridLogGamma);
    assert_eq!(picture.matrix_coefficients(), pixel::MatrixCoefficients::BT2020NonConstantLuminance);
    assert_close(ycbcr_to_rgb(&picture, [0.2627, 0.6780, 0.0593]), [px.r, px.g, px.b], 12.);

    let alpha = decode(parsed.alpha_item.as_deref().expect("alpha item"));
    assert_eq!(alpha.bit_depth(), 12);
    let a = plane_mean(&alpha, PlanarImageComponent::Y);
    assert!((a - 3000.).abs() <= 12., "alpha {a}");
}

#[test]
fn display_p3_10_bit() {
    let rgb = RGB16::new(900, 200, 450);
    let img = Img::new(vec![rgb; 16 * 16], 16, 16);
    let encoded = encoder().with_color_description(ColorDescription::DISPLAY_P3).encode_rgb16(img.as_ref(), 10).unwrap();

    let props = primary_item_properties(&encoded.avif_file);
    assert_eq!(nclx(&props), Some((12, 13, 1, true)));
    assert_eq!(av1c(&props), (1, 10));
    assert!(property(&props, b"clli").is_none());
    assert!(property(&props, b"mdcv").is_none());

    let parsed = avif_parse::read_avif(&mut encoded.avif_file.as_slice()).unwrap();
    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.bit_depth(), 10);
    assert_eq!(picture.color_primaries(), pixel::ColorPrimaries::P3Display);
    assert_eq!(picture.transfer_characteristic(), pixel::TransferCharacteristic::SRGB);
    assert_eq!(picture.matrix_coefficients(), pixel::MatrixCoefficients::BT709);
    assert_close(ycbcr_to_rgb(&picture, [0.2126, 0.7152, 0.0722]), [rgb.r, rgb.g, rgb.b], 3.);
}

#[test]
fn limited_range_and_identity_12_bit() {
    let rgb = RGB16::new(3500, 1234, 400);
    let img = Img::new(vec![rgb; 16 * 16], 16, 16);

    let limited = ColorDescription { pixel_range: PixelRange::Limited, ..ColorDescription::BT2100_PQ };
    let encoded = encoder().with_color_description(limited).encode_rgb16(img.as_ref(), 12).unwrap();
    let props = primary_item_properties(&encoded.avif_file);
    assert_eq!(nclx(&props), Some((9, 16, 9, false)));
    let parsed = avif_parse::read_avif(&mut encoded.avif_file.as_slice()).unwrap();
    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.color_range(), pixel::YUVRange::Limited);
    assert_close(ycbcr_to_rgb(&picture, [0.2627, 0.6780, 0.0593]), [rgb.r, rgb.g, rgb.b], 12.);

    let encoded = encoder()
        .with_color_description(ColorDescription::BT2100_PQ)
        .with_internal_color_model(ColorModel::RGB)
        .encode_rgb16(img.as_ref(), 12)
        .unwrap();
    let props = primary_item_properties(&encoded.avif_file);
    assert_eq!(nclx(&props), Some((9, 16, 0, true)));
    let parsed = avif_parse::read_avif(&mut encoded.avif_file.as_slice()).unwrap();
    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.matrix_coefficients(), pixel::MatrixCoefficients::Identity);
    let [g, b, r] = [PlanarImageComponent::Y, PlanarImageComponent::U, PlanarImageComponent::V].map(|c| plane_mean(&picture, c));
    assert_close([r, g, b], [rgb.r, rgb.g, rgb.b], 12.);
}

#[test]
fn raw_planes_12_bit() {
    let encoded = encoder()
        .with_color_description(ColorDescription::BT2100_HLG)
        .encode_raw_planes_12_bit(16, 16, vec![[2048u16, 1900, 2200]; 16 * 16], None::<[_; 0]>, PixelRange::Full, MatrixCoefficients::BT2020NCL)
        .unwrap();
    let props = primary_item_properties(&encoded.avif_file);
    assert_eq!(nclx(&props), Some((9, 18, 9, true)));
    assert_eq!(av1c(&props), (2, 12));
    let parsed = avif_parse::read_avif(&mut encoded.avif_file.as_slice()).unwrap();
    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.bit_depth(), 12);
    let planes = [PlanarImageComponent::Y, PlanarImageComponent::U, PlanarImageComponent::V].map(|c| plane_mean(&picture, c));
    assert_close(planes, [2048, 1900, 2200], 12.);
}

#[test]
fn default_is_srgb_and_unchanged() {
    let img = Img::new((0..32 * 32).map(|i| RGB8::new(i as u8, (i >> 3) as u8, 200)).collect::<Vec<_>>(), 32, 32);

    let default = encoder().encode_rgb(img.as_ref()).unwrap();
    let explicit = encoder().with_color_description(ColorDescription::SRGB).encode_rgb(img.as_ref()).unwrap();
    assert_eq!(default.avif_file, explicit.avif_file);

    // BT.709/sRGB/BT.601/full is avif-serialize's default, so no `colr` box is written
    let props = primary_item_properties(&default.avif_file);
    assert_eq!(nclx(&props), None);
    assert!(property(&props, b"clli").is_none());

    let parsed = avif_parse::read_avif(&mut default.avif_file.as_slice()).unwrap();
    let picture = decode(&parsed.primary_item);
    assert_eq!(picture.color_primaries(), pixel::ColorPrimaries::BT709);
    assert_eq!(picture.transfer_characteristic(), pixel::TransferCharacteristic::SRGB);
    assert_eq!(picture.bit_depth(), 10);
}

#[test]
fn unsupported_inputs_are_errors() {
    let img = Img::new(vec![RGB16::new(1, 2, 3); 4], 2, 2);
    assert!(matches!(encoder().encode_rgb16(img.as_ref(), 11), Err(Error::Unsupported(_))));
    assert!(matches!(encoder().encode_rgb16(img.as_ref(), 16), Err(Error::Unsupported(_))));

    let xyz = ColorDescription { color_primaries: ColorPrimaries::XYZ, ..ColorDescription::SRGB };
    assert!(matches!(encoder().with_color_description(xyz).encode_rgb16(img.as_ref(), 10), Err(Error::Unsupported(_))));

    let ictcp = ColorDescription { matrix_coefficients: MatrixCoefficients::ICtCp, ..ColorDescription::BT2100_PQ };
    assert!(matches!(encoder().with_color_description(ictcp).encode_rgb16(img.as_ref(), 10), Err(Error::Unsupported(_))));
}
