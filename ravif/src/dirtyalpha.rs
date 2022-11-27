use imgref::Img;
use imgref::ImgRef;
use rgb::ComponentMap;
use rgb::RGB;
use rgb::RGBA8;

#[inline]
fn weighed_pixel(px: RGBA8) -> (u16, RGB<u32>) {
    if px.a == 0 {
        return (0, RGB::new(0,0,0))
    }
    let weight = 256 - px.a as u16;
    (weight, RGB::new(
        px.r as u32 * weight as u32,
        px.g as u32 * weight as u32,
        px.b as u32 * weight as u32))
}

/// Clear/change RGB components of fully-transparent RGBA pixels to make them cheaper to encode with AV1
#[deprecated(note = "use Encoder::with_color_alpha_mode(_::UnassociatedClean) instead")]
#[cold]
pub fn cleared_alpha(img: Img<Vec<RGBA8>>) -> Img<Vec<RGBA8>> {
    blurred_dirty_alpha(img.as_ref()).unwrap_or(img)
}

pub(crate) fn blurred_dirty_alpha(img: ImgRef<RGBA8>) -> Option<Img<Vec<RGBA8>>> {
    // get dominant visible transparent color (excluding opaque pixels)
    let mut sum = RGB::new(0, 0, 0);
    let mut weights = 0;

    // Only consider colors around transparent images
    // (e.g. solid semitransparent area doesn't need to contribute)
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        if mid.curr.a == 255 || mid.curr.a == 0 {
            return;
        }
        if chain(&top, &mid, &bot).any(|px| px.a == 0) {
            let (w, px) = weighed_pixel(mid.curr);
            weights += w as u64;
            sum += px.map(|c| c as u64);
        }
    });
    if weights == 0 {
        return None; // opaque image
    }

    let neutral_alpha = RGBA8::new((sum.r / weights) as u8, (sum.g / weights) as u8, (sum.b / weights) as u8, 0);
    let img2 = bleed_opaque_color(img, neutral_alpha);
    Some(blur_transparent_pixels(img2.as_ref()))
}

/// copy color from opaque pixels to transparent pixels
/// (so that when edges get crushed by compression, the distortion will be away from visible edge)
fn bleed_opaque_color(img: ImgRef<RGBA8>, bg: RGBA8) -> Img<Vec<RGBA8>> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        out.push(if mid.curr.a == 255 {
            mid.curr
        } else {
            let (weights, sum) = chain(&top, &mid, &bot)
                .map(|c| weighed_pixel(*c))
                .fold((0u32, RGB::new(0,0,0)), |mut sum, item| {
                    sum.0 += item.0 as u32;
                    sum.1 += item.1;
                    sum
                });
            if weights == 0 {
                bg
            } else {
                let mut avg = sum.map(|c| (c / weights) as u8);
                if mid.curr.a == 0 {
                    avg.alpha(0)
                } else {
                    // also change non-transparent colors, but only within range where
                    // rounding caused by premultiplied alpha would land on the same color
                    avg.r = clamp(avg.r, premultiplied_minmax(mid.curr.r, mid.curr.a));
                    avg.g = clamp(avg.g, premultiplied_minmax(mid.curr.g, mid.curr.a));
                    avg.b = clamp(avg.b, premultiplied_minmax(mid.curr.b, mid.curr.a));
                    avg.alpha(mid.curr.a)
                }
            }
        });
    });
    Img::new(out, img.width(), img.height())
}

/// ensure there are no sharp edges created by the cleared alpha
fn blur_transparent_pixels(img: ImgRef<RGBA8>) -> Img<Vec<RGBA8>> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        out.push(if mid.curr.a == 255 {
            mid.curr
        } else {
            let sum: RGB<u16> =
                chain(&top, &mid, &bot).map(|px| px.rgb().map(|c| c as u16)).sum();
            let mut avg = sum.map(|c| (c / 9) as u8);
            if mid.curr.a == 0 {
                avg.alpha(0)
            } else {
                // also change non-transparent colors, but only within range where
                // rounding caused by premultiplied alpha would land on the same color
                avg.r = clamp(avg.r, premultiplied_minmax(mid.curr.r, mid.curr.a));
                avg.g = clamp(avg.g, premultiplied_minmax(mid.curr.g, mid.curr.a));
                avg.b = clamp(avg.b, premultiplied_minmax(mid.curr.b, mid.curr.a));
                avg.alpha(mid.curr.a)
            }
        });
    });
    Img::new(out, img.width(), img.height())
}

#[inline(always)]
fn chain<'a, T>(top: &'a loop9::Triple<T>, mid: &'a loop9::Triple<T>, bot: &'a loop9::Triple<T>) -> impl Iterator<Item = &'a T> + 'a {
    top.iter().chain(mid.iter()).chain(bot.iter())
}

#[inline]
fn clamp(px: u8, (min, max): (u8, u8)) -> u8 {
    px.max(min).min(max)
}

/// safe range to change px color given its alpha
/// (mostly-transparent colors tolerate more variation)
#[inline]
fn premultiplied_minmax(px: u8, alpha: u8) -> (u8, u8) {
    let alpha = alpha as u16;
    let rounded = (px as u16) * alpha / 255 * 255;

    // leave some spare room for rounding
    let low = ((rounded + 16) / alpha) as u8;
    let hi = ((rounded + 239) / alpha) as u8;

    (low.min(px), hi.max(px))
}

#[test]
fn preminmax() {
    assert_eq!((100,100), premultiplied_minmax(100, 255));
    assert_eq!((78,100), premultiplied_minmax(100, 10));
    assert_eq!(100*10/255, 78*10/255);
    assert_eq!(100*10/255, 100*10/255);
    assert_eq!((8,119), premultiplied_minmax(100, 2));
    assert_eq!((16,239), premultiplied_minmax(100, 1));
    assert_eq!((15,255), premultiplied_minmax(255, 1));
}
