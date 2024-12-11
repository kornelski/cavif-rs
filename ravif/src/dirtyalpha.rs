use imgref::Img;
use imgref::ImgRef;
use rav1e::Pixel;
use rgb::ComponentMap;
use rgb::Rgb;
use rgb::Rgba;

#[inline]
fn weighed_pixel<P: Pixel + Default>(px: Rgba<P>) -> (P, Rgb<P>) {
    if px.a == P::cast_from(0) {
        return (px.a, Rgb::new(px.a, px.a, px.a));
    }
    let weight = P::cast_from(256) - px.a;
    (
        weight,
        Rgb::new(px.r * weight, px.g * weight, px.b * weight),
    )
}

/// Clear/change RGB components of fully-transparent RGBA pixels to make them cheaper to encode with AV1
pub(crate) fn blurred_dirty_alpha<P: Pixel + Default>(
    img: ImgRef<Rgba<P>>,
) -> Option<Img<Vec<Rgba<P>>>> {
    // get dominant visible transparent color (excluding opaque pixels)
    let mut sum = Rgb::new(0, 0, 0);
    let mut weights = 0;

    // Only consider colors around transparent images
    // (e.g. solid semitransparent area doesn't need to contribute)
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        if mid.curr.a == P::cast_from(255) || mid.curr.a == P::cast_from(0) {
            return;
        }
        if chain(&top, &mid, &bot).any(|px| px.a == P::cast_from(0)) {
            let (w, px) = weighed_pixel(mid.curr);
            weights += Into::<u32>::into(w) as u64;
            sum += Rgb::new(
                Into::<u32>::into(px.r) as u64,
                Into::<u32>::into(px.g) as u64,
                Into::<u32>::into(px.b) as u64,
            );
        }
    });
    if weights == 0 {
        return None; // opaque image
    }

    let neutral_alpha = Rgba::new(
        P::cast_from((sum.r / weights) as u8),
        P::cast_from((sum.g / weights) as u8),
        P::cast_from((sum.b / weights) as u8),
        P::cast_from(0),
    );
    let img2 = bleed_opaque_color(img, neutral_alpha);
    Some(blur_transparent_pixels(img2.as_ref()))
}

/// copy color from opaque pixels to transparent pixels
/// (so that when edges get crushed by compression, the distortion will be away from visible edge)
fn bleed_opaque_color<P: Pixel + Default>(img: ImgRef<Rgba<P>>, bg: Rgba<P>) -> Img<Vec<Rgba<P>>> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        out.push(if mid.curr.a == P::cast_from(255) {
            mid.curr
        } else {
            let (weights, sum) = chain(&top, &mid, &bot).map(|c| weighed_pixel(*c)).fold(
                (
                    0u32,
                    Rgb::new(P::cast_from(0), P::cast_from(0), P::cast_from(0)),
                ),
                |mut sum, item| {
                    sum.0 += Into::<u32>::into(item.0);
                    sum.1 += item.1;
                    sum
                },
            );
            if weights == 0 {
                bg
            } else {
                let mut avg = sum.map(|c| P::cast_from(Into::<u32>::into(c) / weights));
                if mid.curr.a == P::cast_from(0) {
                    avg.with_alpha(mid.curr.a)
                } else {
                    // also change non-transparent colors, but only within range where
                    // rounding caused by premultiplied alpha would land on the same color
                    avg.r = clamp(avg.r, premultiplied_minmax(mid.curr.r, mid.curr.a));
                    avg.g = clamp(avg.g, premultiplied_minmax(mid.curr.g, mid.curr.a));
                    avg.b = clamp(avg.b, premultiplied_minmax(mid.curr.b, mid.curr.a));
                    avg.with_alpha(mid.curr.a)
                }
            }
        });
    });
    Img::new(out, img.width(), img.height())
}

/// ensure there are no sharp edges created by the cleared alpha
fn blur_transparent_pixels<P: Pixel + Default>(img: ImgRef<Rgba<P>>) -> Img<Vec<Rgba<P>>> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        out.push(if mid.curr.a == P::cast_from(255) {
            mid.curr
        } else {
            let sum: Rgb<P> = chain(&top, &mid, &bot).map(|px| px.rgb()).sum();
            let mut avg = sum.map(|c| (c / P::cast_from(9)));
            if mid.curr.a == P::cast_from(0) {
                avg.with_alpha(mid.curr.a)
            } else {
                // also change non-transparent colors, but only within range where
                // rounding caused by premultiplied alpha would land on the same color
                avg.r = clamp(avg.r, premultiplied_minmax(mid.curr.r, mid.curr.a));
                avg.g = clamp(avg.g, premultiplied_minmax(mid.curr.g, mid.curr.a));
                avg.b = clamp(avg.b, premultiplied_minmax(mid.curr.b, mid.curr.a));
                avg.with_alpha(mid.curr.a)
            }
        });
    });
    Img::new(out, img.width(), img.height())
}

#[inline(always)]
fn chain<'a, T>(
    top: &'a loop9::Triple<T>,
    mid: &'a loop9::Triple<T>,
    bot: &'a loop9::Triple<T>,
) -> impl Iterator<Item = &'a T> + 'a {
    top.iter().chain(mid.iter()).chain(bot.iter())
}

#[inline]
fn clamp<P: Pixel + Default>(px: P, (min, max): (P, P)) -> P {
    P::cast_from(
        Into::<u32>::into(px)
            .max(Into::<u32>::into(min))
            .min(Into::<u32>::into(max)),
    )
}

/// safe range to change px color given its alpha
/// (mostly-transparent colors tolerate more variation)
#[inline]
fn premultiplied_minmax<P, T>(px: P, alpha: T) -> (P, T)
where
    P: Pixel + Default,
    T: Pixel + Default,
{
    let alpha = Into::<u32>::into(alpha);
    let rounded = Into::<u32>::into(px) * alpha / 255 * 255;

    // leave some spare room for rounding
    let low = (rounded + 16) / alpha;
    let hi = (rounded + 239) / alpha;

    (
        P::cast_from(low).min(px),
        T::cast_from(hi).max(T::cast_from(Into::<u32>::into(px))),
    )
}

#[test]
fn preminmax() {
    assert_eq!((100, 100), premultiplied_minmax(100, 255));
    assert_eq!((78, 100), premultiplied_minmax(100, 10));
    assert_eq!(100 * 10 / 255, 78 * 10 / 255);
    assert_eq!(100 * 10 / 255, 100 * 10 / 255);
    assert_eq!((8, 119), premultiplied_minmax(100, 2));
    assert_eq!((16, 239), premultiplied_minmax(100, 1));
    assert_eq!((15, 255), premultiplied_minmax(255, 1));
}
