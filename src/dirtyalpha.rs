use imgref::ImgRef;
use imgref::ImgVec;
use rgb::ComponentMap;
use rgb::RGB;
use rgb::RGBA8;

fn weighed_pixel(px: RGBA8) -> (u8, RGB<u32>) {
    if px.a == 0 {
        return (0, RGB::new(0,0,0))
    }
    let weight = 255 - px.a;
    (weight, RGB::new(
        px.r as u32 * weight as u32,
        px.g as u32 * weight as u32,
        px.b as u32 * weight as u32))
}

pub fn cleared_alpha(mut img: ImgVec<RGBA8>) -> ImgVec<RGBA8> {
    // get dominant visible transparent color (excluding opaque pixels)
    let (weights, sum) = img.pixels()
        .filter_map(|px| {
            if px.a != 255 && px.a != 0 {
                Some(weighed_pixel(px))
            } else {
                None
            }
        })
        .fold((0u64, RGB::new(0,0,0)), |mut sum, item| {
            sum.0 += item.0 as u64;
            sum.1 += item.1.map(|c| c as u64);
            sum
        });
    if weights == 0 {
        return img; // opaque image
    }

    let neutral_alpha = RGBA8::new((sum.r / weights) as u8, (sum.g / weights) as u8, (sum.b / weights) as u8, 0);
    img.pixels_mut().filter(|px| px.a == 0).for_each(|px| *px = neutral_alpha);
    let img2 = bleed_opaque_color(img.as_ref());
    drop(img);
    blur_transparent_pixels(img2.as_ref())
}

/// copy color from opaque pixels to transparent pixels
/// (so that when edges get crushed by compression, the distortion will be away from visible edge)
fn bleed_opaque_color(img: ImgRef<RGBA8>) -> ImgVec<RGBA8> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        out.push(if mid.curr.a == 255 {
            mid.curr
        } else {
            let (weights, sum) = [
                weighed_pixel(top.prev),
                weighed_pixel(top.curr),
                weighed_pixel(top.next),
                weighed_pixel(mid.prev),
                weighed_pixel(mid.curr),
                weighed_pixel(mid.next),
                weighed_pixel(bot.prev),
                weighed_pixel(bot.curr),
                weighed_pixel(bot.next),
            ].iter().fold((0u32, RGB::new(0,0,0)), |mut sum, item| {
                sum.0 += item.0 as u32;
                sum.1 += item.1;
                sum
            });
            if weights != 0 {
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
            } else {
                mid.curr
            }
        });
    });
    ImgVec::new(out, img.width(), img.height())
}

/// ensure there are no sharp edges created by the cleared alpha
fn blur_transparent_pixels(img: ImgRef<RGBA8>) -> ImgVec<RGBA8> {
    let mut out = Vec::with_capacity(img.width() * img.height());
    loop9::loop9_img(img, |_, _, top, mid, bot| {
        out.push(if mid.curr.a == 255 {
            mid.curr
        } else {
            let sum =
                top.prev.rgb().map(|c| c as u16) +
                top.curr.rgb().map(|c| c as u16) +
                top.next.rgb().map(|c| c as u16) +
                mid.prev.rgb().map(|c| c as u16) +
                mid.curr.rgb().map(|c| c as u16) +
                mid.next.rgb().map(|c| c as u16) +
                bot.prev.rgb().map(|c| c as u16) +
                bot.curr.rgb().map(|c| c as u16) +
                bot.next.rgb().map(|c| c as u16);
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
    ImgVec::new(out, img.width(), img.height())
}

fn clamp(px: u8, (min, max): (u8, u8)) -> u8 {
    px.max(min).min(max)
}

/// safe range to change px color given its alpha
/// (mostly-transparent colors tolerate more variation)
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
