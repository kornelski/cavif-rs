use imgref::ImgVec;
use rayon::prelude::*;
use rgb::RGBA8;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

mod av1encoder;
mod dirtyalpha;
use crate::dirtyalpha::cleared_alpha;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        let mut source = e.source();
        while let Some(e) = source {
            eprintln!("  because: {}", e);
            source = e.source();
        }
        std::process::exit(1);
    }
}

fn help() {
    println!("cavif {} 08.2020 by Kornel Lesiński. https://lib.rs/cavif

Usage:
    cavif [OPTIONS] IMAGES...

Options:
    --quality=n   Quality from 1 (worst) to 100 (best)
    --speed=n     Encoding speed from 1 (best) to 10 (fast but ugly)
    --overwrite   Replace files if there's .avif already
    -o path       Write output to this path instead of samefile.avif
    --quiet       Don't print anything
    --dirty-alpha Keep RGB colors of fully-transparent pixels
",
        env!("CARGO_PKG_VERSION")
    );
}

fn run() -> Result<(), BoxError> {
    let mut args = pico_args::Arguments::from_env();

    if args.contains(["-h", "--help"]) {
        help();
        std::process::exit(0);
    }

    let output = args.opt_value_from_os_str(["-o", "--output"], |s| {
        Ok::<_, std::convert::Infallible>(PathBuf::from(s))
    })?;
    let quality = args.opt_value_from_str(["-Q", "--quality"])?.unwrap_or(80);
    let alpha_quality = ((quality + 100)/2).min(quality + quality/4 + 2);
    let speed = args.opt_value_from_str(["-s", "--speed"])?.unwrap_or(1);
    let overwrite = args.contains(["-f", "--overwrite"]);
    let quiet = args.contains(["-q", "--quiet"]);
    let premultiplied_alpha = args.contains("--premultiplied-alpha");
    let dirty_alpha = args.contains("--dirty-alpha");
    if dirty_alpha && premultiplied_alpha {
        return Err("premultiplied alpha option makes dirty alpha impossible".into());
    }

    let mut files = args.free_os()?;
    files.retain(|path| Path::new(&path).extension().map_or(true, |e| e != "avif"));

    if files.is_empty() {
        help();
        return Err("No PNG/JPEG files specified".into());
    }

    let use_dir = output.is_some() && files.len() > 1;
    if let Some(out) = &output {
        if use_dir {
            fs::create_dir_all(out)?;
        }
    }

    let process = move |path: &Path| -> Result<(), BoxError> {
        let data = fs::read(&path)?;
        let mut img = load_rgba(&data, premultiplied_alpha)?;
        drop(data);
        let out_path = if let Some(output) = &output {
            if use_dir {
                let file = Path::new(path.file_name().unwrap()).with_extension("avif");
                output.join(file)
            } else {
                output.to_owned()
            }
        } else {
            path.with_extension("avif")
        };
        if !overwrite && out_path.exists() {
            return Err(format!("{} already exists; skipping", out_path.display()).into());
        }
        if !dirty_alpha && !premultiplied_alpha {
            img = cleared_alpha(img);
        }
        let (buffer, width, height) = img.into_contiguous_buf();
        let (out_data, color_size, alpha_size) = av1encoder::encode_rgba(width, height, &buffer, &av1encoder::EncConfig {
            quality, speed,
            alpha_quality, premultiplied_alpha
        })?;
        if !quiet {
            println!("{}: {}KB ({}B color, {}B alpha, {}B HEIF)", out_path.display(), (out_data.len()+999)/1000, color_size, alpha_size, out_data.len() - color_size - alpha_size);
        }
        fs::write(out_path, out_data)?;
        Ok(())
    };

    let failures = files.par_iter().map(|path| {
        let path = Path::new(&path);
        process(path).map_err(|e| -> BoxError {
            format!("{}: error: {}", path.display(), e).into()
        })
    })
    .filter_map(|res| res.err())
    .collect::<Vec<BoxError>>();

    if !failures.is_empty() {
        if !quiet {
            for f in failures {
                eprintln!("{}", f);
            }
        }
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(not(feature = "cocoa_image"))]
fn load_rgba(mut data: &[u8], premultiplied_alpha: bool) -> Result<ImgVec<RGBA8>, BoxError> {
    use rgb::FromSlice;

    let mut img = if data.get(0..4) == Some(&[0x89,b'P',b'N',b'G']) {
        let img = lodepng::decode32(data)?;
        ImgVec::new(img.buffer, img.width, img.height)
    } else {
        let mut jecoder = jpeg_decoder::Decoder::new(&mut data);
        let pixels = jecoder.decode()?;
        let info = jecoder.info().ok_or("Error reading JPEG info")?;
        use jpeg_decoder::PixelFormat::*;
        let buf: Vec<_> = match info.pixel_format {
            L8 => {
                pixels.iter().copied().map(|g| RGBA8::new(g,g,g,255)).collect()
            },
            RGB24 => {
                let rgb = pixels.as_rgb();
                rgb.iter().map(|p| p.alpha(255)).collect()
            },
            CMYK32 => return Err("CMYK JPEG is not supported. Please convert to PNG first".into()),
        };
        ImgVec::new(buf, info.width.into(), info.height.into())
    };
    if premultiplied_alpha {
        img.pixels_mut().for_each(|px| {
            px.r = (px.r as u16 * px.a as u16 / 255) as u8;
            px.g = (px.g as u16 * px.a as u16 / 255) as u8;
            px.b = (px.b as u16 * px.a as u16 / 255) as u8;
        });
    }
    Ok(img)
}

#[cfg(feature = "cocoa_image")]
fn load_rgba(data: &[u8], premultiplied_alpha: bool) -> Result<ImgVec<RGBA8>, BoxError> {
    if premultiplied_alpha {
        Ok(cocoa_image::decode_image_as_rgba_premultiplied(data)?)
    } else {
        Ok(cocoa_image::decode_image_as_rgba(data)?)
    }
}
