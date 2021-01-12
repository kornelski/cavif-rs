use clap::{Arg, App, AppSettings, value_t};
use imgref::ImgVec;
use rayon::prelude::*;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

use ravif::*;

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

enum MaybePath {
    Stdio,
    Path(PathBuf),
}

fn run() -> Result<(), BoxError> {
    let args = App::new("cavif-rs")
        .version(clap::crate_version!())
        .author("Kornel Lesiński <kornel@imageoptim.com>")
        .about("Convert JPEG/PNG images to AVIF image format (based on AV1/rav1e)")
        .setting(AppSettings::DeriveDisplayOrder)
        .setting(AppSettings::ColorAuto)
        .setting(AppSettings::UnifiedHelpMessage)
        .arg(Arg::with_name("quality")
            .short("Q")
            .long("quality")
            .value_name("n")
            .help("Quality from 1 (worst) to 100 (best)")
            .default_value("80")
            .takes_value(true))
        .arg(Arg::with_name("speed")
            .short("s")
            .long("speed")
            .value_name("n")
            .default_value("1")
            .help("Encoding speed from 1 (best) to 10 (fast but ugly)")
            .takes_value(true))
        .arg(Arg::with_name("overwrite")
            .alias("--force")
            .short("f")
            .long("overwrite")
            .help("Replace files if there's .avif already"))
        .arg(Arg::with_name("output")
            .short("o")
            .long("output")
            .value_name("path")
            .help("Write output to this path instead of same_file.avif. It may be a file or a directory.")
            .takes_value(true))
        .arg(Arg::with_name("quiet")
            .short("q")
            .long("quiet")
            .help("Don't print anything"))
        .arg(Arg::with_name("dirty-alpha")
            .long("dirty-alpha")
            .help("Keep RGB data of fully-transparent pixels (makes larger, lower quality files)"))
        .arg(Arg::with_name("color")
            .long("color")
            .default_value("ycbcr")
            .takes_value(true)
            .possible_values(&["ycbcr", "rgb"])
            .help("Internal AVIF color space"))
        .arg(Arg::with_name("IMAGES")
            .index(1)
            .help("One or more JPEG or PNG files to convert. \"-\" is interpreted as stdin/stdout.")
            .multiple(true))
        .get_matches();

    let output = args.value_of("output").map(|s| {
        match s {
            s if s == "-" => MaybePath::Stdio,
            s => MaybePath::Path(PathBuf::from(s)),
        }
    });
    let quality: u8 = value_t!(args, "quality", u8)?;
    let alpha_quality = ((quality + 100)/2).min(quality + quality/4 + 2);
    let speed: u8 = value_t!(args, "speed", u8)?;
    let overwrite = args.is_present("overwrite");
    let quiet = args.is_present("quiet");
    let premultiplied_alpha = args.is_present("premultiplied-alpha");
    let dirty_alpha = args.is_present("dirty-alpha");
    if dirty_alpha && premultiplied_alpha {
        return Err("premultiplied alpha option makes dirty alpha impossible".into());
    }

    let color_space = match args.value_of("color").expect("default") {
        "ycbcr" => ColorSpace::YCbCr,
        "rgb" => ColorSpace::RGB,
        x => Err(format!("bad color type: {}", x))?,
    };
    let files: Vec<_> = args.values_of_os("IMAGES").ok_or("Please specify image paths to convert")?
        .filter(|path| Path::new(&path).extension().map_or(true, |e| e != "avif"))
        .map(|p| if p == "-" {
            MaybePath::Stdio
        } else {
            MaybePath::Path(PathBuf::from(p))
        })
        .collect();

    if files.is_empty() {
        args.usage();
        return Err("No PNG/JPEG files specified".into());
    }

    let use_dir = match output {
        Some(MaybePath::Path(ref path)) => {
            if files.len() > 1 {
                let _ = fs::create_dir_all(path);
            }
            files.len() > 1 || path.is_dir()
        },
        _ => false,
    };

    let process = move |data: Vec<u8>, input_path: &MaybePath| -> Result<(), BoxError> {
        let mut img = load_rgba(&data, premultiplied_alpha)?;
        drop(data);
        let out_path = match (&output, input_path) {
            (None, MaybePath::Path(input)) => MaybePath::Path(input.with_extension("avif")),
            (Some(MaybePath::Path(output)), MaybePath::Path(ref input)) => MaybePath::Path({
                if use_dir {
                    output.join(Path::new(input.file_name().unwrap()).with_extension("avif"))
                } else {
                    output.to_owned()
                }
            }),
            (None, MaybePath::Stdio) |
            (Some(MaybePath::Stdio), _) => MaybePath::Stdio,
            (Some(MaybePath::Path(output)), MaybePath::Stdio) => MaybePath::Path(output.to_owned()),
        };
        match out_path {
            MaybePath::Path(ref p) if !overwrite && p.exists() => {
                return Err(format!("{} already exists; skipping", p.display()).into());
            },
            _ => {},
        }
        if !dirty_alpha && !premultiplied_alpha {
            img = cleared_alpha(img);
        }
        let (out_data, color_size, alpha_size) = encode_rgba(img.as_ref(), &Config {
            quality, speed,
            alpha_quality, premultiplied_alpha,
            color_space,
        })?;
        match out_path {
            MaybePath::Path(ref p) => {
                if !quiet {
                    println!("{}: {}KB ({}B color, {}B alpha, {}B HEIF)", p.display(), (out_data.len()+999)/1000, color_size, alpha_size, out_data.len() - color_size - alpha_size);
                }
                fs::write(p, out_data)
            },
            MaybePath::Stdio => {
                std::io::stdout().write_all(&out_data)
            },
        }.map_err(|e| format!("Unable to write output image: {}", e))?;
        Ok(())
    };

    let failures = files.into_par_iter().map(|path| {
        let tmp;
        let (data, path_str): (_, &dyn std::fmt::Display) = match path {
            MaybePath::Stdio => {
                let mut data = Vec::new();
                std::io::stdin().read_to_end(&mut data)?;
                (data, &"stdin")
            },
            MaybePath::Path(ref path) => {
                let data = fs::read(path)
                    .map_err(|e| format!("Unable to read input image {}: {}", path.display(), e))?;
                tmp = path.display();
                (data, &tmp)
            },
        };
        process(data, &path)
            .map_err(|e| BoxError::from(format!("{}: error: {}", path_str, e)))
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
