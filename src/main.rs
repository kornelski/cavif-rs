use load_image::export::rgb::ComponentMap;
use clap::{Arg, Command, AppSettings};
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
    let args = Command::new("cavif-rs")
        .version(clap::crate_version!())
        .author("Kornel Lesiński <kornel@imageoptim.com>")
        .about("Convert JPEG/PNG images to AVIF image format (based on AV1/rav1e)")
        .setting(AppSettings::DeriveDisplayOrder)
        .arg(Arg::new("quality")
            .short('Q')
            .long("quality")
            .value_name("n")
            .help("Quality from 1 (worst) to 100 (best). NB! Not the same scale as JPEG quality")
            .default_value("80")
            .takes_value(true))
        .arg(Arg::new("speed")
            .short('s')
            .long("speed")
            .value_name("n")
            .default_value("4")
            .help("Encoding speed from 0 (best) to 10 (fast but ugly)")
            .takes_value(true))
        .arg(Arg::new("threads")
            .short('j')
            .long("threads")
            .value_name("n")
            .default_value("0")
            .help("Maximum threads to use (0 = one thread per host core)")
            .takes_value(true))
        .arg(Arg::new("overwrite")
            .alias("--force")
            .short('f')
            .long("overwrite")
            .help("Replace files if there's .avif already"))
        .arg(Arg::new("output")
            .short('o')
            .long("output")
            .allow_invalid_utf8(true)
            .value_name("path")
            .help("Write output to this path instead of same_file.avif. It may be a file or a directory.")
            .takes_value(true))
        .arg(Arg::new("quiet")
            .short('q')
            .long("quiet")
            .help("Don't print anything"))
        .arg(Arg::new("dirty-alpha")
            .long("dirty-alpha")
            .help("Keep RGB data of fully-transparent pixels (makes larger, lower quality files)"))
        .arg(Arg::new("color")
            .long("color")
            .default_value("ycbcr")
            .takes_value(true)
            .possible_values(&["ycbcr", "rgb"])
            .help("Internal AVIF color space"))
        .arg(Arg::new("IMAGES")
            .index(1)
            .allow_invalid_utf8(true)
            .min_values(1)
            .help("One or more JPEG or PNG files to convert. \"-\" is interpreted as stdin/stdout.")
            .multiple_occurrences(true))
        .get_matches();

    let output = args.value_of_os("output").map(|s| {
        match s {
            s if s == "-" => MaybePath::Stdio,
            s => MaybePath::Path(PathBuf::from(s)),
        }
    });
    let quality = args.value_of_t::<f32>("quality")?;
    let alpha_quality = ((quality + 100.)/2.).min(quality + quality/4. + 2.);
    let speed: u8 = args.value_of_t::<u8>("speed")?;
    let overwrite = args.is_present("overwrite");
    let quiet = args.is_present("quiet");
    let threads = Some(args.value_of_t::<usize>("threads")?);
    let dirty_alpha = args.is_present("dirty-alpha");

    let color_space = match args.value_of("color").expect("default") {
        "ycbcr" => ColorSpace::YCbCr,
        "rgb" => ColorSpace::RGB,
        x => Err(format!("bad color type: {}", x))?,
    };
    let files = args.values_of_os("IMAGES").ok_or("Please specify image paths to convert")?;
    let files: Vec<_> = files
        .filter(|pathstr| {
            let path = Path::new(&pathstr);
            if let Some(s) = path.to_str() {
                if quiet && s.parse::<u8>().is_ok() && !path.exists() {
                    eprintln!("warning: -q is not for quality, so '{s}' is misinterpreted as a file. Use -Q {s}", s = s);
                }
            }
            path.extension().map_or(true, |e| if e == "avif" {
                if !quiet { eprintln!("warning: ignoring {}", path.display()); }
                false
            } else {
                true
            })
        })
        .map(|p| if p == "-" {
            MaybePath::Stdio
        } else {
            MaybePath::Path(PathBuf::from(p))
        })
        .collect();

    if files.is_empty() {
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
        let mut img = load_rgba(&data, false)?;
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
        if !dirty_alpha {
            img = cleared_alpha(img);
        }
        let (out_data, color_size, alpha_size) = encode_rgba(img.as_ref(), &Config {
            quality, speed,
            alpha_quality,
            premultiplied_alpha: false,
            color_space, threads,
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
fn load_rgba(data: &[u8], premultiplied_alpha: bool) -> Result<ImgVec<RGBA8>, BoxError> {

    let img = load_image::load_data(data)?.into_imgvec();
    let mut img = match img {
        load_image::export::imgref::ImgVecKind::RGB8(img) => img.map_buf(|buf| buf.into_iter().map(|px| px.alpha(255)).collect()),
        load_image::export::imgref::ImgVecKind::RGBA8(img) => img,
        load_image::export::imgref::ImgVecKind::RGB16(img) => img.map_buf(|buf| buf.into_iter().map(|px| px.map(|c| (c >> 8) as u8).alpha(255)).collect()),
        load_image::export::imgref::ImgVecKind::RGBA16(img) => img.map_buf(|buf| buf.into_iter().map(|px| px.map(|c| (c >> 8) as u8)).collect()),
        load_image::export::imgref::ImgVecKind::GRAY8(img) => img.map_buf(|buf| buf.into_iter().map(|g| { let c = g.0; RGBA8::new(c,c,c,255) }).collect()),
        load_image::export::imgref::ImgVecKind::GRAY16(img) => img.map_buf(|buf| buf.into_iter().map(|g| { let c = (g.0>>8) as u8; RGBA8::new(c,c,c,255) }).collect()),
        load_image::export::imgref::ImgVecKind::GRAYA8(img) => img.map_buf(|buf| buf.into_iter().map(|g| { let c = g.0; RGBA8::new(c,c,c,g.1) }).collect()),
        load_image::export::imgref::ImgVecKind::GRAYA16(img) => img.map_buf(|buf| buf.into_iter().map(|g| { let c = (g.0>>8) as u8; RGBA8::new(c,c,c,(g.1>>8) as u8) }).collect()),
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
