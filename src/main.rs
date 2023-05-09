use clap::ArgAction;
use clap::value_parser;
use load_image::export::rgb::ComponentMap;
use clap::{Arg, Command};
use imgref::ImgVec;
use ravif::{AlphaColorMode, ColorSpace, Encoder, EncodedImage, RGBA8};
use rayon::prelude::*;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        let mut source = e.source();
        while let Some(e) = source {
            eprintln!("  because: {e}");
            source = e.source();
        }
        std::process::exit(1);
    }
}

enum MaybePath {
    Stdio,
    Path(PathBuf),
}

fn parse_quality(arg: &str) -> Result<f32, String> {
    let q = arg.parse::<f32>().map_err(|e| e.to_string())?;
    if q < 1. || q > 100. {
        return Err("quality must be in 1-100 range".into());
    }
    Ok(q)
}

fn parse_speed(arg: &str) -> Result<u8, String> {
    let s = arg.parse::<u8>().map_err(|e| e.to_string())?;
    if s < 1 || s > 100 {
        return Err("speed must be in 1-10 range".into());
    }
    Ok(s)
}

fn run() -> Result<(), BoxError> {
    let args = Command::new("cavif-rs")
        .version(clap::crate_version!())
        .author("Kornel Lesiński <kornel@imageoptim.com>")
        .about("Convert JPEG/PNG images to AVIF image format (based on AV1/rav1e)")
        .arg(Arg::new("quality")
            .short('Q')
            .long("quality")
            .value_name("n")
            .value_parser(parse_quality)
            .default_value("80")
            .help("Quality from 1 (worst) to 100 (best)"))
        .arg(Arg::new("speed")
            .short('s')
            .long("speed")
            .value_name("n")
            .default_value("4")
            .value_parser(parse_speed)
            .help("Encoding speed from 1 (best) to 10 (fast but ugly)"))
        .arg(Arg::new("threads")
            .short('j')
            .long("threads")
            .value_name("n")
            .default_value("0")
            .value_parser(value_parser!(u8))
            .help("Maximum threads to use (0 = one thread per host core)"))
        .arg(Arg::new("overwrite")
            .alias("force")
            .short('f')
            .long("overwrite")
            .action(ArgAction::SetTrue)
            .num_args(0)
            .help("Replace files if there's .avif already"))
        .arg(Arg::new("output")
            .short('o')
            .long("output")
            .value_parser(value_parser!(PathBuf))
            .value_name("path")
            .help("Write output to this path instead of same_file.avif. It may be a file or a directory."))
        .arg(Arg::new("quiet")
            .short('q')
            .long("quiet")
            .action(ArgAction::SetTrue)
            .num_args(0)
            .help("Don't print anything"))
        .arg(Arg::new("dirty-alpha")
            .long("dirty-alpha")
            .action(ArgAction::SetTrue)
            .num_args(0)
            .help("Keep RGB data of fully-transparent pixels (makes larger, lower quality files)"))
        .arg(Arg::new("color")
            .long("color")
            .default_value("ycbcr")
            .value_parser(["ycbcr", "rgb"])
            .help("Internal AVIF color space. YCbCr works better for human eyes."))
        .arg(Arg::new("depth")
            .long("depth")
            .default_value("auto")
            .value_parser(["8", "10", "auto"])
            .help("Write 8-bit (more compatible) or 10-bit (better quality) images"))
        .arg(Arg::new("IMAGES")
            .index(1)
            .num_args(1..)
            .value_parser(value_parser!(PathBuf))
            .help("One or more JPEG or PNG files to convert. \"-\" is interpreted as stdin/stdout."))
        .get_matches();

    let output = args.get_one::<PathBuf>("output").map(|s| {
        match s {
            s if s.as_os_str() == "-" => MaybePath::Stdio,
            s => MaybePath::Path(PathBuf::from(s)),
        }
    });
    let quality = *args.get_one::<f32>("quality").expect("default");
    let alpha_quality = ((quality + 100.)/2.).min(quality + quality/4. + 2.);
    let speed: u8 = *args.get_one::<u8>("speed").expect("default");
    let overwrite = args.get_flag("overwrite");
    let quiet = args.get_flag("quiet");
    let threads = args.get_one::<u8>("threads").copied();
    let dirty_alpha = args.get_flag("dirty-alpha");

    let color_space = match args.get_one::<String>("color").expect("default").as_str() {
        "ycbcr" => ColorSpace::YCbCr,
        "rgb" => ColorSpace::RGB,
        x => Err(format!("bad color type: {x}"))?,
    };

    let depth = match args.get_one::<String>("depth").expect("default").as_str() {
        "8" => Some(8),
        "10" => Some(10),
        _ => None,
    };

    let files = args.get_many::<PathBuf>("IMAGES").ok_or("Please specify image paths to convert")?;
    let files: Vec<_> = files
        .filter(|pathstr| {
            let path = Path::new(&pathstr);
            if let Some(s) = path.to_str() {
                if quiet && s.parse::<u8>().is_ok() && !path.exists() {
                    eprintln!("warning: -q is not for quality, so '{s}' is misinterpreted as a file. Use -Q {s}");
                }
            }
            path.extension().map_or(true, |e| if e == "avif" {
                if !quiet {
                    if path.exists() {
                        eprintln!("warning: ignoring {}, because it's already an AVIF", path.display());
                    } else {
                        eprintln!("warning: Did you mean to use -o {p}?", p = path.display());
                        return true;
                    }
                }
                false
            } else {
                true
            })
        })
        .map(|p| if p.as_os_str() == "-" {
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
        let img = load_rgba(&data, false)?;
        drop(data);
        let out_path = match (&output, input_path) {
            (None, MaybePath::Path(input)) => MaybePath::Path(input.with_extension("avif")),
            (Some(MaybePath::Path(output)), MaybePath::Path(ref input)) => MaybePath::Path({
                if use_dir {
                    output.join(Path::new(input.file_name().unwrap()).with_extension("avif"))
                } else {
                    output.clone()
                }
            }),
            (None, MaybePath::Stdio) |
            (Some(MaybePath::Stdio), _) => MaybePath::Stdio,
            (Some(MaybePath::Path(output)), MaybePath::Stdio) => MaybePath::Path(output.clone()),
        };
        match out_path {
            MaybePath::Path(ref p) if !overwrite && p.exists() => {
                return Err(format!("{} already exists; skipping", p.display()).into());
            },
            _ => {},
        }
        let enc = Encoder::new()
            .with_quality(quality)
            .with_depth(depth)
            .with_speed(speed)
            .with_alpha_quality(alpha_quality)
            .with_internal_color_space(color_space)
            .with_alpha_color_mode(if dirty_alpha { AlphaColorMode::UnassociatedDirty } else { AlphaColorMode::UnassociatedClean })
            .with_num_threads(threads.filter(|&n| n > 0).map(usize::from));
        let EncodedImage { avif_file, color_byte_size, alpha_byte_size , .. } = enc.encode_rgba(img.as_ref())?;
        match out_path {
            MaybePath::Path(ref p) => {
                if !quiet {
                    println!("{}: {}KB ({color_byte_size}B color, {alpha_byte_size}B alpha, {}B HEIF)", p.display(), (avif_file.len()+999)/1000, avif_file.len() - color_byte_size - alpha_byte_size);
                }
                fs::write(p, avif_file)
            },
            MaybePath::Stdio => {
                std::io::stdout().write_all(&avif_file)
            },
        }.map_err(|e| format!("Unable to write output image: {e}"))?;
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
                    .map_err(|e| format!("Unable to read input image {}: {e}", path.display()))?;
                tmp = path.display();
                (data, &tmp)
            },
        };
        process(data, &path)
            .map_err(|e| BoxError::from(format!("{path_str}: error: {e}")))
    })
    .filter_map(|res| res.err())
    .collect::<Vec<BoxError>>();

    if !failures.is_empty() {
        if !quiet {
            for f in failures {
                eprintln!("error: {f}");
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
            px.r = (u16::from(px.r) * u16::from(px.a) / 255) as u8;
            px.g = (u16::from(px.g) * u16::from(px.a) / 255) as u8;
            px.b = (u16::from(px.b) * u16::from(px.a) / 255) as u8;
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
