use std::fs;
type BoxError = Box<dyn std::error::Error + Send + Sync>;
use std::path::PathBuf;
use std::path::Path;
use rayon::prelude::*;
mod av1encoder;
use rgb::RGBA8;
use imgref::ImgVec;

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
    --quality=n Quality from 1 (worst) to 100 (best)
    --speed=n   Encoding speed from 1 (best) to 10 (fast but ugly)
    --overwrite Replace files if there's .avif already
    -o path     Write output to this path instead of samefile.avif
    --quiet     Don't print anything
",
env!("CARGO_PKG_VERSION"));
}

fn run() -> Result<(), BoxError> {
    let mut args = pico_args::Arguments::from_env();

    if args.contains(["-h", "--help"]) {
        help();
        std::process::exit(0);
    }

    let output = args.opt_value_from_os_str(["-o", "--output"], |s| Ok::<_,std::convert::Infallible>(PathBuf::from(s)))?;
    let quality = args.opt_value_from_str(["-Q", "--quality"])?.unwrap_or(80);
    let speed = args.opt_value_from_str(["-s", "--speed"])?.unwrap_or(1);
    let overwrite = args.contains(["-f", "--overwrite"]);
    let quiet = args.contains(["-q", "--quiet"]);

    let mut files = args.free_os()?;
    files.retain(|path| Path::new(&path).extension().map_or(true, |e| e != "avif"));

    if files.is_empty() {
        help();
        return Err("No PNG files specified".into());
    }

    let use_dir = output.is_some() && files.len() > 1;
    if let Some(out) = &output {
        if use_dir {
            fs::create_dir_all(out)?;
        }
    }

    let process = move |path: &Path| -> Result<(), BoxError> {
        let img = load_rgba(&path)?;
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
            return Err(format!("{} already exists; skipping", out_path.display()).into())
        }
        let (buffer, width, height) = img.into_contiguous_buf();
        let (out_data, color_size, alpha_size) = av1encoder::encode_rgba(width, height, &buffer, quality, speed)?;
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
fn load_rgba(path: &Path) -> Result<ImgVec<RGBA8>, BoxError> {
    let img = lodepng::decode32_file(&path)?;
    Ok(ImgVec::new(img.buffer, img.width, img.height))
}

#[cfg(feature = "cocoa_image")]
fn load_rgba(path: &Path) -> Result<ImgVec<RGBA8>, BoxError> {
    let data = fs::read(path)?;
    Ok(cocoa_image::decode_image_as_rgba(&data)?)
}

