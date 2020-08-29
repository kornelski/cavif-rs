# `cavif` — PNG/JPEG to AVIF converter

Encoder/converter for AVIF images. Based on [rav1e](//lib.rs/rav1e) and [avif-serialize](//lib.rs/avif-serialize), which makes it pure Rust, with no C code!

## Installation

➡️ **[Download the latest release](https://github.com/kornelski/cavif/releases)** ⬅️

Alternatively, build it from source. It requires:

* Rust 1.45 or later, preferably via [rustup](//rustup.rs)
* [`nasm`](https://www.nasm.us/) 2.14 or later

Then run in a terminal:

```bash
cargo install cavif
```

## Running

Run in a terminal (hint: you don't need to type the path, terminals accept file drag'n'drop)

```bash
cavif image.png
```

It makes `image.avif`. You can adjust quality (it's in 1-100 scale):

```bash
cavif --quality 60 image.png
```

### Usage

You can also specify multiple images (encoding is multi-threaded, so the more, the better!).

```text
cavif [OPTIONS] IMAGES...
```

 * `--quality=n` — Quality from 1 (worst) to 100 (best), the default value is 80. The numbers have different meaning than JPEG's quality scale. [Beware when comparing codecs](https://kornel.ski/faircomparison).
 * `--speed=n` — Encoding speed between 1 (best, but slowest) and 10 (fastest, but a blurry mess), the default value is 1. Encoding of AVIF is pretty slow, so you need either a) beefy multicore machine b) avoid large images c) patience.
 * `--overwrite` — Replace files if there's .avif already. By default existing files are left untouched.
 * `-o path` — Write output to this path instead of samefile.avif. If multiple input files are specified, it's interpreted as a directory.
 * `--quiet` — Don't print anything during conversion.
 * `--premultiplied-alpha` — Warning: [currently incompatible with libavif](https://github.com/AOMediaCodec/libavif/issues/292). Improves compression of transparent images by clearing RGB of fully transparent pixels and lowering quality of semi-transparent colors.
 * `--dirty-alpha` — Don't change RGB values of transparent pixels. By default irrelevant color of transparent pixels is cleared to avoid wasting space.

