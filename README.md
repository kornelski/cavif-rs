# `cavif` — PNG/JPEG to AVIF converter

Encoder/converter for AVIF images. Based on [rav1e](//lib.rs/crates/rav1e) and [avif-serialize](https://lib.rs/crates/avif-serialize), which makes it an almost pure-Rust tool (it uses C LCMS2 for color profiles).

## Installation

➡️ **[Download the latest release](https://github.com/kornelski/cavif/releases)** ⬅️

The pre-built zip includes a portable static executable, with no dependencies, that runs on any Linux distro. It also includes executables for macOS and Windows.

## Usage

Run in a terminal (hint: you don't need to type the path, terminals accept file drag'n'drop)

```bash
cavif image.png
```

It makes `image.avif`. You can adjust quality (it's in 1-100 scale):

```bash
cavif --quality 60 image.png
```

### Advanced usage

You can also specify multiple images. Encoding is multi-threaded, so the more, the better!

```text
cavif [OPTIONS] IMAGES...
```

 * `--quality=n` — Quality from 1 (worst) to 100 (best), the default value is 80. The numbers are only a rough approximation of JPEG's quality scale. [Beware when comparing codecs](https://kornel.ski/faircomparison). There is no lossless compression support, 100 just gives unreasonably bloated files.
 * `--speed=n` — Encoding speed between 1 (best, but slowest) and 10 (fastest, but a blurry mess), the default value is 4. Speeds 1 and 2 are unbelievably slow, but make files ~3-5% smaller. Speeds 7 and above degrade compression significantly, and are not recommended.
 * `--overwrite` — Replace files if there's `.avif` already. By default the existing files are left untouched.
 * `-o path` — Write images to this path (instead of `same-name.avif`). If multiple input files are specified, it's interpreted as a directory.
 * `--quiet` — Don't print anything during conversion.

There are additional options that tweak AVIF color space. The defaults in `cavif` are chosen to be the best, so use these options only when you know it's necessary:

 * `--dirty-alpha` — Preserve RGB values of fully transparent pixels (not recommended). By default irrelevant color of transparent pixels is cleared to avoid wasting space.
 * `--color=rgb` — Encode using RGB instead of YCbCr color space. Makes colors closer to lossless, but makes files larger. Use only if you need to avoid even smallest color shifts.
 * `--depth=8` — Encode using 8-bit color depth instead of 10-bit. This results in a slightly worse quality/compression ratio, but is more compatible.

## Compatibility

Images [work in all modern browsers](https://caniuse.com/avif).

* Chrome 85+ desktop,
* Chrome on Android 12,
* Firefox 91,
* Safari iOS 16/macOS Ventura.

### Known incompatibilities

* Windows' preview and very old versions of android are reported to show pink line at the right edge. This is probably a bug in an old AVIF decoder they use.
* Windows' preview doesn't seem to support 10-bit deep images. Use `--depth=8` when encoding if this is a problem.

## Building

To build it from source you need Rust 1.67 or later, preferably via [rustup](https://rustup.rs).

Then run in a terminal:

```bash
rustup update
cargo install cavif
```
