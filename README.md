# `cavif` — PNG to AVIF converter

Encoder/converter for AVIF images. Pure Rust, no C dependencies.

## Installation

[Download the latest release](https://github.com/kornelski/cavif/releases), or:

Install Rust 1.46 or later, preferably via [rustup](//rustup.rs), then run in a terminal:

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

```text
cavif [OPTIONS] IMAGES...
```

 * `--quality=n` — Quality from 1 (worst) to 100 (best)
 * `--speed=n` —   Encoding speed from 1 (fast but ugly) to 10 (best)
 * `--overwrite` — Replace files if there's .avif already
 * `-o path` —     Write output to this path instead of samefile.avif
 * `--quiet` —     Don't print anything
