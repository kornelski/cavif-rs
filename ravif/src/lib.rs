//! ```rust
//! use ravif::*;
//! # fn doit(pixels: &[RGBA8], width: usize, height: usize) -> Result<(), Error> {
//! let res = Encoder::new()
//!     .with_quality(70.)
//!     .with_speed(4)
//!     .encode_rgba(Img::new(pixels, width, height))?;
//! std::fs::write("hello.avif", res.avif_file);
//! # Ok(()) }

mod av1encoder;

mod error;
pub use av1encoder::ColorModel;
pub use error::Error;

#[doc(hidden)]
#[deprecated = "Renamed to `ColorModel`"]
pub use ColorModel as ColorSpace;

pub use av1encoder::{AlphaColorMode, BitDepth, EncodedImage, Encoder};
#[doc(inline)]
pub use rav1e::prelude::MatrixCoefficients;

mod dirtyalpha;

#[doc(no_inline)]
pub use imgref::Img;
#[doc(no_inline)]
pub use rgb::{RGB16, RGB8, RGBA16, RGBA8};

#[cfg(not(feature = "threading"))]
mod rayoff {
    pub fn current_num_threads() -> usize {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1)
    }

    pub fn join<A, B>(a: impl FnOnce() -> A, b: impl FnOnce() -> B) -> (A, B) {
        (a(), b())
    }
}
