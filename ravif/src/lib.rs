mod av1encoder;

mod error;
pub use error::Error;

#[allow(deprecated)]
pub use av1encoder::encode_raw_planes;
#[allow(deprecated)]
pub use av1encoder::encode_rgb;
#[allow(deprecated)]
pub use av1encoder::encode_rgba;

pub use av1encoder::ColorSpace;
pub use av1encoder::AlphaColorMode;
pub use av1encoder::Encoder;
pub use av1encoder::EncodedImage;

#[allow(deprecated)]
pub use av1encoder::EncConfig as Config;

mod dirtyalpha;
#[allow(deprecated)]
pub use dirtyalpha::cleared_alpha;

#[doc(no_inline)]
pub use imgref::Img;
#[doc(no_inline)]
pub use rgb::{RGB8, RGBA8};
