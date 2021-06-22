# `ravif` — Pure Rust library for AVIF image encoding

Encoder for AVIF images. Based on [rav1e](//lib.rs/rav1e) and [avif-serialize](//lib.rs/avif-serialize).

The API is just a single `encode_rgba` function call that spits an AVIF image.

This library powers [`cavif`](//lib.rs/cavif) encoder. It has encoding configuration specifically tuned for still images, and gives better quality/performance than stock rav1e.
