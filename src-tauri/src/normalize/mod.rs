//! Output normalization pipeline.
//!
//! Pipeline: raw tokens -> think-strip -> model-parser -> json-repair -> tool-extract -> validate

pub mod images;
pub mod json_repair;
pub mod think_strip;
pub mod tool_extract;
