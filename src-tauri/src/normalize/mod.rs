//! Output normalization pipeline.
//!
//! Pipeline: raw tokens -> think-strip -> model-parser -> json-repair -> tool-extract -> validate

pub mod agent_action;
pub mod capability_truth;
pub mod events;
pub mod images;
pub mod json_repair;
pub mod parse_trace;
pub mod think_strip;
pub mod tool_extract;
