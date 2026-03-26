//! Built-in chat templates as raw strings.
//! These can be overridden by user-provided template files.

/// ChatML template (used by Qwen, Phi, and many others).
pub const CHATML: &str = r#"<|im_start|>{{role}}
{{content}}<|im_end|>
"#;

/// Llama 3 chat template.
pub const LLAMA3: &str = r#"<|start_header_id|>{{role}}<|end_header_id|>

{{content}}<|eot_id|>"#;
