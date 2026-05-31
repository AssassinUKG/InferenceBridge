//! Strict AgentAction extraction and validation.
//!
//! This is the deterministic gate between noisy model text and an
//! orchestrator-owned action loop. The model may propose an action, but callers
//! should only execute it after this module extracts, repairs, and validates it.

use serde::{Deserialize, Serialize};

use crate::models::profiles::ThinkTagStyle;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    Planner,
    Worker,
    Reviewer,
    Summariser,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgentNextStep {
    Continue,
    Retry,
    AskUser,
    Finish,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentAction {
    pub step_id: String,
    pub role: AgentRole,
    pub goal: String,
    pub action: String,
    pub arguments: serde_json::Value,
    pub expected_outcome: String,
    pub success_check: String,
    pub confidence: f64,
    pub next_step: AgentNextStep,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AgentActionValidation {
    pub valid: bool,
    pub action: Option<AgentAction>,
    pub repaired_json: Option<serde_json::Value>,
    pub visible_text: String,
    pub errors: Vec<String>,
}

pub fn extract_first_json_value(text: &str) -> Option<String> {
    let start = text.find(|ch: char| ch == '{' || ch == '[')?;
    let opener = text.as_bytes()[start];
    let closer = if opener == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, byte) in text[start..].bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if byte == b'\\' && in_string {
            escaped = true;
            continue;
        }
        match byte {
            b'"' => in_string = !in_string,
            value if !in_string && value == opener => depth += 1,
            value if !in_string && value == closer => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + offset + 1].to_string());
                }
            }
            _ => {}
        }
    }

    Some(text[start..].to_string())
}

pub fn validate_agent_action_value(value: serde_json::Value) -> AgentActionValidation {
    let mut errors = Vec::new();
    let action = match serde_json::from_value::<AgentAction>(value.clone()) {
        Ok(action) => Some(action),
        Err(error) => {
            errors.push(format!("schema: {error}"));
            None
        }
    };

    if let Some(action) = action.as_ref() {
        if uuid::Uuid::parse_str(&action.step_id).is_err() {
            errors.push("step_id must be a UUID".to_string());
        }
        if action.goal.trim().is_empty() {
            errors.push("goal must not be empty".to_string());
        }
        if action.action.trim().is_empty() {
            errors.push("action must not be empty".to_string());
        }
        if !action.arguments.is_object() {
            errors.push("arguments must be a JSON object".to_string());
        }
        if action.expected_outcome.trim().is_empty() {
            errors.push("expected_outcome must not be empty".to_string());
        }
        if action.success_check.trim().is_empty() {
            errors.push("success_check must not be empty".to_string());
        }
        if !(0.0..=1.0).contains(&action.confidence) {
            errors.push("confidence must be between 0.0 and 1.0".to_string());
        }
    }

    AgentActionValidation {
        valid: action.is_some() && errors.is_empty(),
        action,
        repaired_json: Some(value),
        visible_text: String::new(),
        errors,
    }
}

pub fn extract_repair_validate_agent_action(
    raw: &str,
    think_tag_style: ThinkTagStyle,
) -> AgentActionValidation {
    let visible_text =
        crate::normalize::think_strip::strip_think_tags_with_style(raw, think_tag_style);
    let Some(candidate) = extract_first_json_value(&visible_text) else {
        return AgentActionValidation {
            valid: false,
            action: None,
            repaired_json: None,
            visible_text,
            errors: vec!["no JSON object or array found".to_string()],
        };
    };

    let Some(value) = crate::normalize::json_repair::repair_json(&candidate) else {
        return AgentActionValidation {
            valid: false,
            action: None,
            repaired_json: None,
            visible_text,
            errors: vec!["JSON could not be repaired".to_string()],
        };
    };

    let mut result = validate_agent_action_value(value);
    result.visible_text = visible_text;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_action_json() -> String {
        serde_json::json!({
            "step_id": uuid::Uuid::new_v4().to_string(),
            "role": "worker",
            "goal": "inspect a file",
            "action": "read_file",
            "arguments": { "path": "src/main.rs" },
            "expected_outcome": "file content is returned",
            "success_check": "content length is greater than zero",
            "confidence": 0.91,
            "next_step": "continue"
        })
        .to_string()
    }

    #[test]
    fn extracts_action_from_noisy_text() {
        let raw = format!("I will do it:\n{}\nDone", valid_action_json());
        let result = extract_repair_validate_agent_action(&raw, ThinkTagStyle::Qwen);
        assert!(result.valid, "{:?}", result.errors);
        assert_eq!(result.action.unwrap().action, "read_file");
    }

    #[test]
    fn strips_think_before_validation() {
        let raw = format!("<think>private</think>{}", valid_action_json());
        let result = extract_repair_validate_agent_action(&raw, ThinkTagStyle::Qwen);
        assert!(result.valid, "{:?}", result.errors);
    }

    #[test]
    fn rejects_hallucinated_non_schema_text() {
        let result = extract_repair_validate_agent_action(
            "I will call magic_fix_everything",
            ThinkTagStyle::Qwen,
        );
        assert!(!result.valid);
    }

    #[test]
    fn rejects_bad_confidence() {
        let raw = serde_json::json!({
            "step_id": uuid::Uuid::new_v4().to_string(),
            "role": "worker",
            "goal": "inspect a file",
            "action": "read_file",
            "arguments": {},
            "expected_outcome": "file content is returned",
            "success_check": "content length is greater than zero",
            "confidence": 2.0,
            "next_step": "continue"
        })
        .to_string();
        let result = extract_repair_validate_agent_action(&raw, ThinkTagStyle::Qwen);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("confidence")));
    }

    #[test]
    fn rejects_non_object_arguments() {
        let raw = serde_json::json!({
            "step_id": uuid::Uuid::new_v4().to_string(),
            "role": "worker",
            "goal": "inspect a file",
            "action": "read_file",
            "arguments": "src/main.rs",
            "expected_outcome": "file content is returned",
            "success_check": "content length is greater than zero",
            "confidence": 0.8,
            "next_step": "continue"
        })
        .to_string();
        let result = extract_repair_validate_agent_action(&raw, ThinkTagStyle::Qwen);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("arguments")));
    }
}
