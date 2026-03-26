//! JSON repair pipeline — ported from HelixClaw's repair_json().
//! 5-step repair: fast parse → trailing commas → unclosed strings → rebalance braces → salvage.

/// Attempt to repair malformed JSON and return the parsed value.
pub fn repair_json(input: &str) -> Option<serde_json::Value> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Step 1: Try direct parse
    if let Ok(v) = serde_json::from_str(trimmed) {
        return Some(v);
    }

    // Step 2: Remove trailing commas
    let no_trailing = remove_trailing_commas(trimmed);
    if let Ok(v) = serde_json::from_str(&no_trailing) {
        return Some(v);
    }

    // Step 3: Close unclosed strings
    let closed_strings = close_unclosed_strings(&no_trailing);
    if let Ok(v) = serde_json::from_str(&closed_strings) {
        return Some(v);
    }

    // Step 4: Rebalance braces/brackets
    let balanced = rebalance_braces(&closed_strings);
    if let Ok(v) = serde_json::from_str(&balanced) {
        return Some(v);
    }

    // Step 5: Extract first JSON object/array
    extract_first_json(trimmed)
}

fn remove_trailing_commas(s: &str) -> String {
    let re = regex::Regex::new(r",\s*([}\]])").unwrap();
    re.replace_all(s, "$1").to_string()
}

fn close_unclosed_strings(s: &str) -> String {
    let mut result = s.to_string();
    let mut in_string = false;
    let mut escaped = false;
    let mut last_quote_pos = 0;

    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            if in_string {
                in_string = false;
            } else {
                in_string = true;
                last_quote_pos = i;
            }
        }
    }

    if in_string {
        // Find a good place to close — before any trailing brace/bracket
        let after_quote = &result[last_quote_pos + 1..];
        if let Some(pos) = after_quote.rfind(|c: char| c == '}' || c == ']') {
            result.insert(last_quote_pos + 1 + pos, '"');
        } else {
            result.push('"');
        }
    }
    result
}

fn rebalance_braces(s: &str) -> String {
    let mut result = s.to_string();
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        match ch {
            '"' => in_string = !in_string,
            '{' if !in_string => open_braces += 1,
            '}' if !in_string => open_braces -= 1,
            '[' if !in_string => open_brackets += 1,
            ']' if !in_string => open_brackets -= 1,
            _ => {}
        }
    }

    for _ in 0..open_brackets.max(0) {
        result.push(']');
    }
    for _ in 0..open_braces.max(0) {
        result.push('}');
    }
    result
}

fn extract_first_json(s: &str) -> Option<serde_json::Value> {
    // Try to find the first { or [ and extract a balanced JSON value
    let start = s.find(|c: char| c == '{' || c == '[')?;
    let opener = s.as_bytes()[start];
    let closer = if opener == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (i, ch) in s[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        match ch {
            '"' => in_string = !in_string,
            c if !in_string && c as u8 == opener => depth += 1,
            c if !in_string && c as u8 == closer => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &s[start..start + i + 1];
                    return serde_json::from_str(candidate).ok();
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_trailing_comma() {
        let input = r#"{"name": "test", "args": {"x": 1,}}"#;
        let result = repair_json(input);
        assert!(result.is_some());
    }

    #[test]
    fn repairs_unclosed_brace() {
        let input = r#"{"name": "test", "args": {"x": 1}"#;
        let result = repair_json(input);
        assert!(result.is_some());
    }
}
