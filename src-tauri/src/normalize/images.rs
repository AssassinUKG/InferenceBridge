use base64::Engine as _;

const MAX_REMOTE_IMAGE_BYTES: usize = 15 * 1024 * 1024;

fn normalize_base64_bytes(value: &str) -> Result<String, String> {
    let compact = value.chars().filter(|ch| !ch.is_whitespace()).collect::<String>();
    if compact.is_empty() {
        return Err("Image payload must not be empty.".to_string());
    }

    base64::engine::general_purpose::STANDARD
        .decode(compact.as_bytes())
        .map_err(|_| "Image payload is not valid base64.".to_string())?;

    Ok(compact)
}

pub fn normalize_inline_image_payload(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("Image payload must not be empty.".to_string());
    }

    if trimmed.starts_with("blob:") {
        return Err(
            "Clipboard image is still a browser blob URL. Re-paste the image so it is embedded as image data."
                .to_string(),
        );
    }

    if trimmed.starts_with("file:") {
        return Err("File URLs are not supported here. Paste or upload the image instead.".to_string());
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Err(
            "Remote image URLs are only supported through the public API. Paste or upload the image in the desktop UI instead."
                .to_string(),
        );
    }

    if trimmed.starts_with("data:") {
        let (_, payload) = trimmed.split_once(',').ok_or_else(|| {
            "Image data URLs must include a base64 payload after the comma.".to_string()
        })?;
        return normalize_base64_bytes(payload);
    }

    normalize_base64_bytes(trimmed)
}

async fn fetch_remote_image_as_base64(url: &str) -> Result<String, String> {
    let parsed = reqwest::Url::parse(url).map_err(|_| format!("Invalid image URL: {url}"))?;

    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "Unsupported image URL scheme '{}'. Only http/https URLs are supported.",
            parsed.scheme()
        ));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| format!("Could not initialize remote image fetch client: {e}"))?;

    let response = client
        .get(parsed.clone())
        .header(
            reqwest::header::ACCEPT,
            "image/*,application/octet-stream;q=0.9,*/*;q=0.1",
        )
        .send()
        .await
        .map_err(|e| format!("Could not fetch remote image URL '{url}': {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Remote image URL '{url}' returned HTTP {}.",
            response.status()
        ));
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    if !content_type.is_empty()
        && !content_type.starts_with("image/")
        && !content_type.starts_with("application/octet-stream")
    {
        return Err(format!(
            "Remote image URL '{url}' returned unsupported content type '{content_type}'."
        ));
    }

    if let Some(content_length) = response.content_length() {
        if content_length as usize > MAX_REMOTE_IMAGE_BYTES {
            return Err(format!(
                "Remote image URL '{url}' is too large ({content_length} bytes). Max allowed is {MAX_REMOTE_IMAGE_BYTES} bytes."
            ));
        }
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed while downloading remote image URL '{url}': {e}"))?;

    if bytes.is_empty() {
        return Err(format!("Remote image URL '{url}' returned an empty body."));
    }

    if bytes.len() > MAX_REMOTE_IMAGE_BYTES {
        return Err(format!(
            "Remote image URL '{url}' exceeded the max allowed size of {MAX_REMOTE_IMAGE_BYTES} bytes."
        ));
    }

    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

pub async fn normalize_image_payload(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return fetch_remote_image_as_base64(trimmed).await;
    }

    normalize_inline_image_payload(trimmed)
}

#[cfg(test)]
mod tests {
    use super::normalize_inline_image_payload;

    #[test]
    fn strips_data_url_prefix() {
        let payload = normalize_inline_image_payload("data:image/png;base64,QUFBQQ==")
            .expect("data URL should normalize");
        assert_eq!(payload, "QUFBQQ==");
    }

    #[test]
    fn rejects_blob_urls() {
        let error = normalize_inline_image_payload("blob:http://localhost/image")
            .expect_err("blob URLs should be rejected");
        assert!(error.contains("blob URL"));
    }

    #[test]
    fn rejects_invalid_base64() {
        let error = normalize_inline_image_payload("not base64 !!!")
            .expect_err("invalid base64 should be rejected");
        assert!(error.contains("base64"));
    }
}
