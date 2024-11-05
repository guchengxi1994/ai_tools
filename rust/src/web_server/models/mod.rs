#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletionResponse {
    pub text: String,
}

impl CompletionResponse {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
