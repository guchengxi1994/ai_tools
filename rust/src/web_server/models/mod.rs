#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletionResponse {
    pub text: String,
}
