pub fn qwen2_chat(user_prompt: String, system_prompt: Option<String>, model_path: String) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _ = rt.block_on(async {
        crate::llm::qwen2::qwen2_chat(user_prompt, system_prompt, model_path)?;
        anyhow::Ok(())
    });
}
