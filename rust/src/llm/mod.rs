pub mod model;
pub mod qwen2;
pub mod text_generation;
pub mod token_output_stream;

pub const BASE_TEMPLATE: &str = "
<|im_start|>system
{system}
<|im_end|>

<|im_start|>user
{user}
<|im_end|>

<|im_start|>assistant
";
