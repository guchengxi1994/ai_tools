use actix_web::{web, HttpServer};
use candle_transformers::models::qwen2::ModelForCausalLM;
use tokenizers::Tokenizer;

use crate::llm::{
    model::{Model, ModelRun},
    qwen2::QWEN_MODEL,
    text_generation::TextGeneration,
    token_output_stream::TokenOutputStream,
};

mod controllers;
mod models;

pub fn load_model() -> anyhow::Result<()> {
    let mut global_model = QWEN_MODEL.write().unwrap();
    let device = candle_core::Device::cuda_if_available(0)?;
    let mut model = Model::<ModelForCausalLM>::new(
        r"D:\github_repo\ai_tools\rust\assets\Qwen2___5-0___5B-Instruct".to_owned(),
    );
    model.load()?;
    let tokenizer =
        Tokenizer::from_file(&model.tokenizer_path.clone().unwrap()).map_err(anyhow::Error::msg)?;

    let mut pipeline = TextGeneration::<
        Model<ModelForCausalLM>,
        usize,
        crate::llm::token_output_stream::TokenOutputStream,
    >::new(
        model,
        TokenOutputStream::new(tokenizer),
        128,
        Some(0.7),
        Some(0.9),
        1.25,
        64,
        &device,
        None,
    );

    *global_model = Some(pipeline);

    anyhow::Ok(())
}

#[allow(unused_imports)]
mod test {
    use actix_web::{get, web, App, HttpServer, Responder};

    use crate::web_server::{controllers::sse, load_model};

    #[get("/hello")]
    async fn greet() -> impl Responder {
        format!("good bye!")
    }

    #[tokio::test]
    async fn test_server() -> std::io::Result<()> {
        env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
        let r = load_model();
        match r {
            Ok(_) => {
                HttpServer::new(move || {
                    App::new().service(greet).route("/sse", web::post().to(sse))
                })
                .bind("127.0.0.1:8080")?
                .run()
                .await
            }
            Err(_e) => {
                panic!("load model failed {:?}", _e);
            }
        }
    }
}
