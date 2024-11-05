use std::sync::{Arc, RwLock};

use actix_web::{web, App, HttpServer};
use candle_transformers::models::qwen2::ModelForCausalLM;
use controllers::sse;
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;
use tokio::sync::Notify;

use crate::{
    frb_generated::StreamSink,
    llm::{
        model::{Model, ModelRun},
        qwen2::TOKIO_QWEN_MODEL,
        text_generation::TextGeneration,
        token_output_stream::TokenOutputStream,
    },
};

mod controllers;
pub mod models;

pub static SERVER_STATE_SINK: RwLock<Option<StreamSink<String>>> = RwLock::new(None);

static STOP_NOTIFY: Lazy<Arc<Notify>> = Lazy::new(|| Arc::new(Notify::new()));

pub async fn load_model(model_path: Option<String>) -> anyhow::Result<()> {
    let p = if model_path.is_some() {
        model_path.unwrap()
    } else {
        r"D:\github_repo\ai_tools\rust\assets\Qwen2___5-0___5B-Instruct".to_owned()
    };

    let mut global_model = TOKIO_QWEN_MODEL.write().await;
    let device = candle_core::Device::cuda_if_available(0)?;
    let mut model = Model::<ModelForCausalLM>::new(p);
    model.load()?;
    let tokenizer =
        Tokenizer::from_file(&model.tokenizer_path.clone().unwrap()).map_err(anyhow::Error::msg)?;

    let pipeline = TextGeneration::<
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
    );

    *global_model = Some(pipeline);

    anyhow::Ok(())
}

pub fn stop_server() {
    STOP_NOTIFY.notify_one();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        crate::llm::clear_all_models_async().await;
    });

    if let Some(s) = SERVER_STATE_SINK.read().unwrap().as_ref() {
        let _ = s.add("none".to_string());
    }
}

#[allow(unused_assignments)]
pub async fn start_server(model_path: Option<String>, port: Option<usize>) -> std::io::Result<()> {
    let port = if port.is_some() { port.unwrap() } else { 8080 };
    if let Some(s) = SERVER_STATE_SINK.read().unwrap().as_ref() {
        let _ = s.add("init".to_string());
    }

    let r = load_model(model_path).await;
    if let Some(s) = SERVER_STATE_SINK.read().unwrap().as_ref() {
        let _ = s.add("model loaded".to_string());
    }
    match r {
        Ok(_) => {
            let server = HttpServer::new(move || App::new().route("/sse", web::post().to(sse)))
                .bind(format!("127.0.0.1:{}", port))?
                .run();

            let notify = STOP_NOTIFY.clone();
            let server_handler = server.handle().clone();

            tokio::spawn(async move {
                notify.notified().await;
                server_handler.stop(true).await;
            });

            server.await
        }
        Err(_e) => {
            panic!("load model failed {:?}", _e);
        }
    }
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
        let r = load_model(None).await;
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
