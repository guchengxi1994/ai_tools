use std::time::Duration;

use actix_web::web::Bytes;
use actix_web::{web, Error, HttpResponse};
// use async_channel::{bounded, Receiver};
use futures::stream::{self, Stream};
use futures::FutureExt;
use log::error;
use log::info;
use tokio::sync::mpsc::{self, Receiver};
use tokio::sync::Notify;
use tokio::time::sleep;

use crate::llm::qwen2::QWEN_MODEL;
use crate::llm::BASE_TEMPLATE;

use super::models::CompletionRequest;

fn sse_stream(rx: Receiver<String>) -> impl Stream<Item = Result<Bytes, Error>> {
    stream::unfold(rx, |mut rx| async {
        match rx.recv().await {
            Some(content) => {
                let data = format!("data: {}\n\n", content);
                Some((Ok(web::Bytes::from(data)), rx))
            }
            None => {
                error!("[rust-llm] Error2");
                return None;
            } // Stream ends when the channel is closed
        }
    })
}

pub async fn sse(req: web::Json<CompletionRequest>) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<String>(1);

    let _ = tx.send("data: start\n\n".to_string()).await;

    let notify = Notify::new();

    tokio::spawn({
        let tx_clone = tx.clone();
        async move {
            loop {
                // let _ = tx_clone.send("data: [DONE]\n\n".to_string()).await;
                // tokio::time::sleep(Duration::from_millis(100)).await;
                tokio::select! {
                    // 等待线程1的通知
                    _ = notify.notified() => {
                        info!("Thread 2 exiting");
                        break; // 收到通知后退出循环
                    },
                    _ = sleep(Duration::from_secs(1)) => {
                        let _ = tx_clone.send("data: [DONE]\n\n".to_string()).await;
                    },
                }
            }
        }
    });

    tokio::spawn({
        let tx_clone = tx.clone();

        async move {
            {
                let p = BASE_TEMPLATE.replace("{user}", &req.prompt);

                let mut global_model = QWEN_MODEL.write().unwrap();
                let r = global_model
                    .as_mut()
                    .unwrap()
                    .run_in_actix(&p, 1024, tx_clone.clone()).await;
                match r {
                    Ok(_) => {}
                    Err(e) => {
                        error!("[rust-llm] Error: {}", e);
                    }
                }
            }
        }
    });

    return HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(sse_stream(rx));
}
