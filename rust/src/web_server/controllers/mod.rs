use std::sync::Arc;
use std::time::Duration;

use actix_web::web::Bytes;
use actix_web::{web, Error, HttpResponse};
use futures::stream::{self, Stream};
use log::error;
use log::info;
use tokio::sync::mpsc::{self, Receiver};
use tokio::sync::Notify;
use tokio::time::sleep;

use crate::constant::DEFAULT_SYSTEM_ROLE;
use crate::llm::qwen2::TOKIO_QWEN_MODEL;
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
                info!("[rust-llm] stream done");
                return None;
            } // Stream ends when the channel is closed
        }
    })
}

pub async fn sse(req: web::Json<CompletionRequest>) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<String>(1);

    // let _ = tx.send("data: start\n\n".to_string()).await;

    let notify = Arc::new(Notify::new());

    tokio::spawn({
        let tx_clone = tx.clone();
        let notify_clone = notify.clone();
        async move {
            loop {
                // let _ = tx_clone.send("data: [DONE]\n\n".to_string()).await;
                // tokio::time::sleep(Duration::from_millis(100)).await;
                tokio::select! {
                    // 等待线程1的通知
                    _ = notify_clone.notified() => {
                        info!("Thread 2 exiting");
                        break; // 收到通知后退出循环
                    },
                    _ = sleep(Duration::from_secs(1)) => {
                        let _ = tx_clone.send("".to_string()).await;
                    },
                }
            }
        }
    });

    tokio::spawn({
        let tx_clone = tx.clone();
        let notify_clone = notify.clone();
        async move {
            {
                let p = BASE_TEMPLATE
                    .replace("{user}", &req.prompt)
                    .replace("{system}", DEFAULT_SYSTEM_ROLE);

                let mut global_model = TOKIO_QWEN_MODEL.write().await;
                let r = global_model
                    .as_mut()
                    .unwrap()
                    .run_in_actix(&p, 1024, tx_clone.clone())
                    .await;
                match r {
                    Ok(_) => {}
                    Err(e) => {
                        error!("[rust-llm] Error: {}", e);
                    }
                }
                notify_clone.notify_one();
            }
        }
    });

    return HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(sse_stream(rx));
}
