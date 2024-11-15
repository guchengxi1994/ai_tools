use flutter_rust_bridge::frb;

use crate::web_server::SERVER_STATE_SINK;

pub fn run_server(port: Option<usize>) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let _ = crate::web_server::start_server(port).await;
    });
}

pub fn load_llm_model(model_path: String) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let r = rt.block_on(async { crate::web_server::load_model(Some(model_path)).await });
    match r {
        Ok(_) => {}
        Err(_) => {
            if let Some(s) = SERVER_STATE_SINK.read().unwrap().as_ref() {
                let _ = s.add("model failed".to_string());
            }
        }
    }
}

pub fn stop_server() {
    crate::web_server::stop_server();
}

#[frb(sync)]
pub fn check_modal_status() -> bool {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let r = rt.block_on(async { crate::web_server::check_llm_model_loaded().await });
    r
}
