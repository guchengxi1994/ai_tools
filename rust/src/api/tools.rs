use flutter_rust_bridge::frb;

use crate::{
    frb_generated::StreamSink,
    tools::{TrainMessage, TRAIN_MESSAGE_SINK},
};

#[frb(sync)]
pub fn train_message_stream(s: StreamSink<TrainMessage>) -> anyhow::Result<()> {
    let mut stream = TRAIN_MESSAGE_SINK.write().unwrap();
    *stream = Some(s);
    anyhow::Ok(())
}

pub fn train_a_mlp(csv_path: String) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let r = crate::tools::train_a_mlp(csv_path);
        match r {
            Ok(_) => {}
            Err(e) => {
                println!("[rust] train mlp Error: {}", e);
            }
        }
    });
}
