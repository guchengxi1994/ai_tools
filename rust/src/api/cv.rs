use crate::utils::image_processor::ImageProcessor;

pub fn classify_image(s: String, model_path: String, classes: Option<Vec<String>>) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let r = rt.block_on(async {
        let img = ImageProcessor::load_image(&s, 224, None)?;
        let model = candle_onnx::read_file(model_path)?;
        let graph = model.graph.as_ref().unwrap();
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(graph.input[0].name.to_string(), img.unsqueeze(0)?);
        let mut outputs = candle_onnx::simple_eval(&model, inputs)?;
        let output = outputs.remove(&graph.output[0].name).unwrap();

        let prs = candle_core::IndexOp::i(&output, 0)?.to_vec1::<f32>()?;

        let mut top: Vec<_> = prs.iter().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top = top.into_iter().take(5).collect::<Vec<_>>();

        let cls;

        if let Some(c) = classes {
            cls = c;
        } else {
            cls = crate::constant::CLASSES
                .to_vec()
                .iter()
                .map(|&s| s.to_string())
                .collect();
        }

        anyhow::Ok(cls[top[0].0].clone())
    });

    match r {
        Ok(_r) => _r.to_owned(),
        Err(_e) => format!("error  {:?}", _e),
    }
}
