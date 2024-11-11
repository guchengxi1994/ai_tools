use candle_core::Device;
use candle_onnx::onnx::ModelProto;

use crate::{constant::CLASSES, cv::load_image224_efficientnet_norm};

use super::{CvTask, Model, ModelRun};

impl Model<ModelProto> {
    pub fn new(model_path: String) -> Self {
        Model {
            inner: None,
            model_path,
            device: Device::Cpu,
        }
    }
}

impl ModelRun<Vec<(String, f32)>> for Model<ModelProto> {
    fn run(&self, image_path: String, top_n: usize) -> anyhow::Result<Vec<(String, f32)>> {
        assert!(top_n <= 100 && top_n > 0);

        let img = load_image224_efficientnet_norm(image_path, &self.device)?;
        if let Some(inner) = &self.inner {
            let graph = inner.graph.as_ref().unwrap();
            let mut inputs = std::collections::HashMap::new();
            inputs.insert(graph.input[0].name.to_string(), img.unsqueeze(0)?);
            let mut outputs = candle_onnx::simple_eval(&inner, inputs)?;
            let output = outputs.remove(&graph.output[0].name).unwrap();

            let prs = candle_core::IndexOp::i(&output, 0)?.to_vec1::<f32>()?;
            let mut result: Vec<(String, f32)> = vec![];
            let mut top: Vec<_> = prs.iter().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let top = top.into_iter().take(5).collect::<Vec<_>>();

            // Print the top predictions
            for &(i, p) in &top {
                println!("{:50}: {:.2}%", CLASSES[i], p * 100.0);
                result.push((CLASSES[i].to_string(), *p));
            }
            return Ok(result);
        }

        anyhow::bail!("model not loaded")
    }

    fn load(&mut self) -> anyhow::Result<()> {
        let model = candle_onnx::read_file(self.model_path.clone())?;
        self.inner = Some(model);
        Ok(())
    }

    fn get_task_type(&self) -> super::CvTask {
        CvTask::ImageClassification
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run() -> anyhow::Result<()> {
        let mut model: Model<ModelProto> = Model::<ModelProto>::new(
            r"D:\github_repo\ai_tools\rust\assets\efficientnet-lite4-s.onnx".to_string(),
        );
        model.load()?;

        let result = model.run(
            r"D:\github_repo\ai_tools\rust\assets\sock.jpg".to_string(),
            5,
        )?;

        println!("{:?}", result);
        Ok(())
    }
}
