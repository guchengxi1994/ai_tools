// copy and modify from https://github.com/huggingface/candle/blob/main/candle-examples/examples/beit/main.rs

use candle_core::{DType, Device, IndexOp, D};
use candle_nn::{Module, VarBuilder};

use crate::{constant::CLASSES, cv::load_image384_beit_norm};

use super::{CvTask, Model, ModelRun};
use candle_transformers::models::beit::{self, BeitVisionTransformer};

impl Model<BeitVisionTransformer> {
    pub fn new(model_path: String) -> Self {
        Model {
            inner: None,
            model_path,
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
        }
    }
}

impl ModelRun<Vec<(String, f32)>> for Model<BeitVisionTransformer> {
    fn run_in_bytes(&self, image_bytes: Vec<u8>) -> anyhow::Result<Vec<(String, f32)>> {
        anyhow::bail!("Not implemented")
    }

    fn run(&self, image_path: String, top_n: usize) -> anyhow::Result<Vec<(String, f32)>> {
        assert!(top_n <= 100 && top_n > 0);

        let img = load_image384_beit_norm(image_path, &self.device)?;
        let mut result: Vec<(String, f32)> = vec![];
        if let Some(inner) = &self.inner {
            let logits = inner.forward(&img.unsqueeze(0)?)?;
            let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
                .i(0)?
                .to_vec1::<f32>()?;
            let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
            prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
            for &(category_idx, pr) in prs.iter().take(top_n) {
                println!("{:24}: {:.2}%", CLASSES[category_idx], 100. * pr);
                result.push((CLASSES[category_idx].to_string(), *pr));
            }

            return Ok(result);
        }

        anyhow::bail!("Model not loaded")
    }

    fn load(&mut self) -> anyhow::Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.model_path.clone()],
                DType::F32,
                &self.device,
            )?
        };
        self.inner = Some(beit::vit_base(vb)?);
        Ok(())
    }

    fn get_task_type(&self) -> CvTask {
        CvTask::ImageClassification
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run() -> anyhow::Result<()> {
        let mut model: Model<BeitVisionTransformer> = Model::<BeitVisionTransformer>::new(r"D:\github_repo\ai_tools\rust\assets\beit_base_patch16_384.in22k_ft_in22k_in1k.safetensors".to_string());
        model.load()?;

        let result = model.run(
            r"D:\github_repo\ai_tools\rust\assets\sock.jpg".to_string(),
            5,
        )?;

        println!("{:?}", result);
        Ok(())
    }
}
