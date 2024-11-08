#[cfg(test)]
mod resnet_generator_test {
    use super::super::reflection_pad2d::reflection_pad2d;
    use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

    /// [UNFINISHED]
    /// some operations (InstanceNorm2d, reflection_pad2d) are not
    /// implemented yet in candle
    pub struct DownBlock {
        pub pad_size: usize,
        pub input_nc: usize,
        pub ngf: usize,
        pub kernel_size: usize,
        pub stride: usize,
        pub padding: usize,
        pub bias: bool,
        conv2: Conv2d,
    }

    impl DownBlock {
        pub fn new(
            pad_size: usize,
            input_nc: usize,
            ngf: usize,
            kernel_size: usize,
            stride: usize,
            padding: Option<usize>,
            bias: Option<bool>,
            vs: VarBuilder,
        ) -> anyhow::Result<Self> {
            let mut cfg: Conv2dConfig = Conv2dConfig::default();
            cfg.padding = padding.unwrap_or(0);
            cfg.stride = stride;
            let conv2 = conv2d(input_nc, ngf, kernel_size, cfg, vs.pp("conv2"))?;

            Ok(Self {
                pad_size,
                input_nc,
                ngf,
                kernel_size,
                stride,
                padding: padding.unwrap_or(0),
                bias: bias.unwrap_or(false),
                conv2,
            })
        }
    }

    impl Module for DownBlock {
        fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
            let xs = reflection_pad2d(xs, self.padding)?;
            let xs = self.conv2.forward(&xs)?;
            let xs = xs.relu()?;
            Ok(xs)
        }
    }
}
