use anyhow::Ok;
use candle_core::{Device, Result, Tensor};

fn tensor_flip(input: &Tensor, flip: usize) -> candle_core::Result<Tensor> {
    let mut result = Tensor::zeros(
        (input.dims()[0], input.dims()[1]),
        input.dtype(),
        input.device(),
    )?;
    match flip {
        // row
        0 => {
            for i in 0..input.dims()[0] {
                let row_tensor = input.narrow(0, i, 1)?;
                println!("row_tensor {:?}", row_tensor.shape());

                result = result.slice_assign(
                    &[
                        input.dims()[0] - i - 1..input.dims()[0] - i,
                        0..input.dims()[1],
                    ],
                    &row_tensor,
                )?;
            }
        }
        // column
        1 => {
            for i in 0..input.dims()[1] {
                let column_tensor = input.narrow(1, i, 1)?;
                result = result.slice_assign(
                    &[
                        0..input.dims()[0],
                        input.dims()[1] - i - 1..input.dims()[1] - i,
                    ],
                    &column_tensor,
                )?;
            }
        }
        _ => {}
    }

    candle_core::Result::Ok(result)
}

// Reflection padding function
pub fn reflection_pad2d(
    input: &Tensor,
    padding: usize,
) -> candle_core::Result<candle_core::Tensor> {
    assert!(padding > 0);
    assert!(input.shape().rank() == 3 || input.shape().rank() == 4);
    let this_input;
    if input.shape().rank() == 3 {
        this_input = input.copy()?;
    } else {
        this_input = input.squeeze(0)?;
    }

    let dims = this_input.shape().dims();
    let mut vc: Vec<Tensor> = Vec::new();

    for i in 0..dims[0] {
        let mut t = Tensor::zeros(
            (dims[1] + 2 * padding, dims[2] + 2 * padding),
            input.dtype(),
            input.device(),
        )?;
        let t_dims = [dims[1] + 2 * padding, dims[2] + 2 * padding];
        let tensor_i = this_input.get(i)?;

        let mut left_top = tensor_i.narrow(0, 1, padding)?.narrow(1, 1, padding)?;
        let mut right_bottom = tensor_i.narrow(0, dims[1] - padding - 1, padding)?.narrow(
            1,
            dims[2] - padding - 1,
            padding,
        )?;
        let mut right_top =
            tensor_i
                .narrow(0, 1, padding)?
                .narrow(1, dims[2] - padding - 1, padding)?;
        let mut left_bottom = tensor_i
            .narrow(0, dims[1] - padding - 1, padding)?
            .narrow(1, 1, padding)?;
        let mut top_center = tensor_i.narrow(0, 1, padding)?;
        let mut bottom_center = tensor_i.narrow(0, dims[1] - padding - 1, padding)?;
        let mut left_center = tensor_i.narrow(1, 1, padding)?;
        let mut right_center = tensor_i.narrow(1, dims[2] - padding - 1, padding)?;

        // put input in the center
        t = t.slice_assign(
            &[padding..dims[1] + padding, padding..dims[2] + padding],
            &tensor_i,
        )?;
        // put left_top
        left_top = tensor_flip(&mut left_top, 0)?;
        left_top = tensor_flip(&mut left_top, 1)?;
        t = t.slice_assign(&[0..padding, 0..padding], &left_top)?;
        // put right_top
        right_top = tensor_flip(&mut right_top, 0)?;
        right_top = tensor_flip(&mut right_top, 1)?;
        t = t.slice_assign(&[0..padding, t_dims[1] - padding..t_dims[1]], &right_top)?;
        // put left_bottom
        left_bottom = tensor_flip(&mut left_bottom, 0)?;
        left_bottom = tensor_flip(&mut left_bottom, 1)?;
        t = t.slice_assign(&[t_dims[0] - padding..t_dims[0], 0..padding], &left_bottom)?;
        // put right_bottom
        right_bottom = tensor_flip(&mut right_bottom, 0)?;
        right_bottom = tensor_flip(&mut right_bottom, 1)?;
        // put top_center
        top_center = tensor_flip(&mut top_center, 0)?;
        t = t.slice_assign(&[0..padding, padding..dims[2] + padding], &top_center)?;
        // put bottom_center
        bottom_center = tensor_flip(&mut bottom_center, 0)?;
        t = t.slice_assign(
            &[t_dims[0] - padding..t_dims[0], padding..dims[2] + padding],
            &bottom_center,
        )?;
        // put left_center
        left_center = tensor_flip(&mut left_center, 1)?;
        t = t.slice_assign(&[padding..dims[1] + padding, 0..padding], &left_center)?;

        // put right_center
        right_center = tensor_flip(&mut right_center, 1)?;
        t = t.slice_assign(
            &[padding..dims[1] + padding, t_dims[1] - padding..t_dims[1]],
            &right_center,
        )?;

        t = t.slice_assign(
            &[
                t_dims[0] - padding..t_dims[0],
                t_dims[1] - padding..t_dims[1],
            ],
            &right_bottom,
        )?;

        vc.push(t);
    }

    let output = Tensor::stack(&vc, 0)?;
    println!("output shape: {:?}", output.shape());

    if input.shape().rank() == 3 {
        return candle_core::Result::Ok(output);
    } else {
        return candle_core::Result::Ok(output.unsqueeze(0)?);
    }
}

#[allow(unused_imports)]
mod test {
    use candle_core::Tensor;

    use crate::tests::reflection_pad2d::reflection_pad2d;

    #[test]
    fn test_reflection_pad2d() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let mut v = Vec::new();
        for i in 1..26 {
            v.push(i as f32);
        }

        let mut origin_tensor = Tensor::from_vec(v, &[5, 5], &device)?;

        origin_tensor = origin_tensor.unsqueeze(0)?;

        println!("{:?}", origin_tensor.to_vec3::<f32>()?);

        let result = reflection_pad2d(&origin_tensor, 3)?;

        println!("{:?}", result.to_vec3::<f32>()?);

        anyhow::Result::Ok(())
    }
}
