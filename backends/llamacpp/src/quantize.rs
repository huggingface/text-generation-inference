use crate::llamacpp;

use std::ffi::CString;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum QuantizeType {
    MostlyQ4_0 = 2,
}

pub fn model(
    input_path: &str,
    output_path: &str,
    ftype: QuantizeType,
    n_threads: usize,
) -> Result<(), String> {
    let c_input_path =
        CString::new(input_path).map_err(|e| format!("Failed to convert input path: {}", e))?;

    let c_output_path =
        CString::new(output_path).map_err(|e| format!("Failed to convert output path: {}", e))?;

    let result = unsafe {
        let mut params = llamacpp::model_quantize_default_params();
        params.nthread = n_threads as _;
        params.ftype = ftype as _;
        params.quantize_output_tensor = true;
        llamacpp::model_quantize(c_input_path.as_ptr(), c_output_path.as_ptr(), &params)
    };
    if result == 0 {
        Ok(())
    } else {
        Err(format!("Quantization failed, error code: {}", result))
    }
}
