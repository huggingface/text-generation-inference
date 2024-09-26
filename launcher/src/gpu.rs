use std::sync::LazyLock;

pub static COMPUTE_CAPABILITY: LazyLock<Option<(isize, isize)>> =
    LazyLock::new(get_cuda_capability);

fn get_cuda_capability() -> Option<(isize, isize)> {
    use pyo3::prelude::*;

    let py_get_capability = |py: Python| -> PyResult<(isize, isize)> {
        let torch = py.import_bound("torch.cuda")?;
        let get_device_capability = torch.getattr("get_device_capability")?;
        get_device_capability.call0()?.extract()
    };

    match pyo3::Python::with_gil(py_get_capability) {
        Ok(capability) => Some(capability),
        Err(err) => {
            tracing::warn!("Cannot determine GPU compute capability: {}", err);
            None
        }
    }
}
