use text_generation_backends_vllm::{EngineArgs, LlmEngine};

#[tokio::main]
async fn main() -> Result<(), ()> {
    let args = EngineArgs {
        model: String::from("meta-llama/Llama-3.2-1B-Instruct"),
        pipeline_parallel_size: 1,
        tensor_parallel_size: 1,
    };

    match LlmEngine::from_engine_args(args) {
        Ok(_) => println!("Engine successfully allocated"),
        Err(err) => println!("Got an error: {}", err),
    }

    Ok(())
}
