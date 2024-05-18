use text_generation_router::{internal_main_args, RouterError};

#[tokio::main]
async fn main() -> Result<(), RouterError> {
    internal_main_args().await?;
    Ok(())
}
