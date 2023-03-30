mod app;
mod event;
mod generation;
mod utils;

use crate::app::App;
use crate::event::Event;
use crossterm::ExecutableCommand;
use std::io;
use text_generation_client::ShardedClient;
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, mpsc};
use tui::backend::CrosstermBackend;
use tui::Terminal;

/// Run benchmarking app
#[allow(clippy::too_many_arguments)]
pub async fn run(
    tokenizer_name: String,
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    n_runs: usize,
    warmups: usize,
    client: ShardedClient,
) -> Result<(), crossterm::ErrorKind> {
    // Initialize terminal properties
    crossterm::terminal::enable_raw_mode()?;
    io::stdout().execute(crossterm::terminal::EnterAlternateScreen)?;
    io::stdout().execute(crossterm::cursor::Hide)?;

    // Initialize terminal
    let mut terminal = {
        let backend = CrosstermBackend::new(io::stdout());
        Terminal::new(backend)?
    };

    // Create message channel between generation_task and app
    let (run_sender, run_receiver) = mpsc::channel(8);
    // Crossterm event channel
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    // Shutdown channel to terminate tasks
    let (shutdown_sender, _) = broadcast::channel(1);
    // Channel to check if tasks terminated
    let (shutdown_guard_sender, mut shutdown_guard_receiver) = mpsc::channel(1);

    // Create generation task
    tokio::spawn(generation::generation_task(
        tokenizer,
        batch_size.clone(),
        sequence_length,
        decode_length,
        n_runs,
        warmups,
        client,
        run_sender,
        shutdown_sender.subscribe(),
        shutdown_guard_sender.clone(),
    ));

    // Create event task
    tokio::spawn(event::terminal_event_task(
        250,
        event_sender,
        shutdown_sender.subscribe(),
        shutdown_guard_sender.clone(),
    ));

    // Drop our end of shutdown sender
    drop(shutdown_guard_sender);

    // Create App
    let mut app = App::new(
        run_receiver,
        tokenizer_name,
        sequence_length,
        decode_length,
        n_runs,
        batch_size,
    );

    while app.running {
        // Draw frame
        terminal.draw(|frame| app.render(frame))?;

        // Await a new event from event handling task
        match event_receiver.recv().await {
            None => break,
            // Update app state
            Some(event) => match event {
                Event::Tick => app.tick(),
                Event::Key(key_event) => app.handle_key_event(key_event),
                _ => {}
            },
        }
    }

    // Ask tasks to shutdown
    let _ = shutdown_sender.send(());
    // Wait for tasks to shutdown
    let _ = shutdown_guard_receiver.recv().await;

    // Revert terminal to original view
    io::stdout().execute(crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    io::stdout().execute(crossterm::cursor::Show)?;

    Ok(())
}
