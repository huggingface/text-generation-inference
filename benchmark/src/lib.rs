extern crate core;

mod event;
mod generation;
mod ui;
mod utils;

use crate::event::Event;
use crate::ui::UI;
use crossterm::ExecutableCommand;
use std::io;
use text_generation_client::ShardedClient;
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, mpsc};
use tui::backend::CrosstermBackend;
use tui::Terminal;

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
    let (run_sender, run_receiver) = mpsc::channel(8);
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let (shutdown_sender, _) = broadcast::channel(1);
    let (shutdown_guard_sender, mut shutdown_guard_receiver) = mpsc::channel(1);

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

    tokio::spawn(event::terminal_event_task(
        250,
        event_sender,
        shutdown_sender.subscribe(),
        shutdown_guard_sender.clone(),
    ));

    drop(shutdown_guard_sender);

    let mut ui = UI::new(
        run_receiver,
        tokenizer_name,
        sequence_length,
        decode_length,
        n_runs,
        batch_size,
    );

    crossterm::terminal::enable_raw_mode()?;
    io::stdout().execute(crossterm::terminal::EnterAlternateScreen)?;
    io::stdout().execute(crossterm::cursor::Hide)?;

    let mut terminal = {
        let backend = CrosstermBackend::new(io::stdout());
        Terminal::new(backend)?
    };

    while ui.running {
        terminal.draw(|frame| ui.render(frame))?;

        match event_receiver.recv().await {
            None => break,
            Some(event) => match event {
                Event::Tick => ui.tick(),
                Event::Key(key_event) => ui.handle_key_event(key_event),
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
