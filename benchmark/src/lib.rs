mod app;
mod event;
mod generation;
mod table;
mod utils;

use crate::app::App;
use crate::event::Event;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::ExecutableCommand;
use ratatui::Terminal;
use std::io;
use text_generation_client::v3::{GrammarType, NextTokenChooserParameters, ShardedClient};
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, mpsc};

/// Run benchmarking app
#[allow(clippy::too_many_arguments)]
pub async fn run(
    tokenizer_name: String,
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    top_n_tokens: Option<u32>,
    n_runs: usize,
    warmups: usize,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    typical_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    watermark: bool,
    do_sample: bool,
    client: ShardedClient,
) -> Result<(), std::io::Error> {
    let parameters = NextTokenChooserParameters {
        temperature: temperature.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        top_p: top_p.unwrap_or(1.0),
        typical_p: typical_p.unwrap_or(1.0),
        do_sample,
        seed: 0,
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        watermark,
        grammar: String::new(),
        grammar_type: GrammarType::None as i32,
    };

    // Initialize terminal properties
    ratatui::crossterm::terminal::enable_raw_mode()?;
    io::stdout().execute(ratatui::crossterm::terminal::EnterAlternateScreen)?;
    io::stdout().execute(ratatui::crossterm::cursor::Hide)?;

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
        top_n_tokens,
        n_runs,
        warmups,
        parameters,
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
        tokenizer_name.clone(),
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
    io::stdout().execute(ratatui::crossterm::terminal::LeaveAlternateScreen)?;
    ratatui::crossterm::terminal::disable_raw_mode()?;
    io::stdout().execute(ratatui::crossterm::cursor::Show)?;

    let parameters_table = table::parameters_table(
        tokenizer_name,
        sequence_length,
        decode_length,
        top_n_tokens,
        n_runs,
        warmups,
        temperature,
        top_k,
        top_p,
        typical_p,
        repetition_penalty,
        frequency_penalty,
        watermark,
        do_sample,
    );
    println!("\n{parameters_table}\n");

    let latency_table = table::latency_table(&app.data);
    println!("\n{latency_table}\n");

    let throughput_table = table::throughput_table(&app.data);
    println!("\n{throughput_table}\n");

    Ok(())
}
