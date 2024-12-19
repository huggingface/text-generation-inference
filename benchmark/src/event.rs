/// Inspired by https://github.com/orhun/rust-tui-template/blob/472aa515119d4c94903eac12d9784417281dc7f5/src/event.rs
use ratatui::crossterm::event;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};

/// Events
#[derive(Debug)]
pub(crate) enum Event {
    /// Terminal tick.
    Tick,
    /// Key press.
    Key(event::KeyEvent),
    /// Terminal resize.
    Resize,
}

pub(crate) async fn terminal_event_task(
    fps: u32,
    event_sender: mpsc::Sender<Event>,
    mut shutdown_receiver: broadcast::Receiver<()>,
    _shutdown_guard_sender: mpsc::Sender<()>,
) {
    // End task if a message is received on shutdown_receiver
    // _shutdown_guard_sender will be dropped once the task is finished
    tokio::select! {
        _ = event_loop(fps, event_sender)  => {
        },
        _ = shutdown_receiver.recv() => {}
    }
}

/// Main event loop
async fn event_loop(fps: u32, event_sender: mpsc::Sender<Event>) {
    // Frame budget
    let per_frame = Duration::from_secs(1) / fps;

    // When was last frame executed
    let mut last_frame = Instant::now();

    loop {
        // Sleep to avoid blocking the thread for too long
        if let Some(sleep) = per_frame.checked_sub(last_frame.elapsed()) {
            tokio::time::sleep(sleep).await;
        }

        // Get crossterm event and send a new one over the channel
        if event::poll(Duration::from_secs(0)).expect("no events available") {
            match event::read().expect("unable to read event") {
                event::Event::Key(e) => event_sender.send(Event::Key(e)).await.unwrap_or(()),
                event::Event::Resize(_w, _h) => {
                    event_sender.send(Event::Resize).await.unwrap_or(())
                }
                _ => (),
            }
        }

        // Frame budget exceeded
        if last_frame.elapsed() >= per_frame {
            // Send tick
            event_sender.send(Event::Tick).await.unwrap_or(());
            // Rest last_frame time
            last_frame = Instant::now();
        }
    }
}
