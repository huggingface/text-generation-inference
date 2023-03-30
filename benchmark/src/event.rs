/// Inspired by https://github.com/orhun/rust-tui-template
use crossterm::event;
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
    Resize(u16, u16),
}

pub(crate) async fn terminal_event_task(
    fps: u32,
    event_sender: mpsc::Sender<Event>,
    mut shutdown_receiver: broadcast::Receiver<()>,
    _shutdown_guard_sender: mpsc::Sender<()>,
) {
    tokio::select! {
        _ = event_loop(fps, event_sender)  => {
        },
        _ = shutdown_receiver.recv() => {}
    }
}

async fn event_loop(fps: u32, event_sender: mpsc::Sender<Event>) {
    let per_frame = Duration::from_secs(1) / fps as u32;
    let mut last_frame = Instant::now();
    loop {
        if let Some(sleep) = per_frame.checked_sub(last_frame.elapsed()) {
            tokio::time::sleep(sleep).await;
        }

        if event::poll(Duration::from_secs(0)).expect("no events available") {
            match event::read().expect("unable to read event") {
                event::Event::Key(e) => event_sender.send(Event::Key(e)).await.unwrap_or(()),
                event::Event::Resize(w, h) => {
                    event_sender.send(Event::Resize(w, h)).await.unwrap_or(())
                }
                _ => (),
            }
        }

        if last_frame.elapsed() >= per_frame {
            event_sender.send(Event::Tick).await.unwrap_or(());
            last_frame = Instant::now();
        }
    }
}
