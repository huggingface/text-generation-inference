/// Inspired by https://github.com/hatoo/oha/blob/master/src/monitor.rs

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::ExecutableCommand;
use std::collections::BTreeMap;
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::time::sleep;
use tui::backend::CrosstermBackend;
use tui::layout::{Constraint, Direction, Layout};
use tui::style::{Color, Style};
use tui::text::{Span, Spans};
use tui::widgets::{BarChart, Block, Borders, Gauge, Paragraph};
use tui::Terminal;
use tokio::sync::mpsc::Receiver;

pub struct UI {
    pub n_run: usize,
    pub n_batch: usize,
    pub n_batch_done: usize,
    pub run_receiver: Receiver<()>,
}

impl UI {
    pub async fn draw(
        mut self
    ) -> Result<(), crossterm::ErrorKind> {
        crossterm::terminal::enable_raw_mode()?;
        io::stdout().execute(crossterm::terminal::EnterAlternateScreen)?;
        io::stdout().execute(crossterm::cursor::Hide)?;

        let mut runs = Vec::new();

        let mut terminal = {
            let backend = CrosstermBackend::new(io::stdout());
            Terminal::new(backend)?
        };

        'outer: loop {
            let frame_start = Instant::now();
            loop {
                match self.run_receiver.try_recv() {
                    Ok(run) => {
                        // match report.as_ref() {
                        //     Ok(report) => *status_dist.entry(report.status).or_default() += 1,
                        //     Err(e) => *error_dist.entry(e.to_string()).or_default() += 1,
                        // }
                        // all.push(report);
                        runs.push(run);
                    }
                    Err(TryRecvError::Empty) => {
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        // Application ends.
                        break 'outer;
                    }
                }
            }

            let draw_start = Instant::now();

            let batch_progress = (self.n_batch_done as f64 / self.n_batch as f64).clamp(0.0, 1.0);
            let run_progress = (runs.len() as f64 / self.n_run as f64).clamp(0.0, 1.0);

            terminal.draw(|f| {
                let row3 = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(
                        [
                            Constraint::Length(3),
                            Constraint::Length(10),
                            Constraint::Percentage(45),
                        ].as_ref(),
                    ).split(f.size());

                let top = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(50),
                        Constraint::Percentage(50),
                    ].as_ref()).split(row3[0]);

                let mid = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(20),
                        Constraint::Percentage(30),
                        Constraint::Percentage(20),
                        Constraint::Percentage(30),
                    ].as_ref()).split(row3[1]);

                let prefill_text = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(5),
                        Constraint::Length(5),
                    ].as_ref()).split(mid[0]);

                let decode_text = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(5),
                        Constraint::Length(5),
                    ].as_ref()).split(mid[2]);

                let bottom = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                    ].as_ref()).split(row3[2]);

                let batch_gauge = Gauge::default()
                    .block(Block::default().title("Total Progress").borders(Borders::ALL))
                    .gauge_style(Style::default().fg(Color::White))
                    .label(Span::raw(format!("{} / {}", self.n_batch_done, self.n_batch)))
                    .ratio(batch_progress);
                f.render_widget(batch_gauge, top[0]);

                let run_gauge = Gauge::default()
                    .block(Block::default().title("Batch Progress").borders(Borders::ALL))
                    .gauge_style(Style::default().fg(Color::White))
                    .label(Span::raw(format!("{} / {}", runs.len(), self.n_run)))
                    .ratio(run_progress);
                f.render_widget(run_gauge, top[1]);

                let data = vec![0.0];

                let prefill_latency_texts = statis_spans(&data, "ms", false);
                let prefill_throughput_texts = statis_spans(&data, "tokens/secs", false);

                let prefill_latency_statics = Paragraph::new(prefill_latency_texts).block(
                    Block::default()
                        .title(Span::raw("Prefill Latency"))
                        .borders(Borders::ALL),
                );
                f.render_widget(prefill_latency_statics, prefill_text[0]);

                let prefill_throughput_statics = Paragraph::new(prefill_throughput_texts).block(
                    Block::default()
                        .title(Span::raw("Prefill Throughput"))
                        .borders(Borders::ALL),
                );
                f.render_widget(prefill_throughput_statics, prefill_text[1]);
            })?;

            let per_frame = Duration::from_secs(1) / 30 as u32;
            let elapsed = frame_start.elapsed();
            if per_frame > elapsed {
                sleep(per_frame - elapsed).await;
            }
        }


        io::stdout().execute(crossterm::terminal::LeaveAlternateScreen)?;
        crossterm::terminal::disable_raw_mode()?;
        io::stdout().execute(crossterm::cursor::Show)?;
        Ok(())
    }
}

fn statis_spans<'a>(data: &Vec<f64>, unit: &'static str, color: bool) -> Vec<Spans<'a>> {
    vec![
        Spans::from(vec![Span::styled(
            format!(
                "Lowest: {:.4} {unit}",
                data
                    .iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&std::f64::NAN)
            ),
            Style::default().fg(Color::Reset),
        )]),
        Spans::from(vec![Span::styled(
            format!(
                "Highest: {:.4} {unit}",
                data
                    .iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&std::f64::NAN)
            ),
            Style::default().fg(Color::Reset),
        )]),
        Spans::from(vec![Span::styled(
            format!(
                "Average: {:.4} {unit}",
                data
                    .iter()
                    .sum::<f64>()
                    / data.len() as f64
            ),
            Style::default().fg(Color::Reset),
        )]),
    ]
}