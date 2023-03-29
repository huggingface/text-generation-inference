/// Inspired by https://github.com/hatoo/oha/blob/master/src/monitor.rs
use crate::Message;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::{event, ExecutableCommand};
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::{broadcast, mpsc};
use tokio::time::sleep;
use tui::backend::CrosstermBackend;
use tui::layout::{Constraint, Direction, Layout};
use tui::style::{Color, Modifier, Style};
use tui::text::{Span, Spans};
use tui::widgets::{
    Axis, BarChart, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Tabs,
};
use tui::{symbols, Terminal};

pub(crate) struct UI {
    pub(crate) tokenizer_name: String,
    pub(crate) sequence_length: u32,
    pub(crate) decode_length: u32,
    pub(crate) n_run: usize,
    pub(crate) batch_size: Vec<u32>,
    pub(crate) receiver: mpsc::Receiver<Message>,
    pub(crate) shutdown_sender: broadcast::Sender<()>,
}

impl UI {
    pub async fn draw(mut self) -> Result<(), crossterm::ErrorKind> {
        crossterm::terminal::enable_raw_mode()?;
        io::stdout().execute(crossterm::terminal::EnterAlternateScreen)?;
        io::stdout().execute(crossterm::cursor::Hide)?;

        let mut current_tab_idx = 0;

        let mut prefill_latencies: Vec<Vec<f64>> = (0..self.batch_size.len())
            .map(|_| Vec::with_capacity(self.n_run))
            .collect();
        let mut prefill_throughputs: Vec<Vec<f64>> = (0..self.batch_size.len())
            .map(|_| Vec::with_capacity(self.n_run))
            .collect();

        let mut decode_latencies: Vec<Vec<f64>> = (0..self.batch_size.len())
            .map(|_| Vec::with_capacity(self.n_run))
            .collect();
        let mut decode_throughputs: Vec<Vec<f64>> = (0..self.batch_size.len())
            .map(|_| Vec::with_capacity(self.n_run))
            .collect();

        let mut prefill_batch_latency_throughput: Vec<(f64, f64)> =
            Vec::with_capacity(self.batch_size.len());

        let mut decode_batch_latency_throughput: Vec<(f64, f64)> =
            Vec::with_capacity(self.batch_size.len());

        let mut completed_runs: Vec<usize> = (0..self.batch_size.len()).map(|_| 0).collect();
        let mut completed_batch = 0;
        let mut current_batch_idx = 0;

        let mut terminal = {
            let backend = CrosstermBackend::new(io::stdout());
            Terminal::new(backend)?
        };

        'outer: loop {
            let frame_start = Instant::now();
            loop {
                match self.receiver.try_recv() {
                    Ok(message) => match message {
                        Message::Prefill(step) => {
                            let latency = step.latency.as_millis() as f64;
                            let throughput = step.batch_size as f64 / step.latency.as_secs_f64();
                            prefill_latencies[current_batch_idx].push(latency);
                            prefill_throughputs[current_batch_idx].push(throughput);
                        }
                        Message::Decode(step) => {
                            let latency = step.latency.as_millis() as f64;
                            let throughput = (step.batch_size * step.decode_length) as f64
                                / step.latency.as_secs_f64();
                            decode_latencies[current_batch_idx].push(latency);
                            decode_throughputs[current_batch_idx].push(throughput);
                        }
                        Message::IncreaseRun => {
                            completed_runs[current_batch_idx] += 1;
                        }
                        Message::IncreaseBatch => {
                            prefill_batch_latency_throughput.push((
                                prefill_latencies[current_batch_idx].iter().sum::<f64>()
                                    / completed_runs[current_batch_idx] as f64,
                                prefill_throughputs[current_batch_idx].iter().sum::<f64>()
                                    / completed_runs[current_batch_idx] as f64,
                            ));
                            decode_batch_latency_throughput.push((
                                decode_latencies[current_batch_idx].iter().sum::<f64>()
                                    / completed_runs[current_batch_idx] as f64,
                                decode_throughputs[current_batch_idx].iter().sum::<f64>()
                                    / completed_runs[current_batch_idx] as f64,
                            ));

                            completed_batch += 1;
                            if current_batch_idx < self.batch_size.len() - 1 {
                                current_batch_idx += 1;
                            }
                        }
                    },
                    Err(TryRecvError::Empty) => {
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        break;
                    }
                }
            }

            let batch_progress =
                (completed_batch as f64 / self.batch_size.len() as f64).clamp(0.0, 1.0);
            let run_progress =
                (completed_runs[current_batch_idx] as f64 / self.n_run as f64).clamp(0.0, 1.0);

            terminal.draw(|f| {
                // Vertical layout
                let row5 = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(
                        [
                            Constraint::Length(1),
                            Constraint::Length(3),
                            Constraint::Length(3),
                            Constraint::Length(13),
                            Constraint::Min(10),
                        ]
                        .as_ref(),
                    )
                    .split(f.size());

                // Top row horizontal layout
                let top = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                    .split(row5[2]);

                // Mid row horizontal layout
                let mid = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(
                        [
                            Constraint::Percentage(20),
                            Constraint::Percentage(30),
                            Constraint::Percentage(20),
                            Constraint::Percentage(30),
                        ]
                        .as_ref(),
                    )
                    .split(row5[3]);

                // Left mid row vertical layout
                let prefill_text = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(8), Constraint::Length(5)].as_ref())
                    .split(mid[0]);

                // Right mid row vertical layout
                let decode_text = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(8), Constraint::Length(5)].as_ref())
                    .split(mid[2]);

                // Bottom row horizontal layout
                let bottom = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                    .split(row5[4]);

                // Title
                let title = Block::default().borders(Borders::NONE).title(format!(
                    "Model: {} | Sequence Length: {} | Decode Length: {}",
                    self.tokenizer_name, self.sequence_length, self.decode_length
                )).style(Style::default().add_modifier(Modifier::BOLD).fg(Color::White));
                f.render_widget(title, row5[0]);

                // Batch tabs
                let titles = self
                    .batch_size
                    .iter()
                    .map(|b| {
                        Spans::from(vec![Span::styled(
                            format!("Batch: {b}"),
                            Style::default().fg(Color::White),
                        )])
                    })
                    .collect();
                let tabs = Tabs::new(titles)
                    .block(Block::default().borders(Borders::ALL).title("Tabs"))
                    .select(current_tab_idx)
                    .style(Style::default().fg(Color::LightCyan))
                    .highlight_style(
                        Style::default()
                            .add_modifier(Modifier::BOLD)
                            .bg(Color::Black),
                    );
                f.render_widget(tabs, row5[1]);

                // Total progress bar
                let batch_gauge = progress_gauge(
                    "Total Progress",
                    format!("{} / {}", completed_batch, self.batch_size.len()),
                    batch_progress,
                    Color::LightGreen,
                );
                f.render_widget(batch_gauge, top[0]);

                // Batch progress Bar
                let run_gauge = progress_gauge(
                    "Batch Progress",
                    format!("{} / {}", completed_runs[current_batch_idx], self.n_run),
                    run_progress,
                    Color::LightBlue,
                );
                f.render_widget(run_gauge, top[1]);

                // Prefill text infos
                let (prefill_latency_statics, prefill_throughput_statics) = text_info(
                    &mut prefill_latencies[current_tab_idx],
                    &prefill_throughputs[current_tab_idx],
                    "Prefill",
                );
                f.render_widget(prefill_latency_statics, prefill_text[0]);
                f.render_widget(prefill_throughput_statics, prefill_text[1]);

                // Prefill latency histogram
                let histo_width = 7;
                let bins = if mid[1].width < 2 {
                    0
                } else {
                    (mid[1].width as usize - 2) / (histo_width + 1)
                }
                .max(2);

                let histo_data = latency_histogram_data(&prefill_latencies[current_tab_idx], bins);
                let histo_data_str: Vec<(&str, u64)> =
                    histo_data.iter().map(|(l, v)| (l.as_str(), *v)).collect();
                let prefill_histogram =
                    latency_histogram(&histo_data_str, "Prefill").bar_width(histo_width as u16);
                f.render_widget(prefill_histogram, mid[1]);

                // Decode text info
                let (decode_latency_statics, decode_throughput_statics) = text_info(
                    &mut decode_latencies[current_tab_idx],
                    &decode_throughputs[current_tab_idx],
                    "Decode",
                );
                f.render_widget(decode_latency_statics, decode_text[0]);
                f.render_widget(decode_throughput_statics, decode_text[1]);

                // Decode latency histogram
                let histo_data = latency_histogram_data(&decode_latencies[current_tab_idx], bins);
                let histo_data_str: Vec<(&str, u64)> =
                    histo_data.iter().map(|(l, v)| (l.as_str(), *v)).collect();
                let decode_histogram =
                    latency_histogram(&histo_data_str, "Decode").bar_width(histo_width as u16);
                f.render_widget(decode_histogram, mid[3]);

                // Prefill latency/throughput chart
                let prefill_latency_throughput_chart = latency_throughput_chart(
                    &prefill_batch_latency_throughput,
                    &self.batch_size,
                    "Prefill",
                );
                f.render_widget(prefill_latency_throughput_chart, bottom[0]);

                // Decode latency/throughput chart
                let decode_latency_throughput_chart = latency_throughput_chart(
                    &decode_batch_latency_throughput,
                    &self.batch_size,
                    "Decode",
                );
                f.render_widget(decode_latency_throughput_chart, bottom[1]);
            })?;

            // Quit on q or CTRL+c

            while event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key {
                        KeyEvent {
                            code: KeyCode::Right,
                            ..
                        } => {
                            current_tab_idx = (current_tab_idx + 1) % self.batch_size.len();
                        }
                        KeyEvent {
                            code: KeyCode::Left,
                            ..
                        } => {
                            if current_tab_idx > 0 {
                                current_tab_idx -= 1;
                            } else {
                                current_tab_idx = self.batch_size.len() - 1;
                            }
                        }
                        KeyEvent {
                            code: KeyCode::Char('q'),
                            ..
                        }
                        | KeyEvent {
                            code: KeyCode::Char('c'),
                            modifiers: KeyModifiers::CONTROL,
                            ..
                        } => {
                            break 'outer;
                        }
                        _ => (),
                    }
                }
            }

            // Frame budget
            let per_frame = Duration::from_secs(1) / 30 as u32;
            let elapsed = frame_start.elapsed();
            if per_frame > elapsed {
                sleep(per_frame - elapsed).await;
            }
        }

        // Revert terminal to original view
        io::stdout().execute(crossterm::terminal::LeaveAlternateScreen)?;
        crossterm::terminal::disable_raw_mode()?;
        io::stdout().execute(crossterm::cursor::Show)?;

        let _ = self.shutdown_sender.send(());
        Ok(())
    }
}

fn progress_gauge(title: &str, label: String, progress: f64, color: Color) -> Gauge {
    Gauge::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(color))
        .label(Span::raw(label))
        .ratio(progress)
}

fn text_info<'a>(
    latency: &mut Vec<f64>,
    throughput: &Vec<f64>,
    name: &'static str,
) -> (Paragraph<'a>, Paragraph<'a>) {
    let mut latency_texts = statis_spans(&latency, "ms");
    float_ord::sort(latency);
    let latency_percentiles = crate::utils::percentiles(latency, &[50, 90, 99]);
    let colors = vec![Color::LightGreen, Color::LightYellow, Color::LightRed];
    for (i, (name, value)) in latency_percentiles.iter().enumerate() {
        let span = Spans::from(vec![Span::styled(
            format!("{name}:     {:.4} ms", value),
            Style::default().fg(colors[i]),
        )]);
        latency_texts.push(span);
    }

    let throughput_texts = statis_spans(&throughput, "tokens/secs");

    let latency_statics = Paragraph::new(latency_texts).block(
        Block::default()
            .title(Span::raw(format!("{name} Latency")))
            .borders(Borders::ALL),
    );

    let throughput_statics = Paragraph::new(throughput_texts).block(
        Block::default()
            .title(Span::raw(format!("{name} Throughput")))
            .borders(Borders::ALL),
    );

    (latency_statics, throughput_statics)
}

fn latency_histogram_data(latency: &Vec<f64>, bins: usize) -> Vec<(String, u64)> {
    let histo_data: Vec<(String, u64)> = {
        let histo = crate::utils::histogram(latency, bins);
        histo
            .into_iter()
            .map(|(label, v)| (format!("{label:.2}"), v as u64))
            .collect()
    };

    histo_data
}

fn latency_histogram<'a>(
    histo_data_str: &'a Vec<(&'a str, u64)>,
    name: &'static str,
) -> BarChart<'a> {
    BarChart::default()
        .block(
            Block::default()
                .title(format!("{name} latency histogram"))
                .style(Style::default().fg(Color::LightYellow).bg(Color::Reset))
                .borders(Borders::ALL),
        )
        .data(histo_data_str.as_slice())
}

fn statis_spans<'a>(data: &Vec<f64>, unit: &'static str) -> Vec<Spans<'a>> {
    vec![
        Spans::from(vec![Span::styled(
            format!(
                "Average: {:.4} {unit}",
                data.iter().sum::<f64>() / data.len() as f64
            ),
            Style::default().fg(Color::LightBlue),
        )]),
        Spans::from(vec![Span::styled(
            format!(
                "Lowest:  {:.4} {unit}",
                data.iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&std::f64::NAN)
            ),
            Style::default().fg(Color::Reset),
        )]),
        Spans::from(vec![Span::styled(
            format!(
                "Highest: {:.4} {unit}",
                data.iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&std::f64::NAN)
            ),
            Style::default().fg(Color::Reset),
        )]),
    ]
}

fn latency_throughput_chart<'a>(
    latency_throughput: &'a Vec<(f64, f64)>,
    batch_sizes: &'a Vec<u32>,
    name: &'static str,
) -> Chart<'a> {
    let latency_iter = latency_throughput.iter().map(|(l, _)| l);
    let throughput_iter = latency_throughput.iter().map(|(_, t)| t);

    let min_latency: f64 = *latency_iter
        .clone()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);
    let max_latency: f64 = *latency_iter
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);
    let min_throughput: f64 = *throughput_iter
        .clone()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);
    let max_throughput: f64 = *throughput_iter
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);

    let min_x = ((min_latency - 0.05 * min_latency) / 100.0).floor() * 100.0;
    let max_x = ((max_latency + 0.05 * max_latency) / 100.0).ceil() * 100.0;
    let step_x = (max_x - min_x) / 4.0;

    let min_y = ((min_throughput - 0.05 * min_throughput) / 100.0).floor() * 100.0;
    let max_y = ((max_throughput + 0.05 * max_throughput) / 100.0).ceil() * 100.0;
    let step_y = (max_y - min_y) / 4.0;

    let mut x_labels = vec![Span::styled(
        format!("{:.2}", min_x),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    )];
    for i in 0..3 {
        x_labels.push(Span::styled(
            format!("{:.2}", min_x + ((i + 1) as f64 * step_x)),
            Style::default().fg(Color::Gray).bg(Color::Reset),
        ));
    }
    x_labels.push(Span::styled(
        format!("{:.2}", max_x),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    ));

    let mut y_labels = vec![Span::styled(
        format!("{:.2}", min_y),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    )];
    for i in 0..3 {
        y_labels.push(Span::styled(
            format!("{:.2}", min_y + ((i + 1) as f64 * step_y)),
            Style::default().fg(Color::Gray).bg(Color::Reset),
        ));
    }
    y_labels.push(Span::styled(
        format!("{:.2}", max_y),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    ));

    let colors = color_vec();
    let datasets: Vec<Dataset> = (0..latency_throughput.len())
        .map(|i| {
            Dataset::default()
                .name(batch_sizes[i].to_string())
                .marker(symbols::Marker::Block)
                .style(Style::default().fg(colors[i]))
                .graph_type(GraphType::Scatter)
                .data(&latency_throughput[i..(i + 1)])
        })
        .collect();

    Chart::new(datasets)
        .style(Style::default().fg(Color::Cyan).bg(Color::Reset))
        .block(
            Block::default()
                .title(Span::styled(
                    format!("{name} throughput over latency"),
                    Style::default().fg(Color::Gray).bg(Color::Reset),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title(format!("ms"))
                .style(Style::default().fg(Color::Gray).bg(Color::Reset))
                .labels(x_labels)
                .bounds([min_x, max_x]),
        )
        .y_axis(
            Axis::default()
                .title(format!("tokens/secs"))
                .style(Style::default().fg(Color::Gray).bg(Color::Reset))
                .labels(y_labels)
                .bounds([min_y, max_y]),
        )
}

fn color_vec() -> Vec<Color> {
    vec![
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
        Color::Gray,
        Color::DarkGray,
        Color::LightRed,
        Color::LightGreen,
        Color::LightYellow,
        Color::LightBlue,
        Color::LightMagenta,
        Color::LightCyan,
    ]
}
