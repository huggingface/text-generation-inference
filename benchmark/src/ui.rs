use crate::generation::{Decode, Message, Prefill};
/// Inspired by https://github.com/hatoo/oha/blob/master/src/monitor.rs
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use text_generation_client::ClientError;
use tokio::sync::mpsc;
use tui::backend::Backend;
use tui::layout::{Alignment, Constraint, Direction, Layout};
use tui::style::{Color, Modifier, Style};
use tui::text::{Span, Spans};
use tui::widgets::{
    Axis, BarChart, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Tabs,
};
use tui::{symbols, Frame};

struct Data {
    prefill_latencies: Vec<Vec<f64>>,
    prefill_throughputs: Vec<Vec<f64>>,
    decode_latencies: Vec<Vec<f64>>,
    decode_throughputs: Vec<Vec<f64>>,
    prefill_batch_latency_throughput: Vec<(f64, f64)>,
    decode_batch_latency_throughput: Vec<(f64, f64)>,
}

impl Data {
    fn new(n_run: usize, n_batch: usize) -> Self {
        let prefill_latencies: Vec<Vec<f64>> =
            (0..n_batch).map(|_| Vec::with_capacity(n_run)).collect();
        let prefill_throughputs: Vec<Vec<f64>> =
            (0..n_batch).map(|_| Vec::with_capacity(n_run)).collect();

        let decode_latencies: Vec<Vec<f64>> =
            (0..n_batch).map(|_| Vec::with_capacity(n_run)).collect();
        let decode_throughputs: Vec<Vec<f64>> =
            (0..n_batch).map(|_| Vec::with_capacity(n_run)).collect();

        let prefill_batch_latency_throughput: Vec<(f64, f64)> = Vec::with_capacity(n_batch);

        let decode_batch_latency_throughput: Vec<(f64, f64)> = Vec::with_capacity(n_batch);

        Self {
            prefill_latencies,
            prefill_throughputs,
            decode_latencies,
            decode_throughputs,
            prefill_batch_latency_throughput,
            decode_batch_latency_throughput,
        }
    }

    fn push_prefill(&mut self, prefill: Prefill, batch_idx: usize) {
        let latency = prefill.latency.as_millis() as f64;
        self.prefill_latencies[batch_idx].push(latency);
        self.prefill_throughputs[batch_idx].push(prefill.throughput);
    }

    fn push_decode(&mut self, prefill: Decode, batch_idx: usize) {
        let latency = prefill.latency.as_millis() as f64;
        self.decode_latencies[batch_idx].push(latency);
        self.decode_throughputs[batch_idx].push(prefill.throughput);
    }

    fn end_batch(&mut self, batch_idx: usize) {
        self.prefill_batch_latency_throughput.push((
            self.prefill_latencies[batch_idx].iter().sum::<f64>()
                / self.prefill_latencies[batch_idx].len() as f64,
            self.prefill_throughputs[batch_idx].iter().sum::<f64>()
                / self.prefill_throughputs[batch_idx].len() as f64,
        ));
        self.decode_batch_latency_throughput.push((
            self.decode_latencies[batch_idx].iter().sum::<f64>()
                / self.decode_latencies[batch_idx].len() as f64,
            self.decode_throughputs[batch_idx].iter().sum::<f64>()
                / self.decode_throughputs[batch_idx].len() as f64,
        ));
    }
}

pub(crate) struct UI {
    pub(crate) running: bool,
    completed_runs: Vec<usize>,
    completed_batch: usize,
    current_batch: usize,
    current_tab: usize,
    zoom: bool,
    is_error: bool,
    data: Data,
    tokenizer_name: String,
    sequence_length: u32,
    decode_length: u32,
    n_run: usize,
    batch_size: Vec<u32>,
    receiver: mpsc::Receiver<Result<Message, ClientError>>,
}

impl UI {
    pub(crate) fn new(
        receiver: mpsc::Receiver<Result<Message, ClientError>>,
        tokenizer_name: String,
        sequence_length: u32,
        decode_length: u32,
        n_run: usize,
        batch_size: Vec<u32>,
    ) -> Self {
        let data = Data::new(n_run, batch_size.len());
        let current_tab = 0;

        let completed_runs: Vec<usize> = (0..batch_size.len()).map(|_| 0).collect();
        let completed_batch = 0;
        let current_batch = 0;
        let is_error = false;

        Self {
            running: true,
            completed_runs,
            completed_batch,
            current_batch,
            current_tab,
            zoom: false,
            is_error,
            data,
            tokenizer_name,
            sequence_length,
            decode_length,
            n_run,
            batch_size,
            receiver,
        }
    }

    pub(crate) fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event {
            KeyEvent {
                code: KeyCode::Right,
                ..
            } |
            KeyEvent {
                code: KeyCode::Tab,
                ..
            } => {
                self.current_tab = (self.current_tab + 1) % self.batch_size.len();
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                if self.current_tab > 0 {
                    self.current_tab -= 1;
                } else {
                    self.current_tab = self.batch_size.len() - 1;
                }
            }
            KeyEvent {
                code: KeyCode::Char('+'),
                ..
            } => {
                self.zoom = true;
            }
            KeyEvent {
                code: KeyCode::Char('-'),
                ..
            } => {
                self.zoom = false;
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
                self.running = false;
            }
            _ => (),
        }
    }

    pub(crate) fn tick(&mut self) {
        while let Ok(message) = self.receiver.try_recv() {
            match message {
                Ok(message) => match message {
                    Message::Prefill(step) => self.data.push_prefill(step, self.current_batch),
                    Message::Decode(step) => self.data.push_decode(step, self.current_batch),
                    Message::Run(_) => {
                        self.completed_runs[self.current_batch] += 1;
                    }
                    Message::EndBatch => {
                        self.data.end_batch(self.current_batch);

                        self.completed_batch += 1;
                        if self.current_batch < self.batch_size.len() - 1 {
                            self.current_batch += 1;
                        }
                    }
                    Message::Warmup => {}
                },
                Err(_) => self.is_error = true,
            }
        }
    }

    pub fn render<B: Backend>(&mut self, f: &mut Frame<'_, B>) {
        let batch_progress =
            (self.completed_batch as f64 / self.batch_size.len() as f64).clamp(0.0, 1.0);
        let run_progress =
            (self.completed_runs[self.current_batch] as f64 / self.n_run as f64).clamp(0.0, 1.0);

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
        let title = Block::default()
            .borders(Borders::NONE)
            .title(format!(
                "Model: {} | Sequence Length: {} | Decode Length: {}",
                self.tokenizer_name, self.sequence_length, self.decode_length
            ))
            .style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::White),
            );
        f.render_widget(title, row5[0]);

        // Helper
        let helper = Block::default()
            .borders(Borders::NONE)
            .title(format!(
                "<- | tab | ->: change batch tab | q / CTRL + c: quit | +/-: zoom"
            ))
            .title_alignment(Alignment::Right)
            .style(
                Style::default()
                    .fg(Color::White),
            );
        f.render_widget(helper, row5[0]);

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
            .select(self.current_tab)
            .style(Style::default().fg(Color::LightCyan))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::Black),
            );
        f.render_widget(tabs, row5[1]);

        // Total progress bar
        let color = if self.is_error {
            Color::Red
        } else {
            Color::LightGreen
        };
        let batch_gauge = progress_gauge(
            "Total Progress",
            format!("{} / {}", self.completed_batch, self.batch_size.len()),
            batch_progress,
            color,
        );
        f.render_widget(batch_gauge, top[0]);

        // Batch progress Bar
        let color = if self.is_error {
            Color::Red
        } else {
            Color::LightBlue
        };
        let run_gauge = progress_gauge(
            "Batch Progress",
            format!(
                "{} / {}",
                self.completed_runs[self.current_batch], self.n_run
            ),
            run_progress,
            color,
        );
        f.render_widget(run_gauge, top[1]);

        // Prefill text infos
        let (prefill_latency_statics, prefill_throughput_statics) = text_info(
            &mut self.data.prefill_latencies[self.current_tab],
            &self.data.prefill_throughputs[self.current_tab],
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

        let histo_data =
            latency_histogram_data(&self.data.prefill_latencies[self.current_tab], bins);
        let histo_data_str: Vec<(&str, u64)> =
            histo_data.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let prefill_histogram =
            latency_histogram(&histo_data_str, "Prefill").bar_width(histo_width as u16);
        f.render_widget(prefill_histogram, mid[1]);

        // Decode text info
        let (decode_latency_statics, decode_throughput_statics) = text_info(
            &mut self.data.decode_latencies[self.current_tab],
            &self.data.decode_throughputs[self.current_tab],
            "Decode",
        );
        f.render_widget(decode_latency_statics, decode_text[0]);
        f.render_widget(decode_throughput_statics, decode_text[1]);

        // Decode latency histogram
        let histo_data =
            latency_histogram_data(&self.data.decode_latencies[self.current_tab], bins);
        let histo_data_str: Vec<(&str, u64)> =
            histo_data.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let decode_histogram =
            latency_histogram(&histo_data_str, "Decode").bar_width(histo_width as u16);
        f.render_widget(decode_histogram, mid[3]);

        // Prefill latency/throughput chart
        let prefill_latency_throughput_chart = latency_throughput_chart(
            &self.data.prefill_batch_latency_throughput,
            &self.batch_size,
            self.zoom,
            "Prefill",
        );
        f.render_widget(prefill_latency_throughput_chart, bottom[0]);

        // Decode latency/throughput chart
        let decode_latency_throughput_chart = latency_throughput_chart(
            &self.data.decode_batch_latency_throughput,
            &self.batch_size,
            self.zoom,
            "Decode",
        );
        f.render_widget(decode_latency_throughput_chart, bottom[1]);
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
    zoom: bool,
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

    let min_x = if zoom {
        ((min_latency - 0.05 * min_latency) / 100.0).floor() * 100.0
    } else {
        0.0
    };
    let max_x = ((max_latency + 0.05 * max_latency) / 100.0).ceil() * 100.0;
    let step_x = (max_x - min_x) / 4.0;

    let min_y = if zoom {
        ((min_throughput - 0.05 * min_throughput) / 100.0).floor() * 100.0
    } else {
        0.0
    };
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
