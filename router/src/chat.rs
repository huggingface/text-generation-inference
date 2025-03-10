use crate::{
    infer::InferError, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionDelta,
    ChatCompletionLogprobs, CompletionType, DeltaToolCall, Function, FunctionDefinition,
    StreamOptions, StreamResponse, TextMessage, ToolCallDelta, Usage,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum _NoTool {
    NoTool,
}

#[derive(Debug, Deserialize)]
struct NoToolCall {
    _name: _NoTool,
    content: String,
}
#[derive(Debug, Deserialize)]
struct NoTool {
    function: NoToolCall,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    _name: String,
    #[serde(flatten, default)]
    /// Using Map to preserve order
    arguments: serde_json::Map<String, Value>,
}
#[derive(Debug, Deserialize)]
struct Call {
    function: ToolCall,
}

pub(crate) fn parse_output(
    generated_text: &str,
) -> Result<(Option<Vec<crate::ToolCall>>, Option<String>), InferError> {
    let call: Call = serde_json::from_str(generated_text).map_err(|e| {
        InferError::ToolError(format!(
            "Failed to parse generated text: {} {:?}",
            e, generated_text
        ))
    })?;
    let name = call.function._name;

    match &name[..] {
        "no_tool" => {
            // parse the content message
            let content_message = call
                .function
                .arguments
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    InferError::ToolError("No `content` found in generated text".to_string())
                })?
                .to_string();
            Ok((None, Some(content_message)))
        }
        name => {
            let tool_calls = vec![crate::ToolCall {
                id: "0".to_string(),
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    description: None,
                    name: name.to_string(),
                    arguments: serde_json::to_value(call.function.arguments).map_err(|err| {
                        InferError::ToolError(format!(
                            "Could not convert arguments to JSON map {err}"
                        ))
                    })?,
                },
            }];
            Ok((Some(tool_calls), None))
        }
    }
}

/// Convert a StreamResponse into an Event to be sent over SSE
fn create_event_from_stream_token(
    stream_token: &StreamResponse,
    logprobs: bool,
    inner_using_tools: bool,
    system_fingerprint: String,
    model_id: String,
    function_name: Option<String>,
    id: String,
) -> CompletionType {
    let current_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs();

    let logprobs = logprobs.then(|| {
        ChatCompletionLogprobs::from((stream_token.token.clone(), stream_token.top_tokens.clone()))
    });

    // replace the content with the tool calls if grammar is present
    let content = if !stream_token.token.special {
        Some(stream_token.token.text.clone())
    } else {
        None
    };
    let (content, tool_calls) = if inner_using_tools {
        // Cast into a vec
        (None, content)
    } else {
        (content, None)
    };
    let finish_reason = stream_token
        .details
        .as_ref()
        .map(|details| details.finish_reason.format(true));
    let delta = match (content, tool_calls) {
        (Some(delta), _) => ChatCompletionDelta::Chat(TextMessage {
            role: "assistant".to_string(),
            content: delta,
            ..Default::default()
        }),
        (None, Some(tool_calls)) => ChatCompletionDelta::Tool(ToolCallDelta {
            role: "assistant".to_string(),
            tool_calls: vec![DeltaToolCall {
                index: 0,
                id,
                r#type: "function".to_string(),
                function: Function {
                    name: function_name,
                    arguments: tool_calls,
                },
            }],
        }),
        (None, None) => ChatCompletionDelta::Chat(TextMessage {
            role: "assistant".to_string(),
            content: "".to_string(),
            ..Default::default()
        }),
    };
    let choices = vec![ChatCompletionChoice {
        index: 0,
        delta,
        logprobs,
        finish_reason,
    }];
    CompletionType::ChatCompletionChunk(ChatCompletionChunk::new(
        model_id,
        system_fingerprint,
        current_time,
        choices,
        None,
    ))
}

#[derive(Debug)]
enum StreamState {
    /// Before the tools was parsed
    Buffering,
    /// We detected a tool call here
    Tool,
    /// During the `content` part of the tool call
    NoTool,
    /// Finishing frames of the ToolCall
    NoToolFinish,
    /// This is without tool calling
    Content,
}

pub struct ChatState {
    state: StreamState,
    text: String,
    options: StreamOptions,
    model_id: String,
    fingerprint: String,
    logprobs: bool,
    id: String,
}

impl ChatState {
    pub fn new(
        using_tools: bool,
        options: StreamOptions,
        fingerprint: String,
        model_id: String,
        logprobs: bool,
        id: String,
    ) -> Self {
        let state = if using_tools {
            StreamState::Buffering
        } else {
            StreamState::Content
        };
        let text = String::new();
        Self {
            state,
            text,
            options,
            fingerprint,
            model_id,
            logprobs,
            id,
        }
    }

    pub fn push(&mut self, mut stream_token: StreamResponse) -> Vec<CompletionType> {
        let mut events = vec![];
        let token_text = &stream_token.token.text;
        match self.state {
            StreamState::Buffering => {
                self.text.push_str(token_text);
                // We have a special match for `no_tool` in order to capture directly the `content`
                // key which should be re-emitted as raw text.
                if let Ok(value) = serde_json::from_str::<NoTool>(&format!("{}\"}}}}", self.text)) {
                    self.state = StreamState::NoTool;
                    // Modifiy the content of the token to be whatever was captured by the JSON
                    stream_token.token.text = value.function.content;
                    let chat_complete = create_event_from_stream_token(
                        &stream_token,
                        self.logprobs,
                        false,
                        self.fingerprint.clone(),
                        self.model_id.clone(),
                        None,
                        self.id.clone(),
                    );

                    events.push(chat_complete);
                }
                // XXX Caution, here we do not postfix the quote, so that the current output
                // Is necessarily finished with quotes for us to be able to parse.
                let partial = &self.text;
                let partial = partial.trim_end_matches(|c: char| c.is_whitespace() || c == ',');
                if let Ok(call) = serde_json::from_str::<Call>(&format!("{}}}}}", partial)) {
                    // This can be no_tool before the content has been emitted
                    if call.function._name != "no_tool" {
                        stream_token.token.text = "{".to_string();
                        let chat_complete = create_event_from_stream_token(
                            &stream_token,
                            self.logprobs,
                            true,
                            self.fingerprint.clone(),
                            self.model_id.clone(),
                            Some(call.function._name),
                            self.id.clone(),
                        );

                        events.push(chat_complete);
                        self.state = StreamState::Tool;
                    }
                }
            }
            StreamState::Tool => {
                self.text.push_str(token_text);
                if serde_json::from_str::<Call>(&self.text).is_ok() {
                    self.state = StreamState::Buffering;
                    let mut text = stream_token.token.text.trim_end();
                    // Effectively trimming only the last closing brace
                    if text.ends_with('}') {
                        text = &text[..text.len() - 1];
                    }
                    stream_token.token.text = text.to_string();
                    let chat_complete = create_event_from_stream_token(
                        &stream_token,
                        self.logprobs,
                        true,
                        self.fingerprint.clone(),
                        self.model_id.clone(),
                        None,
                        self.id.clone(),
                    );
                    events.push(chat_complete);
                } else {
                    let chat_complete = create_event_from_stream_token(
                        &stream_token,
                        self.logprobs,
                        true,
                        self.fingerprint.clone(),
                        self.model_id.clone(),
                        None,
                        self.id.clone(),
                    );
                    events.push(chat_complete);
                }
            }
            // if we skipped sending the buffer we need to avoid sending the following json key and quotes
            // We have remainder tokens, ignore everying,
            StreamState::NoToolFinish => {}
            StreamState::NoTool => {
                self.text.push_str(token_text);
                if token_text.contains("\"") {
                    let mut text = self
                        .text
                        .trim_end_matches(|c: char| c.is_whitespace() || c == '}');
                    // Trim once
                    if text.ends_with("\"") {
                        // Verify we have actually trimmed something
                        // The opposite can happen if the model is outputting inline JSON.
                        text = &text[..text.len() - 1];
                        if let Ok(_value) =
                            serde_json::from_str::<NoTool>(&format!("{}\"}}}}", text))
                        {
                            let mut text = token_text
                                .trim_end_matches(|c: char| c.is_whitespace() || c == '}');
                            // Effectively trim_end_match('"', 1)
                            // because we do not want to eventually trim finishing escaped quotes
                            // {{"\"Something\""}}
                            if text.ends_with("\"") {
                                text = &text[..text.len() - 1];
                            }
                            stream_token.token.text = text.to_string();
                            self.state = StreamState::NoToolFinish;
                        }
                    }
                }
                // This escaping is usually inline json escaping and we can therefore remove it.
                stream_token.token.text = stream_token.token.text.replace("\\", "");
                let chat_complete = create_event_from_stream_token(
                    &stream_token,
                    self.logprobs,
                    false,
                    self.fingerprint.clone(),
                    self.model_id.clone(),
                    None,
                    self.id.clone(),
                );

                events.push(chat_complete);
            }
            StreamState::Content => {
                let chat_complete = create_event_from_stream_token(
                    &stream_token,
                    self.logprobs,
                    false,
                    self.fingerprint.clone(),
                    self.model_id.clone(),
                    None,
                    self.id.clone(),
                );

                events.push(chat_complete);
            }
        }

        if self.options.include_usage {
            if let Some(details) = stream_token.details {
                let completion_tokens = details.generated_tokens;
                let prompt_tokens = details.input_length;
                let total_tokens = prompt_tokens + completion_tokens;

                let usage = Usage {
                    completion_tokens,
                    prompt_tokens,
                    total_tokens,
                };
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                    .as_secs();

                let chat_complete = CompletionType::ChatCompletionChunk(ChatCompletionChunk {
                    id: String::new(),
                    created: current_time,
                    model: self.model_id.clone(),
                    system_fingerprint: self.fingerprint.clone(),
                    choices: vec![],
                    usage: Some(Usage {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.total_tokens,
                    }),
                });

                events.push(chat_complete);
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ChatCompletionChoice, ChatCompletionDelta, FinishReason, StreamDetails, TextMessage, Token,
    };

    use super::*;

    fn get_text_content(event: &CompletionType) -> &String {
        match event {
            CompletionType::ChatCompletionChunk(ChatCompletionChunk { choices, .. }) => {
                assert_eq!(choices.len(), 1);
                if let ChatCompletionChoice {
                    delta: ChatCompletionDelta::Chat(TextMessage { content, .. }),
                    ..
                } = &choices[0]
                {
                    content
                } else {
                    panic!("Expected plain message");
                }
            }
            _ => panic!("Unexpected chunk"),
        }
    }

    fn get_tool_call_content(event: &CompletionType) -> (Option<&String>, &String) {
        match event {
            CompletionType::ChatCompletionChunk(ChatCompletionChunk { choices, .. }) => {
                assert_eq!(choices.len(), 1);
                if let ChatCompletionChoice {
                    delta: ChatCompletionDelta::Tool(ToolCallDelta { tool_calls, .. }),
                    ..
                } = &choices[0]
                {
                    assert_eq!(tool_calls.len(), 1);
                    let DeltaToolCall {
                        index,
                        id,
                        r#type,
                        function,
                    } = &tool_calls[0];
                    assert_eq!(*index, 0);
                    assert_eq!(id, "0");
                    assert_eq!(r#type, "function");
                    (function.name.as_ref(), &function.arguments)
                } else {
                    panic!("Expected plain message");
                }
            }
            _ => panic!("Unexpected chunk"),
        }
    }

    #[test]
    fn test_chat_stream() {
        let mut chat_state = ChatState::new(
            false,
            StreamOptions {
                include_usage: false,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let events = chat_state.push(StreamResponse {
            generated_text: None,
            token: Token {
                id: 42,
                text: "Hi".to_string(),
                logprob: 0.0,
                special: false,
            },
            top_tokens: vec![],
            index: 0,
            details: None,
        });
        assert_eq!(events.len(), 1);
        match &events[0] {
            CompletionType::ChatCompletionChunk(ChatCompletionChunk { choices, .. }) => {
                assert_eq!(
                    choices,
                    &[ChatCompletionChoice {
                        index: 0,
                        delta: ChatCompletionDelta::Chat(TextMessage {
                            role: "assistant".to_string(),
                            content: "Hi".to_string(),
                            tool_call_id: None,
                        }),
                        logprobs: None,
                        finish_reason: None,
                    }]
                );
            }
            _ => panic!("Unexpected chunk"),
        }
    }

    #[test]
    fn test_chat_stream_usage() {
        let mut chat_state = ChatState::new(
            false,
            StreamOptions {
                include_usage: true,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let events = chat_state.push(StreamResponse {
            generated_text: None,
            token: Token {
                id: 42,
                text: "Hi".to_string(),
                logprob: 0.0,
                special: false,
            },
            top_tokens: vec![],
            index: 0,
            details: Some(StreamDetails {
                input_length: 2,
                generated_tokens: 10,
                seed: None,
                finish_reason: FinishReason::Length,
            }),
        });
        assert_eq!(events.len(), 2);
        match &events[0] {
            CompletionType::ChatCompletionChunk(ChatCompletionChunk { choices, .. }) => {
                assert_eq!(
                    choices,
                    &[ChatCompletionChoice {
                        index: 0,
                        delta: ChatCompletionDelta::Chat(TextMessage {
                            role: "assistant".to_string(),
                            content: "Hi".to_string(),
                            tool_call_id: None,
                        }),
                        logprobs: None,
                        // HAS A FINISH REASON
                        finish_reason: Some("length".to_string()),
                    }]
                );
            }
            _ => panic!("Unexpected chunk"),
        }
        match &events[1] {
            CompletionType::ChatCompletionChunk(ChatCompletionChunk { usage, .. }) => {
                assert_eq!(
                    *usage,
                    Some(Usage {
                        prompt_tokens: 2,
                        completion_tokens: 10,
                        total_tokens: 12,
                    })
                );
            }
            _ => panic!("Unexpected chunk"),
        }
    }

    #[test]
    fn test_chat_stream_tool_no_tool() {
        let mut chat_state = ChatState::new(
            true,
            StreamOptions {
                include_usage: true,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let tokens = vec![
            "{\"".to_string(),
            "function".to_string(),
            "\":".to_string(),
            " {\"".to_string(),
            "_".to_string(),
            "name".to_string(),
            "\":".to_string(),
            " \"".to_string(),
            "no".to_string(),
            "_tool".to_string(),
            "\",".to_string(),
            " \"".to_string(),
            "content".to_string(),
            "\":".to_string(),
            " \"".to_string(),        // Token 14
            "I".to_string(),          // Event 1
            " am".to_string(),        // Event 2
            " a".to_string(),         // Event 3
            " helpful".to_string(),   // Event 4
            " assistant".to_string(), // Event 5
            "!\"".to_string(),        // Event 6 (with trailing quore removed)
            "}".to_string(),
            "}".to_string(),
        ];
        let tokens: Vec<_> = tokens
            .into_iter()
            .map(|text| StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            })
            .collect();

        // Initial ignored output
        for token in &tokens[..14] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0);
        }

        // No tool output
        let mut output = String::new();
        for token in &tokens[14..14 + 7] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 1);
            let content = get_text_content(&events[0]);
            output.push_str(content);
        }

        assert_eq!(output, "I am a helpful assistant!");

        // No tool finish
        for token in &tokens[14 + 7..] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0);
        }
    }

    #[test]
    fn test_chat_stream_tool_no_tool_many_quotes() {
        let mut chat_state = ChatState::new(
            true,
            StreamOptions {
                include_usage: true,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let tokens = vec![
            "{\"".to_string(),
            "function".to_string(),
            "\":".to_string(),
            " {\"".to_string(),
            "_".to_string(),
            "name".to_string(),
            "\":".to_string(),
            " \"".to_string(),
            "no".to_string(),
            "_tool".to_string(),
            "\",".to_string(),
            " \"".to_string(),
            "content".to_string(),
            "\":".to_string(),
            " \"".to_string(),        // Token 14
            "I".to_string(),          // Event 1
            " am".to_string(),        // Event 2
            " a".to_string(),         // Event 3
            " helpful".to_string(),   // Event 4
            " assistant".to_string(), // Event 5
            "!\\\"\"".to_string(),    // Extra inside the string quote that would get removed
            "}".to_string(),
            "}".to_string(),
        ];

        // Initial ignored output
        for text in &tokens[..14] {
            let events = chat_state.push(StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            });
            assert_eq!(events.len(), 0);
        }

        // No tool output
        let mut output = String::new();
        for text in &tokens[14..14 + 7] {
            let events = chat_state.push(StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            });
            assert_eq!(events.len(), 1);
            match &events[0] {
                CompletionType::ChatCompletionChunk(ChatCompletionChunk { choices, .. }) => {
                    assert_eq!(choices.len(), 1);
                    if let ChatCompletionChoice {
                        delta: ChatCompletionDelta::Chat(TextMessage { content, .. }),
                        ..
                    } = &choices[0]
                    {
                        output.push_str(content);
                    } else {
                        panic!("Expected plain message");
                    }
                }
                _ => panic!("Unexpected chunk"),
            }
        }

        assert_eq!(output, "I am a helpful assistant!\"");

        // No tool finish
        for text in &tokens[14 + 7..] {
            let events = chat_state.push(StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            });
            assert_eq!(events.len(), 0);
        }
    }

    #[test]
    fn test_chat_stream_tool_no_tool_inline_json() {
        let mut chat_state = ChatState::new(
            true,
            StreamOptions {
                include_usage: true,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let tokens = vec![
            "{\"".to_string(),
            "function".to_string(),
            "\":".to_string(),
            " {\"".to_string(),
            "_".to_string(),
            "name".to_string(),
            "\":".to_string(),
            " \"".to_string(),
            "no".to_string(),
            "_tool".to_string(),
            "\",".to_string(),
            " \"".to_string(),
            "content".to_string(),
            "\":".to_string(),
            " \"".to_string(),    // Token 14
            "{\\\"".to_string(),  // Event 1
            "a".to_string(),      // Event 1
            "\\\":".to_string(),  // Event 1
            "2".to_string(),      // Event 2
            ",\\".to_string(),    // Event 2
            "\"".to_string(),     // Event 2
            "b".to_string(),      // Event 3
            "\\\": ".to_string(), // Event 4
            "1".to_string(),      // Event 5
            "}".to_string(),      // Event 5
            "\"}".to_string(),    // Extra inside the string quote that would get removed
            "}".to_string(),
        ];
        let tokens: Vec<_> = tokens
            .into_iter()
            .map(|text| StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            })
            .collect();

        // Initial ignored output
        for token in &tokens[..14] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0);
        }

        // No tool output
        let mut output = String::new();
        for token in &tokens[14..14 + 12] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 1, "Current text is {output:?}");
            let content = get_text_content(&events[0]);
            output.push_str(content);
        }

        assert_eq!(output, "{\"a\":2,\"b\": 1}");

        // No tool finish
        for token in &tokens[14 + 12..] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0, "Extra events {events:?}");
        }
    }

    #[test]
    fn test_chat_stream_tool_no_tool_empty() {
        let mut chat_state = ChatState::new(
            true,
            StreamOptions {
                include_usage: true,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let tokens = vec![
            "{\"".to_string(),
            "function".to_string(),
            "\":".to_string(),
            " {\"".to_string(),
            "_".to_string(),
            "name".to_string(),
            "\":".to_string(),
            " \"".to_string(),
            "no".to_string(),
            "_tool".to_string(),
            "\",".to_string(),
            " \"".to_string(),
            "content".to_string(),
            "\":\"".to_string(),
            "\"}".to_string(), // Token 13
            "}".to_string(),   // Event 1
        ];
        let tokens: Vec<_> = tokens
            .into_iter()
            .map(|text| StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            })
            .collect();

        // Initial ignored output
        for token in &tokens[..13] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0);
        }

        // No tool output
        let mut output = String::new();
        for token in &tokens[13..13 + 2] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 1, "Current text is {output:?}");
            let content = get_text_content(&events[0]);
            output.push_str(content);
        }

        assert_eq!(output, "");

        // No tool finish
        for token in &tokens[13 + 2..] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0, "Extra events {events:?}");
        }
    }

    #[test]
    fn test_chat_stream_tool_get_weather() {
        let mut chat_state = ChatState::new(
            true,
            StreamOptions {
                include_usage: true,
            },
            "fingerprint".to_string(),
            "model_id".to_string(),
            false,
            "0".to_string(),
        );

        let tokens = vec![
            "{\"".to_string(),
            "function".to_string(),
            "\":".to_string(),
            " {\"".to_string(),
            "_".to_string(),
            "name".to_string(),
            "\":".to_string(),
            " \"".to_string(),
            "get".to_string(),
            "_current".to_string(),
            "_weather".to_string(),
            "\",".to_string(),
            // Event 1 is the function name
            // Event 2 is the start of the arguments "{"
            " \"".to_string(),        // Event 3
            "location".to_string(),   // Event 4
            "\":".to_string(),        // Event 5
            " \"".to_string(),        // Event 6
            "San".to_string(),        // Event 7
            " Francisco".to_string(), // Event 8
            ",".to_string(),          // Event 9
            " CA".to_string(),        // Event 10
            "\",".to_string(),        // Event 11
            " \"".to_string(),        // Event 12
            "format".to_string(),     // Event 13
            "\":".to_string(),        // Event 14
            " \"".to_string(),        // Event 15
            "c".to_string(),          // Event 16
            "elsius".to_string(),     // Event 17
            "\"}}".to_string(),       // Event 18 retained (trailing brace removed)
        ];
        let tokens: Vec<_> = tokens
            .into_iter()
            .map(|text| StreamResponse {
                generated_text: None,
                token: Token {
                    id: 42,
                    text: text.to_string(),
                    logprob: 0.0,
                    special: false,
                },
                top_tokens: vec![],
                index: 0,
                details: None,
            })
            .collect();

        // Initial ignored output
        for token in &tokens[..11] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0, "{events:?}");
        }

        // No tool output
        let mut output = String::new();
        let mut output_name = String::new();
        for token in &tokens[11..11 + 17] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 1);
            let (name, arguments) = get_tool_call_content(&events[0]);
            if let Some(name) = name {
                assert_eq!(name, "get_current_weather");
                output_name.push_str(&name);
            }
            output.push_str(arguments);
        }

        assert_eq!(output_name, "get_current_weather");
        assert_eq!(
            output,
            "{ \"location\": \"San Francisco, CA\", \"format\": \"celsius\"}"
        );

        // No tool finish
        for token in &tokens[11 + 17..] {
            let events = chat_state.push(token.clone());
            assert_eq!(events.len(), 0);
        }
    }
}
