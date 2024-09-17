use crate::infer::InferError;
use crate::{ChatTemplateInputs, Message, MessageChunk, TextMessage, TokenizerConfigToken, Tool};
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;
use std::collections::HashSet;

/// Raise a exception (custom function) used in the chat templates
pub(crate) fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

#[derive(Clone)]
pub(crate) struct ChatTemplate {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    use_default_tool_template: bool,
    variables: HashSet<String>,
}

impl ChatTemplate {
    pub(crate) fn new(
        template: String,
        bos_token: Option<TokenizerConfigToken>,
        eos_token: Option<TokenizerConfigToken>,
    ) -> Self {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        let template_str = template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);
        tracing::debug!("Loading template: {:#?}", template_str);

        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        // get the list of variables that are used in the template
        let variables = template.undeclared_variables(true);
        // check if the `tools` variable is used in the template
        let use_default_tool_template = !variables.contains("tools");
        tracing::debug!("Use default tool template: {}", use_default_tool_template);

        Self {
            template,
            bos_token: bos_token.map(|token| token.as_str().to_string()),
            eos_token: eos_token.map(|token| token.as_str().to_string()),
            use_default_tool_template,
            variables,
        }
    }

    pub(crate) fn apply(
        &self,
        guideline: Option<&str>,
        mut messages: Vec<Message>,
        tools_and_prompt: Option<(Vec<Tool>, String)>,
    ) -> Result<String, InferError> {
        // check if guideline is expected but not provided
        if self.variables.contains("guideline") && guideline.is_none() {
            return Err(InferError::MissingTemplateVariable("guideline".to_string()));
        }

        let tools = match tools_and_prompt {
            Some((tools, tool_prompt)) => {
                // check if the `tools` variable is used in the template
                // if not, we need to append the tools to the last message
                let text = if self.use_default_tool_template {
                    match serde_json::to_string(&tools) {
                        Ok(tools_str) => format!("\n---\n{}\n{}", tools_str, tool_prompt),
                        Err(e) => return Err(InferError::ToolError(e.to_string())),
                    }
                } else {
                    // if the `tools` variable is used in the template, we just append the tool_prompt
                    format!("\n---\n{}", tool_prompt)
                };
                if let Some(last_message) = messages.last_mut() {
                    last_message.content.push(MessageChunk::Text { text });
                }
                Some(tools)
            }
            None => None,
        };

        let messages: Vec<TextMessage> = messages.into_iter().map(|c| c.into()).collect();

        self.template
            .render(ChatTemplateInputs {
                guideline,
                messages,
                bos_token: self.bos_token.as_deref(),
                eos_token: self.eos_token.as_deref(),
                add_generation_prompt: true,
                tools,
            })
            .map_err(InferError::TemplateError)
    }
}

// tests
#[cfg(test)]
mod tests {
    use crate::infer::chat_template::raise_exception;
    use crate::infer::ChatTemplate;
    use crate::{
        ChatTemplateInputs, Message, MessageContent, TextMessage, TokenizerConfigToken, Tool,
    };
    use minijinja::Environment;

    #[test]
    fn test_chat_template() {
        let env = Environment::new();

        let source = r#"
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {% if message['content']%}
                    {{'### System:\n' + message['content']+'\n\n'}}
                {% endif %}
            {% elif message['role'] == 'user' %}
                {{'### User:\n' + message['content']+'\n\n'}}
            {% elif message['role'] == 'assistant' %}
                {{'### Assistant:\n'  + message['content']}}
            {% endif %}
            {% if loop.last and add_generation_prompt %}
                {{ '### Assistant:\n' }}
            {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();

        assert_eq!(
            result,
            "### User:\nHi!\n\n### Assistant:\nHello how can I help?### User:\nWhat is Deep Learning?\n\n### Assistant:\nmagic!### Assistant:\n"
        );
    }

    #[test]
    fn test_chat_template_invalid_with_raise() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let source = r#"
        {{ bos_token }}
        {% for message in messages %}
        {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {% endif %}
        {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + ' [/INST]' }}
        {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token}}
        {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
        {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi again!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs); //.err().unwrap();

        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => {
                assert_eq!(
                    e.detail().unwrap(),
                    "Conversation roles must alternate user/assistant/user/assistant/..."
                );
            }
        }
    }

    #[test]
    fn test_chat_template_valid_with_raise() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let source = r#"
        {{ bos_token }}
        {% for message in messages %}
        {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {% endif %}
        {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + ' [/INST]' }}
        {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token}}
        {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
        {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();
        assert_eq!(result, "[BOS][INST] Hi! [/INST]Hello how can I help?[EOS][INST] What is Deep Learning? [/INST]magic![EOS]");
    }

    #[test]
    fn test_chat_template_valid_with_add_generation_prompt() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let source = r#"
        {% for message in messages %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
        {% endfor %}
        {% if add_generation_prompt %}
            {{ '<|im_start|>assistant\n' }}
        {% endif %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();
        assert_eq!(result, "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHello how can I help?<|im_end|>\n<|im_start|>user\nWhat is Deep Learning?<|im_end|>\n<|im_start|>assistant\nmagic!<|im_end|>\n<|im_start|>assistant\n");
    }

    struct ChatTemplateTestItem {
        name: &'static str,
        chat_template: &'static str,
        input: ChatTemplateInputs<'static>,
        target: &'static str,
    }

    #[test]
    fn test_many_chat_templates() {
        let example_chat = vec![
            TextMessage {
                role: "user".to_string(),
                content: "Hello, how are you?".to_string(),
            },
            TextMessage {
                role: "assistant".to_string(),
                content: "I'm doing great. How can I help you today?".to_string(),
            },
            TextMessage {
                role: "user".to_string(),
                content: "I'd like to show off how chat templating works!".to_string(),
            },
        ];

        let example_chat_with_system = [TextMessage {
            role: "system".to_string(),
            content: "You are a friendly chatbot who always responds in the style of a pirate"
                .to_string(),
        }]
        .iter()
        .chain(&example_chat)
        .cloned()
        .collect::<Vec<_>>();

        let test_default_templates = vec![
            ChatTemplateTestItem {
                name: "_base",
                chat_template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some(""),
                    ..Default::default()
                },
                target: "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "blenderbot",
                chat_template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: " Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "blenderbot_small",
                chat_template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: " Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "bloom",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Hello, how are you?</s>I'm doing great. How can I help you today?</s>I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "gpt_neox",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>",
            },
            ChatTemplateTestItem {
                name: "gpt2",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>",
            },
            ChatTemplateTestItem {
                name: "llama",
                // NOTE: the `.strip()` has been replaced with `| trim` in the following template
                chat_template: "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token +'[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content | trim + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: true,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>[INST] <<SYS>>\nYou are a friendly chatbot who always responds in the style of a pirate\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "whisper",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: true,
                    bos_token: Some(""),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>",
            },
        ];

        #[allow(unused_variables)] // name is unused
        for ChatTemplateTestItem {
            name,
            chat_template,
            input,
            target,
        } in test_default_templates
        {
            let mut env = Environment::new();
            env.add_function("raise_exception", raise_exception);
            let tmpl = env.template_from_str(chat_template);
            let result = tmpl.unwrap().render(input).unwrap();
            assert_eq!(result, target);
        }

        let test_custom_templates = vec![
            ChatTemplateTestItem {
                name: "HuggingFaceH4/zephyr-7b-beta (add_generation_prompt=false)",
                chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s><|user|>\nHello, how are you?</s><|assistant|>\nI'm doing great. How can I help you today?</s><|user|>\nI'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "HuggingFaceH4/zephyr-7b-beta (add_generation_prompt=true)",
                chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: vec![
                        TextMessage {
                            role: "system".to_string(),
                            content: "You are a friendly chatbot who always responds in the style of a pirate".to_string(),
                        },
                        TextMessage {
                            role: "user".to_string(),
                            content: "How many helicopters can a human eat in one sitting?".to_string(),
                        },
                    ],
                    add_generation_prompt: true,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s><|user|>\nHow many helicopters can a human eat in one sitting?</s><|assistant|>",
            },
            ChatTemplateTestItem {
                name: "HuggingFaceH4/zephyr-7b-gemma-v0.1",
                chat_template: "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<bos>"),
                    eos_token: Some("<eos>"),
                    ..Default::default()
                },
                target: "<bos><|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "mistralai/Mistral-7B-Instruct-v0.1",
                chat_template: "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "mistralai/Mixtral-8x7B-Instruct-v0.1",
                chat_template: "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s>[INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "openchat/openchat-3.5-0106",
                // `.title()` has been replaced with `| upper` in the following template
                chat_template: "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + (message['role'] | title) + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>GPT4 Correct User: Hello, how are you?<|end_of_turn|>GPT4 Correct Assistant: I'm doing great. How can I help you today?<|end_of_turn|>GPT4 Correct User: I'd like to show off how chat templating works!<|end_of_turn|>",
            },
            ChatTemplateTestItem {
                name: "upstage/SOLAR-10.7B-Instruct-v1.0",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Hello, how are you?</s>I'm doing great. How can I help you today?</s>I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "codellama/CodeLlama-70b-Instruct-hf",
                // NOTE: `.strip()` has been replaced with `| trim` in the following template
                chat_template: "{% if messages[0]['role'] == 'system' %}{% set user_index = 1 %}{% else %}{% set user_index = 0 %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + user_index) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ '<s>' }}{% endif %}{% set content = 'Source: ' + message['role'] + '\\n\\n ' + message['content'] | trim %}{{ content + ' <step> ' }}{% endfor %}{{'Source: assistant\\nDestination: user\\n\\n '}}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>Source: user\n\n Hello, how are you? <step> Source: assistant\n\n I'm doing great. How can I help you today? <step> Source: user\n\n I'd like to show off how chat templating works! <step> Source: assistant\nDestination: user\n\n ",
            },
            ChatTemplateTestItem {
                name: "Deci/DeciLM-7B-instruct",
                chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '### User:\\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '### System:\\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '### Assistant:\\n'  + message['content'] }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### Assistant:' }}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "### User:\nHello, how are you?### Assistant:\nI'm doing great. How can I help you today?### User:\nI'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "Qwen/Qwen1.5-72B-Chat",
                chat_template: "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "deepseek-ai/deepseek-llm-7b-chat",
                chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<｜begin▁of▁sentence｜>"),
                    eos_token: Some("<｜end▁of▁sentence｜>"),
                    ..Default::default()
                },
                target: "<｜begin▁of▁sentence｜>User: Hello, how are you?\n\nAssistant: I'm doing great. How can I help you today?<｜end▁of▁sentence｜>User: I'd like to show off how chat templating works!\n\n",
            },
            ChatTemplateTestItem {
                name: "h2oai/h2o-danube-1.8b-chat",
                chat_template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompt|>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|answer|>'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|answer|>' }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|prompt|>Hello, how are you?</s><|answer|>I'm doing great. How can I help you today?</s><|prompt|>I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "internlm/internlm2-chat-7b",
                chat_template: "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s><|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "TheBloke/deepseek-coder-33B-instruct-AWQ",
                chat_template: "{%- set found_item = false -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set found_item = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not found_item -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response:\\n'}}\n",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<｜begin▁of▁sentence｜>"),
                    eos_token: Some("<|EOT|>"),
                    ..Default::default()
                },
                target: "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\nHello, how are you?\n### Response:\nI'm doing great. How can I help you today?\n<|EOT|>\n### Instruction:\nI'd like to show off how chat templating works!\n### Response:\n",
            },
            ChatTemplateTestItem {
                name: "ericzzz/falcon-rw-1b-chat",
                // `.strip()` has been replaced with `| trim` in the following template
                chat_template: "{% for message in messages %}{% if loop.index > 1 and loop.previtem['role'] != 'assistant' %}{{ ' ' }}{% endif %}{% if message['role'] == 'system' %}{{ '[SYS] ' + message['content'] | trim }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim }}{% elif message['role'] == 'assistant' %}{{ '[RESP] '  + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' [RESP] ' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<|endoftext|>"),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "[INST] Hello, how are you? [RESP] I'm doing great. How can I help you today?<|endoftext|>[INST] I'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "abacusai/Smaug-34B-v0.1",
                chat_template: "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "maywell/Synatra-Mixtral-8x7B",
                chat_template: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'system' %}{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\n### Response:\n{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Below is an instruction that describes a task. Write a response that appropriately completes the request.### Instruction:Hello, how are you?### Response:I'm doing great. How can I help you today?### Instruction:I'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "deepseek-ai/deepseek-coder-33b-instruct",
                chat_template: "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<｜begin▁of▁sentence｜>"),
                    eos_token: Some("</EOT>"),
                    ..Default::default()
                },
                target: "<｜begin▁of▁sentence｜>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\nHello, how are you?\n### Response:\nI'm doing great. How can I help you today?\n<|EOT|>\n### Instruction:\nI'd like to show off how chat templating works!\n",
            },
            // NOT INCLUDED
            // - meetkai/functionary-medium-v3.2
            // - fireworks-ai/firefunction-v1
            // https://github
            ChatTemplateTestItem {
                name: "maywell/PiVoT-MoE",
                chat_template: "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content']|trim }}{% elif message['role'] == 'user' %}### Instruction: {{ message['content']|trim }}{% elif message['role'] == 'assistant' %}### Response: {{ message['content']|trim }}{% elif message['role'] == 'user_context' %}### Input: {{ message['content']|trim }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}### Response:{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "You are a friendly chatbot who always responds in the style of a pirateYou are a friendly chatbot who always responds in the style of a pirate### Instruction: Hello, how are you?### Response: I'm doing great. How can I help you today?### Instruction: I'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "google/shieldgemma-9b",
                chat_template: "{{- bos_token }}\n{%- if messages[-1].role == 'user' %}\n    {{- \"You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\\n\\n\" }}\n    {{- \"<start_of_turn>\\n\" }}\n    {{- \"Human Question: \" + messages[-1].content }}\n    {{- \"\\n<end_of_turn>\\n\\n\" }}\n    {{- \"Our safety principle is defined in the below:\\n\\n\" }}\n    {{- \"* \" + guideline + \"\\n\" }}\n    {{- \"\\n===\\n\\n\" }}\n    {{- \"Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\\n\\n\" }}\n{%- else %}\n    {{- \"You are a policy expert trying to help determine whether an AI response to a prompt is in violation of the defined safety policies.\\n\\n\" }}\n    {{- \"<start_of_turn>\\n\" }}\n    {{- \"Human Question: \" + messages[-2].content }}\n    {{- \"\\n<end_of_turn>\\n\" }}\n    {{- \"<start_of_turn>\\n\" }}\n    {{- \"Chatbot Response: \" + messages[-1].content }}\n    {{- \"\\n<end_of_turn>\\n\\n\" }}\n    {{- \"Our safety principle is defined in the below:\\n\\n\" }}\n    {{- \"* \" + guideline + \"\\n\" }}\n    {{- \"\\n===\\n\\n\" }}\n    {{- \"Does the Chatbot Response violate the above principle? Your answer must  start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\\n\\n\" }}\n{%- endif %}\n\n",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    guideline: Some("Do not use offensive language."),
                    ..Default::default()
                },
                target: "<s>You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\n\n<start_of_turn>\nHuman Question: I'd like to show off how chat templating works!\n<end_of_turn>\n\nOur safety principle is defined in the below:\n\n* Do not use offensive language.\n\n===\n\nDoes the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\n\n",
            },
        ];

        #[allow(unused_variables)] // name is unused
        for ChatTemplateTestItem {
            name,
            chat_template,
            input,
            target,
        } in test_custom_templates
        {
            let mut env = Environment::new();
            env.add_function("raise_exception", raise_exception);
            // trim all the whitespace
            let chat_template = chat_template
                .lines()
                .map(|line| line.trim())
                .collect::<Vec<&str>>()
                .join("");

            let tmpl = env.template_from_str(&chat_template);
            let result = tmpl.unwrap().render(input).unwrap();
            assert_eq!(result, target);
        }
    }

    #[test]
    fn test_chat_template_invalid_with_guideline() {
        let ct = ChatTemplate::new(
            "{{- bos_token }}\n{%- if messages[-1].role == 'user' %}\n    {{- \"You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\\n\\n\" }}\n    {{- \"<start_of_turn>\\n\" }}\n    {{- \"Human Question: \" + messages[-1].content }}\n    {{- \"\\n<end_of_turn>\\n\\n\" }}\n    {{- \"Our safety principle is defined in the below:\\n\\n\" }}\n    {{- \"* \" + guideline + \"\\n\" }}\n    {{- \"\\n===\\n\\n\" }}\n    {{- \"Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\\n\\n\" }}\n{%- else %}\n    {{- \"You are a policy expert trying to help determine whether an AI response to a prompt is in violation of the defined safety policies.\\n\\n\" }}\n    {{- \"<start_of_turn>\\n\" }}\n    {{- \"Human Question: \" + messages[-2].content }}\n    {{- \"\\n<end_of_turn>\\n\" }}\n    {{- \"<start_of_turn>\\n\" }}\n    {{- \"Chatbot Response: \" + messages[-1].content }}\n    {{- \"\\n<end_of_turn>\\n\\n\" }}\n    {{- \"Our safety principle is defined in the below:\\n\\n\" }}\n    {{- \"* \" + guideline + \"\\n\" }}\n    {{- \"\\n===\\n\\n\" }}\n    {{- \"Does the Chatbot Response violate the above principle? Your answer must  start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.\\n\\n\" }}\n{%- endif %}\n\n".to_string(),
            Some(TokenizerConfigToken::String("<s>".to_string())),
            Some(TokenizerConfigToken::String("</s>".to_string())),
        );

        // convert TextMessage to Message
        let msgs: Vec<Message> = vec![
            Message {
                name: None,
                role: "user".to_string(),
                content: MessageContent::SingleText(
                    "I'd like to show off how chat templating works!".to_string(),
                ),
            },
            Message {
                name: None,
                role: "assistant".to_string(),
                content: MessageContent::SingleText(
                    "I'm doing great. How can I help you today?".to_string(),
                ),
            },
            Message {
                name: None,
                role: "user".to_string(),
                content: MessageContent::SingleText("Hello, how are you?".to_string()),
            },
        ];

        let result = ct.apply(None, msgs, None);

        match result {
            Ok(_) => panic!("Should have failed since no guideline is provided"),
            Err(e) => {
                assert_eq!(e.to_string(), "Missing template vatiable: guideline")
            }
        }
    }

    #[test]
    fn test_chat_template_with_default_tool_template() {
        let ct = ChatTemplate::new(
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}".to_string(),
            Some(TokenizerConfigToken::String("<s>".to_string())),
            Some(TokenizerConfigToken::String("</s>".to_string())),
        );

        // convert TextMessage to Message
        let msgs: Vec<Message> = vec![
            Message {
                name: None,
                role: "user".to_string(),
                content: MessageContent::SingleText(
                    "I'd like to show off how chat templating works!".to_string(),
                ),
            },
            Message {
                name: None,
                role: "assistant".to_string(),
                content: MessageContent::SingleText("Great! How can I help you today?".to_string()),
            },
            Message {
                name: None,
                role: "user".to_string(),
                content: MessageContent::SingleText("Just testing".to_string()),
            },
        ];
        let tools_string = r#"[{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather","parameters": {"type": "object","properties": {"location": {"type": "string","description": "The city and state, e.g. San Francisco, CA"},"format": {"type": "string","enum": ["celsius", "fahrenheit"],"description": "The temperature unit to use. Infer this from the users location."}},"required": ["location", "format"]}}}]"#.to_string();
        let tools: Vec<Tool> = serde_json::from_str(&tools_string).unwrap();
        let tool_prompt = "This default prompt will be used".to_string();
        let tools_and_prompt = Some((tools, tool_prompt));
        let result = ct.apply(None, msgs, tools_and_prompt);
        let expected = "<s>[INST] I'd like to show off how chat templating works! [/INST]Great! How can I help you today?</s> [INST] Just testing\n---\n[{\"type\":\"function\",\"function\":{\"description\":\"Get the current weather\",\"name\":\"get_current_weather\",\"arguments\":{\"properties\":{\"format\":{\"description\":\"The temperature unit to use. Infer this from the users location.\",\"enum\":[\"celsius\",\"fahrenheit\"],\"type\":\"string\"},\"location\":{\"description\":\"The city and state, e.g. San Francisco, CA\",\"type\":\"string\"}},\"required\":[\"location\",\"format\"],\"type\":\"object\"}}}]\nThis default prompt will be used [/INST]".to_string();
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_chat_template_with_custom_tool_template() {
        // chat template from meta-llama/Meta-Llama-3.1-8B-Instruct
        let ct = ChatTemplate::new(
            "{{- bos_token }}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n".to_string(),
            Some(TokenizerConfigToken::String("<s>".to_string())),
            Some(TokenizerConfigToken::String("</s>".to_string())),
        );
        let msgs: Vec<Message> = vec![
            Message {
                name: None,
                role: "system".to_string(),
                content: MessageContent::SingleText(
                    "Youre a helpful assistant! Answer the users question best you can."
                        .to_string(),
                ),
            },
            Message {
                name: None,
                role: "user".to_string(),
                content: MessageContent::SingleText(
                    "What is the weather like in Brooklyn, New York?".to_string(),
                ),
            },
        ];
        let tools_string = r#"[{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather","parameters": {"type": "object","properties": {"location": {"type": "string","description": "The city and state, e.g. San Francisco, CA"},"format": {"type": "string","enum": ["celsius", "fahrenheit"],"description": "The temperature unit to use. Infer this from the users location."}},"required": ["location", "format"]}}}]"#.to_string();
        let tools: Vec<Tool> = serde_json::from_str(&tools_string).unwrap();
        let tool_prompt = "This default prompt will be used".to_string();
        let tools_and_prompt = Some((tools, tool_prompt));
        let result = ct.apply(None, msgs, tools_and_prompt);
        let expected = "<s><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYoure a helpful assistant! Answer the users question best you can.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n{\n    \"function\": {\n        \"arguments\": {\n            \"properties\": {\n                \"format\": {\n                    \"description\": \"The temperature unit to use. Infer this from the users location.\",\n                    \"enum\": [\n                        \"celsius\",\n                        \"fahrenheit\"\n                    ],\n                    \"type\": \"string\"\n                },\n                \"location\": {\n                    \"description\": \"The city and state, e.g. San Francisco, CA\",\n                    \"type\": \"string\"\n                }\n            },\n            \"required\": [\n                \"location\",\n                \"format\"\n            ],\n            \"type\": \"object\"\n        },\n        \"description\": \"Get the current weather\",\n        \"name\": \"get_current_weather\"\n    },\n    \"type\": \"function\"\n}\n\nWhat is the weather like in Brooklyn, New York?\n---\nThis default prompt will be used<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_string();
        assert_eq!(result.unwrap(), expected);
    }
}
