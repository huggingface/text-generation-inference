use crate::infer::InferError;
use crate::{ChatTemplateInputs, GrammarType, Message, MessageChunk, Text, TextMessage};
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;

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
}

impl ChatTemplate {
    pub(crate) fn new(
        template: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Self {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        let template_str = template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);

        // check if contains the tools variable within the template
        let use_default_tool_template =
            !template_str.as_ref().replace(' ', "").contains("{{tools}}");
        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        Self {
            template,
            bos_token,
            eos_token,
            use_default_tool_template,
        }
    }

    pub(crate) fn apply(
        &self,
        mut messages: Vec<Message>,
        grammar_with_prompt: Option<(GrammarType, String)>,
    ) -> Result<String, InferError> {
        if self.use_default_tool_template {
            if let Some(last_message) = messages.last_mut() {
                if let Some((GrammarType::Json(tools), tool_prompt)) = grammar_with_prompt {
                    last_message.content.push(MessageChunk::Text(Text {
                        text: format!("\n---\n{}\n{}", tool_prompt, tools),
                    }));
                }
            }
        }

        let messages: Vec<TextMessage> = messages.into_iter().map(|c| c.into()).collect();

        self.template
            .render(ChatTemplateInputs {
                messages,
                bos_token: self.bos_token.as_deref(),
                eos_token: self.eos_token.as_deref(),
                add_generation_prompt: true,
                tools: None,
                tools_prompt: None,
            })
            .map_err(InferError::TemplateError)
    }
}
