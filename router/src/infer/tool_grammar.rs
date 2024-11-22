use crate::infer::InferError;
use crate::{
    FunctionDefinition, FunctionRef, FunctionsMap, JsonSchemaTool, Properties, Tool, ToolChoice,
};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub(crate) struct ToolGrammar {}

impl ToolGrammar {
    // find a tool by name
    fn find_tool_by_name(tools: &[Tool], name: &str) -> Result<Tool, InferError> {
        tools
            .iter()
            .find(|tool| tool.function.name == name)
            .cloned()
            .ok_or_else(|| InferError::ToolError(format!("Tool with name {} not found", name)))
    }

    pub fn apply(
        tools: Vec<Tool>,
        tool_choice: ToolChoice,
    ) -> Result<Option<(Vec<Tool>, JsonSchemaTool)>, InferError> {
        let tools_to_use = match tool_choice {
            ToolChoice::Function(function) => {
                vec![Self::find_tool_by_name(&tools, &function.name)?]
            }
            ToolChoice::Required => tools,
            ToolChoice::Auto => {
                // only add the no_tool function if the user has selected the auto option
                tools
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Tool {
                        r#type: "function".to_string(),
                        function: FunctionDefinition {
                            name: "no_tool".to_string(),
                            description: Some(
                                "Open ended response with no specific tool selected".to_string(),
                            ),
                            arguments: json!({
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "The response content",
                                    }
                                },
                                "required": ["content"]
                            }),
                        },
                    }))
                    .collect::<Vec<_>>()
            }
            ToolChoice::NoTool => vec![],
        };

        // if no tools are provided or if the user has selected the no_tool option, return None
        if tools_to_use.is_empty() {
            return Ok(None);
        }

        let functions: HashMap<String, serde_json::Value> = tools_to_use
            .iter()
            .map(|tool| {
                let func = tool.function.clone();

                let mut params = Map::new();

                params.insert(
                    "description".to_string(),
                    Value::String(func.description.unwrap_or_default()),
                );

                let mut properties = Map::new();
                let mut required = vec![Value::String("_name".to_string())];

                properties.insert(
                    "_name".to_string(),
                    json!({
                        "type": "string",
                        "const": func.name.clone(),
                    }),
                );

                if let Value::Object(args) = func.arguments {
                    if let Some(Value::Object(props)) = args.get("properties") {
                        properties.extend(props.clone());
                    }
                    if let Some(Value::Array(reqs)) = args.get("required") {
                        required.extend(reqs.clone());
                    }
                    params.insert(
                        "additionalProperties".to_string(),
                        Value::Bool(
                            args.get("additionalProperties").and_then(|v| v.as_str())
                                == Some("true"),
                        ),
                    );
                }

                params.insert("properties".to_string(), Value::Object(properties));
                params.insert("required".to_string(), Value::Array(required));

                (func.name, Value::Object(params))
            })
            .collect();

        let tool_schema = JsonSchemaTool {
            functions_map: FunctionsMap { functions },
            properties: Properties {
                function: tools_to_use
                    .iter()
                    .map(|tool| FunctionRef {
                        ref_path: format!("#/$functions/{}", tool.function.name.clone()),
                    })
                    .collect(),
            },
        };

        Ok(Some((tools_to_use, tool_schema)))
    }
}
