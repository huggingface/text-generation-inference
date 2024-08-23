use crate::infer::InferError;
use crate::{
    FunctionDefinition, FunctionRef, FunctionsMap, JsonSchemaTool, Properties, Tool, ToolChoice,
    ToolType,
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
    ) -> Result<(Vec<Tool>, Option<JsonSchemaTool>), InferError> {
        // if no tools are provided, we return None
        if tools.is_empty() {
            return Ok((tools, None));
        }

        let tool_choice = tool_choice.0.unwrap_or(ToolType::OneOf);

        let mut tools = tools.clone();

        // add the notify_error function to the tools
        let notify_error = Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "notify_error".to_string(),
                description: Some("Notify an error or issue".to_string()),
                arguments: json!({
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "The error or issue to notify"
                        },
                        "_name": {
                            "type": "string",
                            "const": "notify_error"
                        }
                    },
                    "required": ["error", "_name"]
                }),
            },
        };
        tools.push(notify_error);

        // if tools are provided and no tool_choice we default to the OneOf
        let tools_to_use = match tool_choice {
            ToolType::FunctionName(name) => {
                vec![Self::find_tool_by_name(&tools, &name)?]
            }
            ToolType::Function { function } => {
                vec![Self::find_tool_by_name(&tools, &function.name)?]
            }
            ToolType::OneOf => tools.clone(),
            ToolType::NoTool => return Ok((tools, None)),
        };

        let functions: HashMap<String, serde_json::Value> = tools_to_use
            .iter()
            .map(|tool| {
                let func = tool.function.clone();

                // Clone the existing parameters, which are expected to be a JSON object
                let mut params = if let Value::Object(params) = &func.arguments {
                    params.clone()
                } else {
                    Map::new()
                };

                // Insert the function's description at the top level, outside of properties
                params.insert(
                    "description".to_string(),
                    Value::String(func.description.clone().unwrap_or_default()),
                );

                // Ensure 'properties' exists and is an object
                let properties = params
                    .entry("properties".to_string())
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
                    .unwrap();

                // Insert the constant for the function name inside 'properties'
                properties.insert(
                    "_name".to_string(),
                    json!({
                        "type": "string",
                        "const": func.name.clone(),
                        // "description": "The name of the function"
                    }),
                );

                // Check if 'required' exists, and it is an array. If not, create an empty array.
                let required = params
                    .entry("required".to_string())
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .unwrap();

                // Add 'name' to the 'required' array if it is not already present
                if !required.iter().any(|r| r == "_name") {
                    required.push(json!("_name"));
                }

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
                    .chain(std::iter::once(FunctionRef {
                        ref_path: "#/$functions/notify_error".to_string(),
                    }))
                    .collect(),
            },
        };

        Ok((tools, Some(tool_schema)))
    }
}
