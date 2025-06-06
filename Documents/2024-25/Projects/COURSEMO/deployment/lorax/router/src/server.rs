/// HTTP Server logic
use crate::adapter::{extract_adapter_params, BASE_MODEL_ADAPTER_ID};
use crate::config::Config;
use crate::health::Health;
use crate::infer::{InferError, InferResponse, InferStreamResponse};
use crate::tool_grammar::ToolGrammar;
use crate::validation::ValidationError;
use crate::{json, HubPreprocessorConfig, HubProcessorConfig, HubTokenizerConfig};
use crate::{
    AdapterParameters, AlternativeToken, BatchClassifyRequest, BatchEmbedRequest, BestOfSequence,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionStreamResponse, ChatCompletionStreamResponseChoice, ChatMessage, ClassifyRequest,
    CompatEmbedRequest, CompatEmbedResponse, CompatEmbedding, CompatGenerateRequest,
    CompletionFinishReason, CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, Details, EmbedParameters,
    EmbedRequest, EmbedResponse, Entity, ErrorResponse, FinishReason, GenerateParameters,
    GenerateRequest, GenerateResponse, HubModelInfo, Infer, Info, JsonSchema, LogProbs, Message,
    OpenAiResponseFormat, PrefillToken, ResponseFormat, ResponseFormatType,
    ReturnFunctionDefinition, SimpleToken, StreamDetails, StreamResponse, StringOrVec, Token,
    TokenizeRequest, TokenizeResponse, Tool, ToolCall, ToolChoice, UsageInfo, Validation,
};
use axum::extract::Extension;
use axum::extract::Query;
use axum::http::header;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::stream::StreamExt;
use futures::Stream;
use lorax_client::{ShardInfo, ShardedClient};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use once_cell::sync::OnceCell;
use reqwest_middleware::ClientBuilder;
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Mutex;
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use tower_http::cors::{
    AllowCredentials, AllowHeaders, AllowMethods, AllowOrigin, CorsLayer, ExposeHeaders,
};
use tracing::{info_span, instrument, Instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub static DEFAULT_ADAPTER_SOURCE: OnceCell<String> = OnceCell::new();

/// Generate tokens if `stream == false` or a stream of token if `stream == true`
#[utoipa::path(
post,
tag = "LoRAX",
path = "/",
request_body = CompatGenerateRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = GenerateResponse),
("text/event-stream" = StreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn compat_generate(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<CompatGenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut req = req.0;

    // default return_full_text given the pipeline_tag
    if req.parameters.return_full_text.is_none() {
        req.parameters.return_full_text = Some(default_return_full_text.0)
    }

    // switch on stream
    if req.stream {
        Ok(generate_stream(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(req.into()),
        )
        .await
        .into_response())
    } else {
        let (headers, generation) = generate(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(req.into()),
        )
        .await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(vec![generation.0])).into_response())
    }
}

const OPEN_AI_END_EVENT: &str = "[DONE]";

/// OpenAI compatible completions endpoint
#[utoipa::path(
post,
tag = "OpenAI Compatible",
path = "/v1/completions",
request_body = CompletionRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = CompletionResponse),
("text/event-stream" = CompletionStreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn completions_v1(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut req = req.0;
    if req.model == info.model_id.as_str() {
        // Allow user to specify the base model, but treat it as an empty adapter_id
        tracing::info!("Replacing base model {0} with empty adapter_id", req.model);
        req.model = "".to_string();
    }
    let mut gen_req = CompatGenerateRequest::from(req);

    // default return_full_text given the pipeline_tag
    if gen_req.parameters.return_full_text.is_none() {
        gen_req.parameters.return_full_text = Some(default_return_full_text.0)
    }

    // switch on stream
    if gen_req.stream {
        let callback = move |resp: StreamResponse| {
            Event::default()
                .json_data(CompletionStreamResponse::from(resp))
                .map_or_else(
                    |err| {
                        tracing::error!("Failed to serialize CompletionStreamResponse: {err}");
                        Event::default()
                    },
                    |data| data,
                )
        };

        let (headers, stream) = generate_stream_with_callback(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
            callback,
            Some(Event::default().data(OPEN_AI_END_EVENT)),
        )
        .await;
        Ok((headers, Sse::new(stream).keep_alive(KeepAlive::default())).into_response())
    } else {
        let (headers, generation) = generate(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
        )
        .await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(CompletionResponse::from(generation.0))).into_response())
    }
}

/// OpenAI compatible chat completions endpoint
#[utoipa::path(
post,
tag = "OpenAI Compatible",
path = "/v1/chat/completions",
request_body = ChatCompletionRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = ChatCompletionResponse),
("text/event-stream" = ChatCompletionStreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn chat_completions_v1(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut req = req.0;
    let model_id = info.model_id.clone();
    if req.model == info.model_id.as_str() {
        // Allow user to specify the base model, but treat it as an empty adapter_id
        tracing::info!("Replacing base model {0} with empty adapter_id", req.model);
        req.model = "".to_string();
    }

    let system_fingerprint = format!("{}-{}", info.version, info.docker_label.unwrap_or("native"));
    let (mut gen_req, using_tools): (CompatGenerateRequest, bool) =
        req.try_into_generate(&infer)?;

    // default return_full_text given the pipeline_tag
    if gen_req.parameters.return_full_text.is_none() {
        gen_req.parameters.return_full_text = Some(default_return_full_text.0)
    }

    // switch on stream
    if gen_req.stream {
        let callback = move |resp: StreamResponse| {
            Event::default()
                .json_data(ChatCompletionStreamResponse::from(resp))
                .map_or_else(
                    |err| {
                        tracing::error!("Failed to serialize ChatCompletionStreamResponse: {err}");
                        Event::default()
                    },
                    |data| data,
                )
        };

        let (headers, stream) = generate_stream_with_callback(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
            callback,
            Some(Event::default().data(OPEN_AI_END_EVENT)),
        )
        .await;
        Ok((headers, Sse::new(stream).keep_alive(KeepAlive::default())).into_response())
    } else {
        let (headers, generation) = generate(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
        )
        .await?;

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        let generation = &generation.0;
        let mut generations = vec![generation.generated_text.clone()];
        if let Some(best_of_sequences) = generation
            .details
            .as_ref()
            .unwrap()
            .best_of_sequences
            .as_ref()
        {
            generations.extend(
                best_of_sequences
                    .into_iter()
                    .map(|seq| seq.generated_text.clone()),
            );
        }

        let mut choice_content = vec![];
        for (_, gen) in generations.iter().enumerate() {
            let (tool_calls, output) = if using_tools {
                let gen_text_value: Value = serde_json::from_str(&gen).map_err(|e| {
                    InferError::ToolError(format!(
                        "Failed to parse generated text: {} {:?}",
                        e, gen
                    ))
                })?;
                let function = gen_text_value.get("function").ok_or(InferError::ToolError(
                    "No function found in generated text".to_string(),
                ))?;

                let name = function
                    .get("_name")
                    .and_then(Value::as_str)
                    .ok_or(InferError::ToolError(
                        "No _name found in generated text".to_string(),
                    ))?
                    .to_string();

                let mut arguments = function.clone();
                if let Value::Object(ref mut props) = arguments {
                    props.remove("_name");
                }
                match name.as_str() {
                    "no_tool" => {
                        // parse the content message
                        let content_message = arguments
                            .get("content")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                InferError::ToolError(
                                    "No `content` found in generated text".to_string(),
                                )
                            })?
                            .to_string();
                        (None, Some(content_message))
                    }
                    _ => {
                        let arguments = serde_json::to_string(&arguments).map_err(|e| {
                            InferError::ToolError(format!("Failed to serialize arguments: {}", e))
                        })?;
                        let tool_calls = vec![ToolCall {
                            id: "0".to_string(),
                            r#type: "function".to_string(),
                            function: ReturnFunctionDefinition {
                                description: None,
                                name,
                                arguments,
                            },
                        }];
                        (Some(tool_calls), None)
                    }
                }
            } else {
                (None, Some(gen.clone()))
            };
            choice_content.push((tool_calls, output));
        }

        // build the complete response object with the full text
        let response = ChatCompletionResponse::new(
            generation,
            model_id,
            system_fingerprint,
            choice_content,
            current_time,
        );

        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(response)).into_response())
    }
}

#[derive(Debug, Error)]
pub enum WebServerError {
    #[error("Axum error: {0}")]
    Axum(#[from] axum::BoxError),
}

type PreparedInput = (String, Option<ResponseFormat>, bool);

pub(crate) fn prepare_chat_input(
    infer: &Infer,
    response_format: Option<ResponseFormat>,
    tools: Option<Vec<Tool>>,
    tool_choice: ToolChoice,
    tool_prompt: &str,
    guideline: Option<String>,
    messages: Vec<Message>,
) -> Result<PreparedInput, InferError> {
    if response_format.is_some() && tools.is_some() {
        return Err(InferError::ToolError(
            "Response format and tools are mutually exclusive".into(),
        ));
    }

    // when response_format is set, tools are not included when applying the chat template to generate inputs
    if let Some(format) = response_format {
        let inputs = infer.apply_chat_template(guideline, messages, None)?;
        return Ok((inputs, Some(format), false));
    }

    // when no response_format is set and tools are included, apply the chat template with the tools
    // to generate inputs
    if let Some(tools) = tools {
        let (updated_tools, tool_schema) = ToolGrammar::apply(tools, tool_choice)?;

        let grammar = tool_schema.as_ref().map(|t| ResponseFormat {
            r#type: ResponseFormatType::JsonSchema,
            schema: Some(serde_json::json!(t)),
        });

        let inputs: String = infer.apply_chat_template(
            guideline,
            messages,
            Some((updated_tools, tool_prompt.into())),
        )?;
        return Ok((inputs, grammar, tool_schema.is_some()));
    }

    // if no response_format or tools are set simply apply the chat template to generate inputs
    let inputs = infer.apply_chat_template(guideline, messages, None)?;
    Ok((inputs, None, false))
}

/// LoRAX endpoint info
#[utoipa::path(
get,
tag = "LoRAX",
path = "/info",
responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
get,
tag = "LoRAX",
path = "/startup",
responses(
(status = 200, description = "Everything is working fine and ready"),
(status = 503, description = "LoRAX is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
)
)]
/// For k8s startup probe
async fn is_startup_ready(
    infer: Extension<Infer>,
    health: Extension<Health>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if health.shard_info().supports_classification {
        let classify_request = ClassifyRequest {
            inputs: "San Francisco".to_string(),
        };
        match infer.classify(classify_request).await {
            Ok(_) => {}
            Err(error) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: error.error_type().to_string(),
                    }),
                ));
            }
        }
    }
    if health.shard_info().supports_embeddings {
        let embed_request = EmbedRequest {
            inputs: "San Francisco".to_string(),
            parameters: Some(EmbedParameters {
                adapter_id: None,
                adapter_source: None,
                adapter_parameters: None,
                api_token: None,
            }),
        };
        match infer.embed(embed_request).await {
            Ok(_) => {}
            Err(error) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: error.error_type().to_string(),
                    }),
                ));
            }
        }
    }
    if health.shard_info().supports_generation {
        let generate_request = GenerateRequest {
            inputs: "Who?".to_string(),
            parameters: GenerateParameters {
                adapter_id: None,
                adapter_source: None,
                adapter_parameters: None,
                api_token: None,
                best_of: None,
                temperature: None,
                top_k: None,
                top_p: None,
                typical_p: None,
                do_sample: false,
                seed: None,
                repetition_penalty: None,
                frequency_penalty: None,
                presence_penalty: None,
                watermark: false,
                return_full_text: None,
                stop: vec![],
                truncate: None,
                details: false,
                decoder_input_details: false,
                return_k_alternatives: None,
                apply_chat_template: false,
                response_format: None,
                max_new_tokens: Some(1),
                ignore_eos_token: false,
            },
            add_special_tokens: true,
        };
        match infer.generate(generate_request).await {
            Ok(response) => {
                if response.generated_text.text.len() == 0 {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "Empty generation".to_string(),
                            error_type: "failed healthcheck".to_string(),
                        }),
                    ));
                }
            }
            Err(error) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: error.error_type().to_string(),
                    }),
                ));
            }
        }
    }
    Ok(())
}

#[utoipa::path(
get,
tag = "LoRAX",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 503, description = "LoRAX is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
)
)]
/// Health check method
async fn health(mut health: Extension<Health>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match health.check().await {
        true => Ok(()),
        false => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "unhealthy".to_string(),
                error_type: "healthcheck".to_string(),
            }),
        )),
    }
}

/// Generate tokens
#[utoipa::path(
post,
tag = "LoRAX",
path = "/generate",
request_body = GenerateRequest,
responses(
(status = 200, description = "Generated Text", body = GenerateResponse),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(
skip_all,
fields(
parameters = ? req.0.parameters,
total_time,
validation_time,
queue_time,
inference_time,
time_per_token,
seed,
)
)]
async fn generate(
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    mut req: Json<GenerateRequest>,
) -> Result<(HeaderMap, Json<GenerateResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");

    tracing::debug!("Input: {}", req.0.inputs);

    let compute_characters = req.0.inputs.chars().count();
    let mut add_prompt = None;
    if req.0.parameters.return_full_text.unwrap_or(false) {
        add_prompt = Some(req.0.inputs.clone());
    }

    let inputs = req.0.inputs.clone();

    let details = req.0.parameters.details || req.0.parameters.decoder_input_details;
    let (adapter_source, adapter_parameters) = extract_adapter_params(
        req.0.parameters.adapter_id.clone(),
        req.0.parameters.adapter_source.clone(),
        req.0.parameters.adapter_parameters.clone(),
    );

    if req.parameters.api_token.is_none() {
        // If no API token was explicitly provided in the request payload, try to set it from the request headers.
        let _ = req_headers.get("authorization").map_or((), |x| {
            x.to_str().map_or((), |y| {
                y.strip_prefix("Bearer ").map_or((), |token| {
                    req.parameters.api_token = Some(token.to_string());
                })
            })
        });
    }

    let api_token = req.parameters.api_token.clone();

    // Inference
    let (response, best_of_responses) = match req.0.parameters.best_of {
        Some(best_of) if best_of > 1 => {
            let (response, best_of_responses) = infer
                .generate_best_of(req.0, best_of, infer.prefix_caching)
                .await?;
            (response, Some(best_of_responses))
        }
        _ => (infer.generate(req.0).await?, None),
    };

    let generated_tokens = response.generated_text.generated_tokens;
    let skipped_tokens = response.generated_text.skipped_tokens;
    let prompt_tokens = response.prompt_tokens;
    let total_tokens = prompt_tokens + generated_tokens;

    // Token details
    let details = match details {
        true => {
            // convert best_of_responses
            let best_of_sequences = best_of_responses.map(|responses: Vec<InferResponse>| {
                responses
                    .into_iter()
                    .map(|response: InferResponse| {
                        // Add prompt if return_full_text
                        let mut output_text = response.generated_text.text;
                        if let Some(prompt) = &add_prompt {
                            output_text = prompt.clone() + &output_text;
                        }

                        BestOfSequence {
                            generated_text: output_text,
                            finish_reason: FinishReason::from(
                                response.generated_text.finish_reason,
                            ),
                            generated_tokens: response.generated_text.generated_tokens,
                            prefill: response.prefill,
                            tokens: response.tokens,
                            seed: response.generated_text.seed,
                        }
                    })
                    .collect()
            });

            Some(Details {
                finish_reason: FinishReason::from(response.generated_text.finish_reason),
                prompt_tokens: prompt_tokens,
                generated_tokens: generated_tokens,
                skipped_tokens: skipped_tokens,
                prefill: response.prefill,
                tokens: response.tokens,
                seed: response.generated_text.seed,
                best_of_sequences,
            })
        }
        false => None,
    };

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;
    let time_per_token = inference_time / response.generated_text.generated_tokens;
    let time_to_first_token = response.prefill_time - response.start;
    let time_per_output_token = (inference_time - time_to_first_token)
        / cmp::max(response.generated_text.generated_tokens - 1, 1);

    // Rust Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));
    span.record("time_per_token", format!("{time_per_token:?}"));
    span.record("time_to_first_token", format!("{time_to_first_token:?}"));
    span.record(
        "time_per_output_token",
        format!("{time_per_output_token:?}"),
    );
    span.record("seed", format!("{:?}", response.generated_text.seed));
    span.record("prompt_tokens", format!("{prompt_tokens:?}"));
    span.record("generated_tokens", format!("{generated_tokens:?}"));
    span.record("skipped_tokens", format!("{skipped_tokens:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-prompt-tokens",
        prompt_tokens.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-generated-tokens",
        generated_tokens.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-skipped-tokens",
        skipped_tokens.to_string().parse().unwrap(),
    );
    headers.insert("x-total-tokens", total_tokens.to_string().parse().unwrap());
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-per-token",
        time_per_token.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-to-first-token",
        time_to_first_token.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-per-output-token",
        time_per_output_token
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );

    headers.insert("x-model-id", info.model_id.parse().unwrap());

    let adapter_id_string = adapter_parameters
        .adapter_ids
        .iter()
        .map(|id| id.as_str())
        // filter out base model adapter id
        .filter(|id| *id != BASE_MODEL_ADAPTER_ID)
        .collect::<Vec<_>>()
        .join(",");

    if adapter_id_string.len() > 0 {
        headers.insert("x-adapter-id", adapter_id_string.parse().unwrap());
        headers.insert("x-adapter-source", adapter_source.unwrap().parse().unwrap());
    }

    // Metrics
    metrics::increment_counter!("lorax_request_success");
    metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_inference_duration",
        inference_time.as_secs_f64()
    );
    metrics::histogram!(
        "lorax_request_mean_time_per_token_duration",
        time_per_token.as_secs_f64()
    );
    metrics::histogram!(
        "lorax_request_generated_tokens",
        response.generated_text.generated_tokens as f64
    );

    if info.request_logger_url.is_some() {
        let _ = request_logger_sender
            .send((
                total_tokens as i64,
                adapter_id_string,
                inputs,
                response.generated_text.text.clone(),
                api_token.unwrap_or("".to_string()),
                info.model_id.clone(),
            ))
            .await;
    }

    // Send response
    let mut output_text = response.generated_text.text;
    if let Some(prompt) = add_prompt {
        output_text = prompt + &output_text;
    }

    tracing::debug!("Output: {}", output_text);
    tracing::info!("Success");

    let response = GenerateResponse {
        generated_text: output_text,
        details,
    };
    Ok((headers, Json(response)))
}

/// Generate a stream of token using Server-Sent Events
#[utoipa::path(
post,
tag = "LoRAX",
path = "/generate_stream",
request_body = GenerateRequest,
responses(
(status = 200, description = "Generated Text", body = StreamResponse,
content_type = "text/event-stream"),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"}),
content_type = "text/event-stream"),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"}),
content_type = "text/event-stream"),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"}),
content_type = "text/event-stream"),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"}),
content_type = "text/event-stream"),
)
)]
#[instrument(
skip_all,
fields(
parameters = ? req.0.parameters,
total_time,
validation_time,
queue_time,
inference_time,
time_per_token,
seed,
)
)]
async fn generate_stream(
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<GenerateRequest>,
) -> (
    HeaderMap,
    Sse<impl Stream<Item = Result<Event, Infallible>>>,
) {
    let callback = |resp: StreamResponse| Event::default().json_data(resp).unwrap();
    let (headers, stream) = generate_stream_with_callback(
        infer,
        info,
        request_logger_sender,
        req_headers,
        req,
        callback,
        None,
    )
    .await;
    (headers, Sse::new(stream).keep_alive(KeepAlive::default()))
}
async fn generate_stream_with_callback(
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    mut req: Json<GenerateRequest>,
    callback: impl Fn(StreamResponse) -> Event,
    end_event: Option<Event>,
) -> (HeaderMap, impl Stream<Item = Result<Event, Infallible>>) {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");

    tracing::debug!("Input: {}", req.0.inputs);

    let compute_characters = req.0.inputs.chars().count();

    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());

    if req.parameters.api_token.is_none() {
        // If no API token was explicitly provided in the request payload, try to set it from the request headers.
        let _ = req_headers.get("authorization").map_or((), |x| {
            x.to_str().map_or((), |y| {
                y.strip_prefix("Bearer ").map_or((), |token| {
                    req.parameters.api_token = Some(token.to_string());
                })
            })
        });
    }

    let api_token = req.parameters.api_token.clone();

    let (adapter_source, adapter_parameters) = extract_adapter_params(
        req.0.parameters.adapter_id.clone(),
        req.0.parameters.adapter_source.clone(),
        req.0.parameters.adapter_parameters.clone(),
    );

    let adapter_id_string = adapter_parameters
        .adapter_ids
        .iter()
        .map(|id| id.as_str())
        // filter out base model adapter id
        .filter(|id| *id != BASE_MODEL_ADAPTER_ID)
        .collect::<Vec<_>>()
        .join(",");

    if adapter_id_string.len() > 0 {
        headers.insert("x-adapter-id", adapter_id_string.parse().unwrap());
        headers.insert("x-adapter-source", adapter_source.unwrap().parse().unwrap());
    }

    headers.insert("x-model-id", info.model_id.parse().unwrap());

    let stream = async_stream::stream! {
            // Inference
            let mut end_reached = false;
            let mut error = false;

            let mut prefill_tokens_length = 0;

            let mut add_prompt = None;
            if req.0.parameters.return_full_text.unwrap_or(false) {
                add_prompt = Some(req.0.inputs.clone());
            }
            let inputs = req.0.inputs.clone();
            let details = req.0.parameters.details;

            let best_of = req.0.parameters.best_of.unwrap_or(1);
            if best_of != 1 {
                let err = InferError::from(ValidationError::BestOfStream);
                metrics::increment_counter!("lorax_request_failure", "err" => "validation");
                tracing::error!("{err}");
                yield Ok(Event::from(err));
            } else if req.0.parameters.decoder_input_details {
                let err = InferError::from(ValidationError::PrefillDetailsStream);
                metrics::increment_counter!("lorax_request_failure", "err" => "validation");
                tracing::error!("{err}");
                yield Ok(Event::from(err));
            } else {
                match infer.generate_stream(req.0).instrument(info_span!(parent: &span, "async_stream")).await {
                    // Keep permit as long as generate_stream lives
                    Ok((_permit, mut response_stream)) => {
                        // Server-Sent Event stream
                        while let Some(response) = response_stream.next().await {
                            match response {
                                Ok(response) => {
                                    match response {
                                        // Prefill is ignored
                                        InferStreamResponse::Prefill {
                                            tokens_length,
                                            ..
                                        } => {
                                            prefill_tokens_length = tokens_length;
                                        }
                                        // Yield event for every new token
                                        InferStreamResponse::Token(token) => {
                                            tracing::debug!(parent: &span, "Token: {:?}", token);

                                            // StreamResponse
                                            let stream_token = StreamResponse {
                                                token,
                                                generated_text: None,
                                                details: None,
                                            };

                                            yield Ok(callback(stream_token))
                                        }
                                        // Yield event for last token and compute timings
                                        InferStreamResponse::End {
                                            token,
                                            generated_text,
                                            start,
                                            queued,
                                        } => {
                                            // Token details
                                            let details = match details {
                                                true => Some(StreamDetails {
                                                    finish_reason: FinishReason::from(generated_text.finish_reason),
                                                    prompt_tokens: prefill_tokens_length,
                                                    generated_tokens: generated_text.generated_tokens,
                                                    seed: generated_text.seed,
                                                }),
                                                false => None,
                                            };

                                            // Timings
                                            let total_time = start_time.elapsed();
                                            let validation_time = queued - start_time;
                                            let queue_time = start - queued;
                                            let inference_time = Instant::now() - start;
                                            let time_per_token = inference_time / generated_text.generated_tokens;

                                            // Tracing metadata
                                            span.record("total_time", format!("{total_time:?}"));
                                            span.record("validation_time", format!("{validation_time:?}"));
                                            span.record("queue_time", format!("{queue_time:?}"));
                                            span.record("inference_time", format!("{inference_time:?}"));
                                            span.record("time_per_token", format!("{time_per_token:?}"));
                                            span.record("seed", format!("{:?}", generated_text.seed));
                                            span.record("prompt_tokens",  format!("{prefill_tokens_length:?}"));
                                            span.record("generated_tokens",  format!("{:?}", generated_text.generated_tokens));

                                            // Metrics
                                            metrics::increment_counter!("lorax_request_success");
                                            metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_validation_duration", validation_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_inference_duration", inference_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_mean_time_per_token_duration", time_per_token.as_secs_f64());
                                            metrics::histogram!("lorax_request_generated_tokens", generated_text.generated_tokens as f64);



                                            // StreamResponse
                                            end_reached = true;

                                            let mut output_text = generated_text.text;
                                            if let Some(prompt) = add_prompt {
                                                output_text = prompt + &output_text;
                                            }

                                            tracing::debug!(parent: &span, "Output: {}", output_text);
                                            tracing::info!(parent: &span, "Success");

                                            let total_tokens = generated_text.generated_tokens + prefill_tokens_length;
                                            if info.request_logger_url.is_some() {
                                                let _ = request_logger_sender.send((
                                                    total_tokens as i64,
                                                    adapter_id_string,
                                                    inputs,
                                                    output_text.clone(),
                                                    api_token.unwrap_or("".to_string()),
                                                    info.model_id.clone(),
                                                ))
                                                .await;
                                            }

                                            let stream_token = StreamResponse {
                                                token,
                                                generated_text: Some(output_text),
                                                details
                                            };

                                            yield Ok(callback(stream_token));
                                            if let Some(end_event) = end_event {
                                                yield Ok(end_event);
                                            }
                                            break;
                                        },
                                        InferStreamResponse::Embed {
                                            ..
                                        } => {
                                            let err = InferError::from(ValidationError::EmbeddingModel);
                                            metrics::increment_counter!("lorax_request_failure", "err" => "bad_request");
                                            tracing::error!("{err}");
                                            yield Ok(Event::from(err));
                                            break;
                                        }
                                        InferStreamResponse::Classify {
                                            ..
                                        } => {
                                            let err = InferError::from(ValidationError::ClassifyModelError);
                                            metrics::increment_counter!("lorax_request_failure", "err" => "bad_request");
                                            tracing::error!("{err}");
                                            yield Ok(Event::from(err));
                                            break;
                                        }
                                    }
                                }
                                // yield error
                                Err(err) => {
                                    error = true;
                                    yield Ok(Event::from(err));
                                    break;
                                }
                            }
                        }
                    },
                    // yield error
                    Err(err) => {
                        error = true;
                        yield Ok(Event::from(err));
                    }
                }
                // Check if generation reached the end
                // Skip if we already sent an error
                if !end_reached && !error {
                    let err = InferError::IncompleteGeneration;
                    metrics::increment_counter!("lorax_request_failure", "err" => "incomplete");
                    tracing::error!("{err}");
                    yield Ok(Event::from(err));
                }
            }
    };

    (headers, stream)
}

#[derive(Serialize, Deserialize, Debug)]
struct MetricFamily {
    r#type: String,
    data: Vec<DataPoint>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DataPoint {
    key: String,
    value: f64,
}

fn parse_text_to_metrics(text: &str) -> HashMap<String, MetricFamily> {
    let mut metrics = HashMap::new();
    let mut current_metric = String::new();

    for line in text.lines() {
        if line.is_empty() {
            continue;
        }

        if line.starts_with("# TYPE ") {
            // Extract metric name from TYPE declaration
            // # TYPE <metric_name> <metric_type>
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                current_metric = parts[2].to_string();
                metrics.insert(
                    current_metric.clone(),
                    MetricFamily {
                        r#type: parts[3].to_string(), // Metric type -> histogram, counter, etc
                        data: Vec::new(),
                    },
                );
                continue;
            }
        }

        // Parse metric line if it belongs to current metric family
        if let Some(metric_family) = metrics.get_mut(&current_metric) {
            let mut parts = line.split_whitespace();
            if let (Some(metric_name), Some(value_str)) = (parts.next(), parts.next()) {
                if let Ok(value) = value_str.parse::<f64>() {
                    let key = match metric_name {
                        name if name.contains('{') => name.to_string(),
                        name if name.ends_with("_sum") => "sum".to_string(),
                        name if name.ends_with("_count") => "count".to_string(),
                        _ => "".to_string(),
                    };

                    // Add the parsed metric data point
                    metric_family.data.push(DataPoint { key, value });
                }
            }
        }
    }

    metrics
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
get,
tag = "LoRAX",
path = "/metrics",
params(
    ("format", Query, description = "Optional format parameter (prometheus|json)", example = "json")
),
responses(
    (status = 200, description = "Prometheus or JSON Metrics",
     content(
         ("text/plain" = String),
         ("application/json" = Object)
     ))
)
)]
async fn metrics(
    prom_handle: Extension<PrometheusHandle>,
    format: Option<Query<String>>,
    headers: HeaderMap,
) -> Response {
    let want_json = format
        .map(|f| f.0.to_lowercase() == "json")
        .unwrap_or_else(|| {
            headers
                .get(header::ACCEPT)
                .and_then(|h| h.to_str().ok())
                .map(|h| h.contains("application/json"))
                .unwrap_or(false)
        });

    let prometheus_text = prom_handle.render();

    if want_json {
        let metrics = parse_text_to_metrics(&prometheus_text);

        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            Json(metrics),
        )
            .into_response()
    } else {
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain")],
            prometheus_text,
        )
            .into_response()
    }
}

async fn request_logger(
    request_logger_url: Option<String>,
    mut rx: mpsc::Receiver<(i64, String, String, String, String, String)>,
) {
    if request_logger_url.is_none() {
        tracing::info!("REQUEST_LOGGER_URL not set, request logging is disabled");
        return;
    }

    let url_string = request_logger_url.unwrap();
    tracing::info!("Request logging enabled, sending logs to {url_string}");

    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
    let client = ClientBuilder::new(reqwest::Client::new())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build();
    while let Some((tokens, adapter_id, input, output, api_token, model_id)) = rx.recv().await {
        // Make a request out to localhost:8899 with the tokens, api_token, and model_id
        let res = client
            .post(&url_string)
            .json(&json!({
                "tokens": tokens,
                "adapter_id": adapter_id,
                "input": input,
                "output": output,
                "api_token": api_token,
                "model_id": model_id
            }))
            .send()
            .await;

        if let Err(e) = res {
            tracing::error!("Failed to log request: {e}");
        }
    }
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_info: HubModelInfo,
    shard_info: ShardInfo,
    compat_return_full_text: bool,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    max_active_adapters: usize,
    adapter_cycle_time_s: u64,
    client: ShardedClient,
    config: Option<Config>,
    tokenizer: Option<Tokenizer>,
    (preprocessor_config, _processor_config): (Option<HubPreprocessorConfig>, HubProcessorConfig),
    validation_workers: usize,
    addr: SocketAddr,
    cors_allow_origin: Option<AllowOrigin>, // exact match
    cors_allow_methods: Option<AllowMethods>,
    cors_allow_credentials: Option<AllowCredentials>,
    cors_allow_headers: Option<AllowHeaders>,
    cors_expose_headers: Option<ExposeHeaders>,
    tokenizer_config: HubTokenizerConfig,
    ngrok: bool,
    _ngrok_authtoken: Option<String>,
    _ngrok_edge: Option<String>,
    adapter_source: String,
    eager_prefill: bool,
    prefix_caching: bool,
) -> Result<(), axum::BoxError> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    health,
    get_model_info,
    compat_generate,
    generate,
    generate_stream,
    completions_v1,
    chat_completions_v1,
    tokenize,
    metrics,
    ),
    components(
    schemas(
    Info,
    UsageInfo,
    ResponseFormat,
    ResponseFormatType,
    OpenAiResponseFormat,
    JsonSchema,
    CompatGenerateRequest,
    GenerateRequest,
    GenerateParameters,
    AdapterParameters,
    AlternativeToken,
    PrefillToken,
    Token,
    SimpleToken,
    TokenizeRequest,
    TokenizeResponse,
    GenerateResponse,
    BestOfSequence,
    Details,
    FinishReason,
    StreamResponse,
    StreamDetails,
    ErrorResponse,
    ChatMessage,
    LogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    CompletionResponseStreamChoice,
    CompletionFinishReason,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    )
    ),
    tags(
    (name = "LoRAX", description = "LoRAX API"),
    (name = "OpenAI Compatible", description = "OpenAI compatible API"),
    (name = "Tokenization", description = "Tokenizer API"),
    ),
    info(
    title = "LoRAX",
    license(
    name = "Apache 2.0",
    url = "https://www.apache.org/licenses/LICENSE-2.0"
    )
    )
    )]
    struct ApiDoc;

    let cloned_tokenizer = tokenizer.clone().map(|t| Arc::new(Mutex::new(t)));
    let arc_tokenizer = tokenizer.clone().map(Arc::new);

    // Create state
    let validation = Validation::new(
        validation_workers,
        tokenizer,
        config,
        preprocessor_config,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
    );
    let inference_health = Arc::new(AtomicBool::new(false));
    let health_ext = Health::new(client.clone(), inference_health.clone(), shard_info.clone());

    // For non-causal LMs, the max batch total tokens is equal to the max batch prefill tokens
    let is_causal_lm = shard_info.supports_generation;
    let effective_max_batch_total_tokens = if is_causal_lm {
        max_batch_total_tokens
    } else {
        max_batch_prefill_tokens
    };

    let infer = Infer::new(
        client.clone(),
        validation,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        effective_max_batch_total_tokens,
        max_waiting_tokens,
        max_concurrent_requests,
        max_active_adapters,
        adapter_cycle_time_s,
        shard_info.requires_padding,
        shard_info.window_size,
        inference_health,
        eager_prefill,
        tokenizer_config,
        arc_tokenizer,
        shard_info.block_size,
        shard_info.speculate,
        shard_info.preloaded_adapters,
        prefix_caching,
        shard_info.chunked_prefill,
        shard_info.requires_block_allocator,
    );

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("lorax_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Generated tokens buckets
    let generated_tokens_matcher = Matcher::Full(String::from("lorax_request_generated_tokens"));
    let generated_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Input Length buckets
    let max_new_tokens_matcher = Matcher::Full(String::from("lorax_request_max_new_tokens"));
    let max_new_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("lorax_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..1024).map(|x| (x + 1) as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(generated_tokens_matcher, &generated_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(max_new_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // CORS layer
    let cors_allow_origin = cors_allow_origin.unwrap_or(AllowOrigin::any());
    // unwrap allow methods with default get and post
    let cors_allow_methods =
        cors_allow_methods.unwrap_or(AllowMethods::list(vec![Method::GET, Method::POST]));

    let cors_allow_headers =
        cors_allow_headers.unwrap_or(AllowHeaders::list(vec![http::header::CONTENT_TYPE]));

    let cors_expose_headers = cors_expose_headers.unwrap_or(ExposeHeaders::default());
    let cors_allow_credentials = cors_allow_credentials.unwrap_or(AllowCredentials::default());

    // log cors stuff
    tracing::info!(
        "CORS: origin: {cors_allow_origin:?}, methods: {cors_allow_methods:?}, headers: {cors_allow_headers:?}, expose-headers: {cors_expose_headers:?} credentials: {cors_allow_credentials:?}",
    );

    let cors_layer = CorsLayer::new()
        .allow_methods(cors_allow_methods)
        .allow_headers(cors_allow_headers)
        .allow_credentials(cors_allow_credentials)
        .expose_headers(cors_expose_headers)
        .allow_origin(cors_allow_origin);

    // log all the cors layer
    tracing::info!("CORS: {cors_layer:?}");

    // Endpoint info
    let info = Info {
        model_id: model_info.model_id,
        model_sha: model_info.sha,
        model_dtype: shard_info.dtype,
        model_device_type: shard_info.device_type,
        model_pipeline_tag: model_info.pipeline_tag,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_total_tokens,
        max_waiting_tokens,
        validation_workers,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
        request_logger_url: std::env::var("REQUEST_LOGGER_URL").ok(),
        eager_prefill,
    };

    DEFAULT_ADAPTER_SOURCE
        .set(adapter_source.clone())
        .unwrap_or_else(|_| {
            panic!("DEFAULT_ADAPTER_SOURCE was already set!");
        });

    // Kick off thread here that writes to the log file
    let (tx, rx) = mpsc::channel(32);
    let request_logger_sender = Arc::new(tx);
    if info.request_logger_url.is_some() {
        tokio::spawn(request_logger(info.request_logger_url.clone(), rx));
    } else {
        tracing::info!("REQUEST_LOGGER_URL not set, request logging is disabled");
    }

    #[allow(unused_mut)] // mut is needed for conditional compilation
    let mut doc = ApiDoc::openapi();

    // Configure Swagger UI
    let swagger_ui = SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc);

    // Create router
    let base_routes = Router::new()
        // Base routes
        .route("/", post(compat_generate))
        .route("/generate", post(generate))
        .route("/embed", post(embed))
        .route("/classify", post(classify))
        .route("/classify_batch", post(classify_batch))
        .route("/generate_stream", post(generate_stream))
        .route("/v1/completions", post(completions_v1))
        .route("/v1/embeddings", post(compat_embed))
        .route("/v1/chat/completions", post(chat_completions_v1))
        // AWS Sagemaker route
        .route("/invocations", post(compat_generate));

    let info_routes = Router::new()
        .route("/", get(health))
        // Base Health route
        .route("/startup", get(is_startup_ready))
        .route("/health", get(health))
        .route("/info", get(get_model_info))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics))
        .route("/tokenize", post(tokenize));

    // Combine routes and layers
    let mut app = Router::new()
        .merge(swagger_ui)
        .merge(base_routes)
        .merge(info_routes);

    // add layers after routes
    app = app
        .layer(Extension(info))
        .layer(Extension(client.clone()))
        .layer(Extension(request_logger_sender.clone()))
        .layer(Extension(health_ext.clone()))
        .layer(Extension(compat_return_full_text))
        .layer(Extension(infer))
        .layer(Extension(prom_handle.clone()))
        .layer(OtelAxumLayer::default())
        .layer(cors_layer)
        .layer(Extension(cloned_tokenizer));

    if ngrok {
        #[cfg(feature = "ngrok")]
        {
            panic!("ngrok feature is not functional with axum=0.7 and hyper=1, waiting on https://github.com/ngrok/ngrok-rust/pull/137/files to re-enable.");

            // Run server
        }
        #[cfg(not(feature = "ngrok"))]
        {
            let _ngrok_authtoken = ngrok_authtoken;
            let _ngrok_domain = ngrok_domain;
            let _ngrok_username = ngrok_username;
            let _ngrok_password = ngrok_password;

            panic!("`lorax-router` was compiled without the `ngrok` feature");
        }
    } else {
        // Run server

        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|err| WebServerError::Axum(Box::new(err)))?;
    }
    Ok(())
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
    opentelemetry::global::shutdown_tracer_provider();
}

impl From<i32> for FinishReason {
    fn from(finish_reason: i32) -> Self {
        let finish_reason = lorax_client::FinishReason::from_i32(finish_reason).unwrap();
        match finish_reason {
            lorax_client::FinishReason::Length => FinishReason::Length,
            lorax_client::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            lorax_client::FinishReason::StopSequence => FinishReason::StopSequence,
        }
    }
}

/// Convert to Axum supported formats
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        let status_code = match err {
            InferError::GenerationError(_) => StatusCode::FAILED_DEPENDENCY,
            InferError::Overloaded(_) => StatusCode::TOO_MANY_REQUESTS,
            InferError::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::IncompleteGeneration => StatusCode::INTERNAL_SERVER_ERROR,
            InferError::TemplateError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::EmbeddingFailure => StatusCode::INTERNAL_SERVER_ERROR,
            InferError::ClassificationFailure => StatusCode::INTERNAL_SERVER_ERROR,
            InferError::ToolError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::MissingTemplateVariable(_) => StatusCode::UNPROCESSABLE_ENTITY,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            })
            .unwrap()
    }
}

/// Embed inputs
#[utoipa::path(
    post,
    tag = "Embedding",
    path = "/embed",
    request_body = EmbedRequest,
    responses(
    (status = 200, description = "Embeddings ids", body = EmbedResponse),
    (status = 500, description = "Incomplete embedding", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn embed(
    infer: Extension<Infer>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Input: {}", req.inputs);
    // Inference
    let response = infer.embed(req).await?;
    Ok(Json(response))
}

/// Embed inputs
#[utoipa::path(
    post,
    tag = "OpenAI Compatible",
    path = "/v1/embeddings",
    request_body = CompatEmbedRequest,
    responses(
    (status = 200, description = "Embeddings ids", body = CompatEmbedResponse),
    (status = 500, description = "Incomplete embedding", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
#[axum::debug_handler]
async fn compat_embed(
    infer: Extension<Infer>,
    Json(req): Json<CompatEmbedRequest>,
) -> Result<Json<CompatEmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Input: {}", req.input);
    if let StringOrVec::Vec(inputs) = req.input {
        let batch_embed_req = BatchEmbedRequest {
            inputs,
            parameters: req.parameters,
        };
        let response = infer.embed_batch(batch_embed_req).await?;
        let compat_embeddings = response
            .into_iter()
            .enumerate()
            .map(|(i, e)| -> CompatEmbedding {
                CompatEmbedding {
                    index: i as i32,
                    embedding: e.embeddings,
                    object: "embedding".to_string(),
                }
            })
            .collect();
        Ok(Json(CompatEmbedResponse {
            embeddings: compat_embeddings,
        }))
    } else if let StringOrVec::String(input) = req.input {
        let embed_req = EmbedRequest {
            inputs: input.to_string(),
            parameters: req.parameters,
        };
        let response = infer.embed(embed_req).await?;
        Ok(Json(CompatEmbedResponse {
            embeddings: vec![CompatEmbedding {
                index: 0,
                embedding: response.embeddings,
                object: "embedding".to_string(),
            }],
        }))
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid input".to_string(),
                error_type: "invalid_input".to_string(),
            }),
        ))
    }
}

#[utoipa::path(
    post,
    tag = "Classify",
    path = "/classify",
    request_body = TokenizeRequest,
    responses(
    (status = 200, description = "Classifications", body = ClassifyResponse),
    (status = 500, description = "Incomplete classification", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn classify(
    infer: Extension<Infer>,
    Json(req): Json<ClassifyRequest>,
) -> Result<(HeaderMap, Json<Vec<Entity>>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Input: {}", req.inputs);
    let response = infer.classify(req).await?;

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;

    // Rust Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("lorax_request_success");
    metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_inference_duration",
        inference_time.as_secs_f64()
    );
    metrics::histogram!(
        "lorax_request_classify_output_count",
        response.predictions.len() as f64
    );

    tracing::debug!("Output: {:?}", response.predictions);
    tracing::info!("Success");

    Ok((headers, Json(response.predictions)))
}

#[utoipa::path(
    post,
    tag = "ClassifyBatch",
    path = "/classify_batch",
    request_body = BatchClassifyRequest,
    responses(
    (status = 200, description = "Classifications", body = BatchClassifyResponse),
    (status = 500, description = "Incomplete classification", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn classify_batch(
    infer: Extension<Infer>,
    Json(req): Json<BatchClassifyRequest>,
) -> Result<(HeaderMap, Json<Vec<Vec<Entity>>>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Inputs: {:?}", req.inputs);
    let num_inputs = req.inputs.len();
    let responses = infer.classify_batch(req).await?;

    // Timings
    let now = Instant::now();
    let total_time = start_time.elapsed();
    let mut validation_times = Vec::with_capacity(responses.len());
    let mut queue_times = Vec::with_capacity(responses.len());
    let mut inference_times = Vec::with_capacity(responses.len());

    for r in &responses {
        validation_times.push(r.queued - r.start);
        queue_times.push(r.start - r.queued);
        inference_times.push(now - r.start);
    }

    let validation_time = validation_times.iter().sum::<Duration>() / responses.len() as u32;
    let queue_time = queue_times.iter().sum::<Duration>() / responses.len() as u32;
    let inference_time = inference_times.iter().sum::<Duration>() / responses.len() as u32;

    // Rust Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("num_inputs", num_inputs);
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("lorax_request_success");
    metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
    metrics::histogram!("lorax_request_input_count", num_inputs as f64);
    metrics::histogram!(
        "lorax_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_inference_duration",
        inference_time.as_secs_f64()
    );

    let batch_entity_vec: Vec<Vec<Entity>> = responses
        .into_iter()
        .map(|r| {
            let entity_vec = r.predictions;
            metrics::histogram!(
                "lorax_request_classify_output_count",
                entity_vec.len() as f64
            );
            entity_vec
        })
        .collect();
    tracing::debug!("Output: {:?}", batch_entity_vec);
    tracing::info!("Success");
    Ok((headers, Json(batch_entity_vec)))
}

/// Tokenize inputs
#[utoipa::path(
    post,
    tag = "Tokenization",
    path = "/tokenize",
    request_body = TokenizeRequest,
    responses(
    (status = 200, description = "Tokenized ids", body = TokenizeResponse),
    (status = 404, description = "No tokenizer found", body = ErrorResponse,
    example = json ! ({"error": "No fast tokenizer available"})),
    )
    )]
#[instrument(skip_all)]
async fn tokenize(
    Extension(cloned_tokenizer): Extension<Option<Arc<Mutex<Tokenizer>>>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(tokenizer) = cloned_tokenizer {
        let input = req.inputs.clone();
        let add_special_tokens = match req.add_special_tokens {
            None => true,
            _ => req.add_special_tokens.unwrap(),
        };
        let tokenizer = tokenizer.lock().unwrap();
        let char_offset = tokenizer
            .encode_char_offsets(&input[..], add_special_tokens)
            .unwrap();
        let tokens: Vec<SimpleToken> = char_offset
            .get_ids()
            .iter()
            .zip(char_offset.get_offsets().iter())
            .map(|(&id, &(start, stop))| {
                let text: String = tokenizer.id_to_token(id).unwrap();
                SimpleToken {
                    id,
                    text,
                    start,
                    stop,
                }
            })
            .collect();
        Ok(Json(TokenizeResponse(tokens)))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "No fast tokenizer or tokenizer.json for this model".to_string(),
                error_type: "no fast tokenizer".to_string(),
            }),
        ))
    }
}
