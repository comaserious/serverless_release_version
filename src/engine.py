import os
import logging
import json
import asyncio

from dotenv import load_dotenv
from typing import AsyncGenerator, Optional
import time

from vllm import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_models import BaseModelPath, LoRAModulePath, OpenAIServingModels


from utils import DummyRequest, JobInput, BatchSize, create_error_response
from constants import DEFAULT_MAX_CONCURRENCY, DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE_GROWTH_FACTOR, DEFAULT_MIN_BATCH_SIZE
from tokenizer import TokenizerWrapper
from engine_args import get_engine_args

class vLLMEngine:
    def __init__(self, engine = None):
        load_dotenv() # For local development
        self.engine_args = get_engine_args()
        logging.info(f"Engine args: {self.engine_args}")
        
        # Initialize vLLM engine first
        self.llm = self._initialize_llm() if engine is None else engine.llm
        
        # Only create custom tokenizer wrapper if not using mistral tokenizer mode
        # For mistral models, let vLLM handle tokenizer initialization
        if self.engine_args.tokenizer_mode != 'mistral':
            self.tokenizer = TokenizerWrapper(self.engine_args.tokenizer or self.engine_args.model, 
                                              self.engine_args.tokenizer_revision, 
                                              self.engine_args.trust_remote_code)
        else:
            # For mistral models, we'll get the tokenizer from vLLM later
            self.tokenizer = None
            
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
        self.default_batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        self.batch_size_growth_factor = int(os.getenv("BATCH_SIZE_GROWTH_FACTOR", DEFAULT_BATCH_SIZE_GROWTH_FACTOR))
        self.min_batch_size = int(os.getenv("MIN_BATCH_SIZE", DEFAULT_MIN_BATCH_SIZE))

    def _get_tokenizer_for_chat_template(self):
        """Get tokenizer for chat template application"""
        if self.tokenizer is not None:
            return self.tokenizer
        else:
            # For mistral models, get tokenizer from vLLM engine
            # This is a fallback - ideally chat templates should be handled by vLLM directly
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.engine_args.tokenizer or self.engine_args.model,
                    revision=self.engine_args.tokenizer_revision or "main",
                    trust_remote_code=self.engine_args.trust_remote_code
                )
                # Create a minimal wrapper
                class MinimalTokenizerWrapper:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
                        self.has_chat_template = bool(self.tokenizer.chat_template) or bool(self.custom_chat_template)
                        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
                            self.tokenizer.chat_template = self.custom_chat_template
                    
                    def apply_chat_template(self, input):
                        if isinstance(input, list):
                            if not self.has_chat_template:
                                raise ValueError(
                                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                                )
                        elif isinstance(input, str):
                            input = [{"role": "user", "content": input}]
                        else:
                            raise ValueError("Input must be a string or a list of messages")
                        
                        return self.tokenizer.apply_chat_template(
                            input, tokenize=False, add_generation_prompt=True
                        )
                
                return MinimalTokenizerWrapper(tokenizer)
            except Exception as e:
                logging.error(f"Failed to create fallback tokenizer: {e}")
                raise e

    def dynamic_batch_size(self, current_batch_size, batch_size_growth_factor):
        return min(current_batch_size*batch_size_growth_factor, self.default_batch_size)
                           
    async def generate(self, job_input: JobInput):
        try:
            async for batch in self._generate_vllm(
                llm_input=job_input.llm_input,
                validated_sampling_params=job_input.sampling_params,
                batch_size=job_input.max_batch_size,
                stream=job_input.stream,
                apply_chat_template=job_input.apply_chat_template,
                request_id=job_input.request_id,
                batch_size_growth_factor=job_input.batch_size_growth_factor,
                min_batch_size=job_input.min_batch_size
            ):
                yield batch
        except Exception as e:
            yield {"error": create_error_response(str(e)).model_dump()}

    async def _generate_vllm(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id, batch_size_growth_factor, min_batch_size: str) -> AsyncGenerator[dict, None]:
        if apply_chat_template or isinstance(llm_input, list):
            tokenizer_wrapper = self._get_tokenizer_for_chat_template()
            llm_input = tokenizer_wrapper.apply_chat_template(llm_input)
        
        results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
        n_responses = validated_sampling_params.n
        n_input_tokens = 0
        is_first_output = True
        last_output_texts = ["" for _ in range(n_responses)]
        token_counters = {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }
        
        max_batch_size = batch_size or self.default_batch_size
        batch_size_growth_factor = batch_size_growth_factor or self.batch_size_growth_factor
        min_batch_size = min_batch_size or self.min_batch_size
        batch_size_obj = BatchSize(max_batch_size, min_batch_size, batch_size_growth_factor)

        try:
            async for request_output in results_generator:
                # Count input tokens only once
                if is_first_output:
                    n_input_tokens = len(request_output.prompt_token_ids)
                    is_first_output = False

                for output in request_output.outputs:
                    output_index = output.index
                    
                    # Update last output text first (항상 업데이트)
                    current_text = output.text
                    
                    if stream:
                        # Calculate new output (delta from last output)
                        new_output = current_text[len(last_output_texts[output_index]):]
                        
                        # Only process non-empty outputs
                        if new_output:
                            batch["choices"][output_index]["tokens"].append(new_output)
                            token_counters["batch"] += 1
                            token_counters["total"] += 1

                            # Yield when batch size is reached
                            if token_counters["batch"] >= batch_size_obj.current_batch_size:
                                batch["usage"] = {
                                    "input": n_input_tokens,
                                    "output": token_counters["total"],
                                }
                                yield batch
                                batch = {
                                    "choices": [{"tokens": []} for _ in range(n_responses)],
                                }
                                token_counters["batch"] = 0
                                batch_size_obj.update()
                    else:
                        # Non-streaming: just count tokens
                        token_counters["total"] += 1
                    
                    # Update last output text after processing
                    last_output_texts[output_index] = current_text

        except Exception as e:
            logging.error(f"Error during vLLM generation: {e}")
            # Yield any remaining batch before raising error
            if token_counters["batch"] > 0:
                batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
                yield batch
            raise

        # Handle non-streaming final output
        if not stream:
            for output_index, output_text in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output_text]
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch
        # Handle streaming final batch (if any remaining)
        elif token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch

    def _initialize_llm(self):
        try:
            start = time.time()
            engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            end = time.time()
            logging.info(f"Initialized vLLM engine in {end - start:.2f}s")
            return engine
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e


class OpenAIvLLMEngine(vLLMEngine):
    def __init__(self, vllm_engine):
        super().__init__(vllm_engine)
        self.served_model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or self.engine_args.model
        self.response_role = os.getenv("OPENAI_RESPONSE_ROLE") or "assistant"
        self.lora_adapters = self._load_lora_adapters()
        asyncio.run(self._initialize_engines())
        # Handle both integer and boolean string values for RAW_OPENAI_OUTPUT
        raw_output_env = os.getenv("RAW_OPENAI_OUTPUT", "1")
        if raw_output_env.lower() in ('true', 'false'):
            self.raw_openai_output = raw_output_env.lower() == 'true'
        else:
            self.raw_openai_output = bool(int(raw_output_env))

    def _load_lora_adapters(self):
        adapters = []
        try:
            adapters = json.loads(os.getenv("LORA_MODULES", '[]'))
        except Exception as e:
            logging.info(f"---Initialized adapter json load error: {e}")

        for i, adapter in enumerate(adapters):
            try:
                adapters[i] = LoRAModulePath(**adapter)
                logging.info(f"---Initialized adapter: {adapter}")
            except Exception as e:
                logging.info(f"---Initialized adapter not worked: {e}")
                continue
        return adapters

    async def _initialize_engines(self):
        self.model_config = await self.llm.get_model_config()
        self.base_model_paths = [
            BaseModelPath(name=self.engine_args.model, model_path=self.engine_args.model)
        ]

        self.serving_models = OpenAIServingModels(
            engine_client=self.llm,
            model_config=self.model_config,
            base_model_paths=self.base_model_paths,
            lora_modules=self.lora_adapters,
        )
        await self.serving_models.init_static_loras()
        
        # Get chat template from vLLM tokenizer if available
        chat_template = None
        if self.tokenizer and hasattr(self.tokenizer, 'tokenizer'):
            chat_template = self.tokenizer.tokenizer.chat_template
        
        self.chat_engine = OpenAIServingChat(
            engine_client=self.llm, 
            model_config=self.model_config,
            models=self.serving_models,
            response_role=self.response_role,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            # enable_reasoning=os.getenv('ENABLE_REASONING', 'false').lower() == 'true',
            reasoning_parser= os.getenv('REASONING_PARSER', "") or None,
            # return_token_as_token_ids=False,
            enable_auto_tools=os.getenv('ENABLE_AUTO_TOOL_CHOICE', 'false').lower() == 'true',
            tool_parser=os.getenv('TOOL_CALL_PARSER', "") or None,
            enable_prompt_tokens_details=False
        )
        self.completion_engine = OpenAIServingCompletion(
            engine_client=self.llm, 
            model_config=self.model_config,
            models=self.serving_models,
            request_logger=None,
            # return_token_as_token_ids=False,
        )
    
    async def generate(self, openai_request: JobInput):
        if openai_request.openai_route == "/v1/models":
            yield await self._handle_model_request()
        elif openai_request.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(openai_request):
                yield response
        else:
            yield create_error_response("Invalid route").model_dump()
    
    async def _handle_model_request(self):
        models = await self.serving_models.show_available_models()
        return models.model_dump()
    
    async def _handle_chat_or_completion_request(self, openai_request: JobInput):
        if openai_request.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
            generator_function = self.chat_engine.create_chat_completion
        elif openai_request.openai_route == "/v1/completions":
            request_class = CompletionRequest
            generator_function = self.completion_engine.create_completion
        
        try:
            request = request_class(
                **openai_request.openai_input
            )
        except Exception as e:
            yield create_error_response(str(e)).model_dump()
            return
        
        dummy_request = DummyRequest()
        response_generator = await generator_function(request, raw_request=dummy_request)

        # FIX: ErrorResponse 먼저 체크 (스트리밍 여부와 별개)
        if isinstance(response_generator, ErrorResponse):
            yield response_generator.model_dump()
            return
            
        # Non-streaming mode
        if not openai_request.openai_input.get("stream", False):
            yield response_generator.model_dump()
            return
        
        # Streaming mode
        batch = []
        batch_token_counter = 0
        batch_size = BatchSize(self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor)
    
        try:
            async for chunk_str in response_generator:
                # Skip [DONE] message (배치에 추가하기 전에)
                if "[DONE]" in chunk_str:
                    continue
                
                # Skip empty chunks
                if not chunk_str or not chunk_str.strip():
                    continue
                
                # FIX: 더 엄격한 데이터 청크 체크 (startswith 사용)
                if chunk_str.startswith("data:") or chunk_str.startswith("data "):
                    try:
                        if self.raw_openai_output:
                            data = chunk_str
                        else:
                            # Remove "data: " prefix and parse JSON
                            chunk_data = chunk_str.removeprefix("data:").removeprefix("data: ").removeprefix("data ").strip()
                            if not chunk_data:
                                continue
                            data = json.loads(chunk_data)
                        
                        batch.append(data)
                        batch_token_counter += 1
                        
                        # Yield when batch size is reached
                        if batch_token_counter >= batch_size.current_batch_size:
                            if self.raw_openai_output:
                                yield "".join(batch)
                            else:
                                yield batch
                            batch = []
                            batch_token_counter = 0
                            batch_size.update()
                            
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse chunk: {chunk_str}, error: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error processing chunk: {e}")
                        continue
        except Exception as e:
            logging.error(f"Error during streaming: {e}")
            # Yield any remaining batch before raising error
            if batch:
                if self.raw_openai_output:
                    yield "".join(batch)
                else:
                    yield batch
            raise
                    
        # FIX: 마지막 배치 반드시 yield (스트림 종료 시)
        if batch:
            if self.raw_openai_output:
                yield "".join(batch)
            else:
                yield batch
            