################################################################
# Simple Inference class for OpenAI API (Direct)
################################################################

import os
import time
import random
import concurrent.futures
from openai import OpenAI
from openai.types.chat import ChatCompletion
from vllm import SamplingParams, RequestOutput, CompletionOutput
from src.utils.utils import init_logger
logger = init_logger()

class OpenAIInference:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Initial API key from environment variable.
        initial_api_key = os.getenv("OPENAI_API_KEY")
        if not initial_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please add 'OPENAI_API_KEY=your_api_key_here' to your .env file."
            )
        
        logger.info("✅ OpenAI API Key loaded successfully")
        self.client = OpenAI(
            api_key=initial_api_key
        )

    def create_client(self, api_key: str):
        client = OpenAI(
            api_key=api_key
        )
        return client

    def run_batch(self, conversations: list, sampling_params: SamplingParams, meta=None):
        def _execute_one(conversation):
            max_retries = 100000
            backoff = 1
            # Start with the API key from the environment.
            current_api_key = os.getenv("OPENAI_API_KEY")
            for attempt in range(1, max_retries + 1):
                try:
                    client = self.create_client(current_api_key)
                    completion_outputs: list[CompletionOutput] = []
                    for i in range(sampling_params.n):
                        completion: ChatCompletion = client.chat.completions.create(
                            model=self.model_name,
                            messages=conversation,
                            temperature=sampling_params.temperature,
                            max_tokens=sampling_params.max_tokens,
                            top_p=sampling_params.top_p,
                        )
                        completion_outputs.append(CompletionOutput(
                            index=i,
                            text=completion.choices[0].message.content,
                            token_ids=completion.choices[0].message.content,
                            cumulative_logprob=0.0,
                            logprobs=[],
                        ))

                    req_out = RequestOutput(
                        request_id="",
                        prompt="",
                        outputs=completion_outputs,
                        prompt_token_ids=[],
                        prompt_logprobs=[],
                        finished=True
                    )
                    print(f"OpenAI API attempt {attempt} succeeded")
                    return req_out
                except Exception as e:
                    logger.warning(f"OpenAI API attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff, capped below.
                        if backoff > 30:
                            backoff = 30
                        jitter = random.uniform(0, 5)
                        time.sleep(jitter)

                        # Check for fallback OpenAI API keys
                        fallback_keys = [key for key in os.environ.keys() if key.startswith("OPENAI_API_KEY")]
                        if fallback_keys:
                            fallback_key = random.choice(fallback_keys)
                            logger.warning("Switching to fallback OpenAI API key")
                            current_api_key = os.getenv(fallback_key)
                    else:
                        logger.warning(f"All OpenAI API attempts failed for conversation {conversation}")
                        return None
        
        request_outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            futures = [executor.submit(_execute_one, conversation) for conversation in conversations]
            for future in futures:
                result = future.result()
                if result is None:
                    logger.warning("One of the OpenAI API requests failed after all retries.")
                request_outputs.append(result)
        return request_outputs

    def sleep(self):
        pass 