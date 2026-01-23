import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    """
    Main handler for RunPod serverless requests.
    Streams results in batches to optimize HTTP calls while preventing token loss.
    """
    try:
        job_input = JobInput(job["input"])
        engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        
        # Stream all batches from the generator
        async for batch in results_generator:
            if batch:  # Only yield non-empty batches
                yield batch
                
    except Exception as e:
        # Log and yield error response
        import logging
        logging.error(f"Error in handler: {e}")
        yield {"error": str(e)}

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)