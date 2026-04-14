import asyncio
import os
import uuid
from pathlib import Path

import torch

from modules import shared
from modules.logging_colors import logger

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
except ModuleNotFoundError:
    raise ModuleNotFoundError("Failed to import 'vllm'. Please install it manually with: pip install vllm")


def _run_async(coro):
    """Run an async coroutine in a sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)


class vLLMModel:
    def __init__(self):
        self.engine = None
        self.tokenizer = None
        self.max_model_len = None
        self.last_prompt_token_count = 0

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    @classmethod
    def from_pretrained(cls, model_name):
        path_to_model = Path(f'{shared.args.model_dir}') / Path(model_name)

        tensor_parallel_size = getattr(shared.args, 'tensor_parallel_size', 1)
        gpu_memory_utilization = getattr(shared.args, 'gpu_memory_utilization', 0.85)
        max_num_seqs = getattr(shared.args, 'max_num_seqs', 64)
        ctx_size = shared.args.ctx_size if shared.args.ctx_size > 0 else 8192
        gpu_devices = getattr(shared.args, 'gpu_devices', None)

        if gpu_devices:
            valid_devices = []
            for d in gpu_devices.split(','):
                d = d.strip()
                if d.startswith('CUDA'):
                    d = d.replace('CUDA', '')
                try:
                    valid_devices.append(int(d))
                except ValueError:
                    pass
            if valid_devices:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, valid_devices))
                tensor_parallel_size = len(valid_devices)
                logger.info(f"Using GPUs: {valid_devices} for tensor parallelism")

        logger.info(f"Loading vLLM model: {model_name}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"Context size: {ctx_size}")

        engine_args = AsyncEngineArgs(
            model=str(path_to_model),
            trust_remote_code=shared.args.trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=ctx_size,
            enforce_eager=False,
        )

        engine = AsyncLLMEngine.from_engine_args(engine_args)

        result = cls()
        result.engine = engine
        result.max_model_len = ctx_size

        async def get_tokenizer():
            return await engine.get_tokenizer()

        try:
            result.tokenizer = _run_async(get_tokenizer())
        except Exception as e:
            logger.warning(f"Could not get tokenizer from vLLM: {e}")
            result.tokenizer = None

        logger.info(f"vLLM model loaded successfully. Max context length: {ctx_size}")

        return result, result

    def encode(self, text, **kwargs):
        """Encode text to token IDs using vLLM tokenizer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")

        add_bos_token = kwargs.get('add_bos_token', False)
        add_eos_token = kwargs.get('add_eos_token', False)

        return self.tokenizer.encode(text, add_special_tokens=not add_bos_token, add_bos_token=add_bos_token, add_eos_token=add_eos_token)

    def is_multimodal(self) -> bool:
        """Check if this model supports multimodal input."""
        return False

    def generate_with_streaming(self, prompt, state):
        """
        Generate text with streaming using vLLM async engine.
        Yields accumulated text instead of just new tokens.
        """
        temperature = state.get('temperature', 1.0)
        top_p = state.get('top_p', 1.0)
        top_k = state.get('top_k', -1)
        max_new_tokens = state.get('max_new_tokens', 256)
        repetition_penalty = state.get('repetition_penalty', 1.0)
        presence_penalty = state.get('presence_penalty', 0.0)
        frequency_penalty = state.get('frequency_penalty', 0.0)

        stop_token_ids = []
        if state.get('ban_eos_token', False):
            if self.tokenizer:
                stop_token_ids.append(self.tokenizer.eos_token_id)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        request_id = f"tgw-{uuid.uuid4().hex[:8]}"

        if state.get('auto_max_new_tokens', False):
            prompt_tokens = self.encode(prompt, add_bos_token=state.get('add_bos_token', False))
            self.last_prompt_token_count = len(prompt_tokens)
            max_new_tokens = state.get('truncation_length', 8192) - self.last_prompt_token_count
            sampling_params.max_tokens = max(max_new_tokens, 1)
        else:
            prompt_tokens = self.encode(prompt, add_bos_token=state.get('add_bos_token', False))
            self.last_prompt_token_count = len(prompt_tokens)

        stop_event = state.get('stop_event')
        full_text = ""
        prev_text_len = 0

        async def generate():
            async_generator = self.engine.generate(prompt, sampling_params, request_id)
            async for request_output in async_generator:
                if request_output.finished:
                    break
                if shared.stop_everything or (stop_event and stop_event.is_set()):
                    break
                text = request_output.outputs[0].text
                if text and len(text) > prev_text_len:
                    yield text
                    prev_text_len = len(text)

        try:
            for _ in _run_async(generate()):
                if _:
                    full_text = _
                    yield full_text
        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}")
            raise

    def generate(self, prompt, state):
        """Generate without streaming for non-streaming mode"""
        output = ""
        for output in self.generate_with_streaming(prompt, state):
            pass
        return output

    def unload(self):
        """Unload the model and clean up resources"""
        if self.engine:
            self.engine = None
        logger.info("vLLM model unloaded")