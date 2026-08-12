import asyncio
import contextlib
import gc
import json
import os
import os.path as osp
from pathlib import Path

import lmdeploy
from lmdeploy import Pipeline, PytorchEngineConfig, TurbomindEngineConfig, Tokenizer
from lmdeploy.messages import GenerationConfig

from modules import shared
from modules.logging_colors import logger

import torch

CACHE_SENTINEL = 'tm_weights_cache'
_WEIGHT_CACHE_VERSION = '1.0'


def _parse_extra_flags(flags_str: str | None) -> dict:
    if not flags_str:
        return {}

    flags_str = flags_str.strip()
    if not flags_str:
        return {}

    if flags_str.startswith('-'):
        import shlex
        tokens = shlex.split(flags_str)
        kwargs = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('--'):
                key = token[2:].replace('-', '_')
                i += 1
                if i < len(tokens) and not tokens[i].startswith('-'):
                    kwargs[key] = tokens[i]
                    i += 1
            elif token.startswith('-'):
                key = token[1:].replace('-', '_')
                i += 1
                if i < len(tokens) and not tokens[i].startswith('-'):
                    kwargs[key] = tokens[i]
                    i += 1
            else:
                i += 1
        return kwargs
    else:
        kwargs = {}
        for item in flags_str.split(','):
            item = item.strip()
            if not item:
                continue
            if '=' in item:
                key, value = item.split('=', 1)
                kwargs[key.strip().replace('-', '_')] = value.strip()
            else:
                kwargs[item.strip().replace('-', '_')] = True
        return kwargs


# ---------------------------------------------------------------------------
# CPU checkpoint conversion  (v0.15 engine, low-VRAM safe)
# ---------------------------------------------------------------------------

class _cpu_checkpoint_load:
    """
    Keep the HF -> TurboMind conversion on CPU so the V100 (16 GB) survives
    the load.

    v0.15 streams checkpoints through SafetensorsCheckpoint.get/pop, each of
    which moves its tensor to GPU (``.cuda()``). During conversion that pushes
    the peak (final weights + fp16 dequant transients + the GDN state cache)
    past 16 GB. Patching get/pop to return the mmap-backed CPU tensor keeps
    every intermediate on CPU; ``_copy_shard_to_param`` then performs the
    single H2D copy into each pre-allocated C++ param slot. Same goal as the
    pre-0.15 ``_cpu_realtime_conversion`` patch, one order of magnitude
    smaller.
    """

    def __enter__(self):
        from lmdeploy.turbomind.checkpoint import PytorchCheckpoint, SafetensorsCheckpoint
        self._classes = (SafetensorsCheckpoint, PytorchCheckpoint)
        logger.info("[cpu_checkpoint] CPU conversion active (checkpoint get/pop patched)")
        self._saved = {}
        for cls in self._classes:
            for method in ('get', 'pop'):
                self._saved[(cls, method)] = getattr(cls, method)

        def _cpu_get(self_, key, index=None):
            t = self_._data[key]
            if index is not None:
                t = t[index]
            return t

        def _cpu_pop(self_, key, index=None):
            t = self_._data.pop(key)
            if index is not None:
                t = t[index]
            return t

        for cls in self._classes:
            cls.get = _cpu_get
            cls.pop = _cpu_pop
        return self

    def __exit__(self, *_):
        for (cls, method), orig in self._saved.items():
            setattr(cls, method, orig)


# ---------------------------------------------------------------------------
# Workspace / cache helpers
# ---------------------------------------------------------------------------

def get_weight_cache_dir(model_path: str) -> str:
    base = Path(model_path).resolve()
    return str(base.parent / f'{base.name}_tm_cache')


def check_weight_cache(model_path: str) -> bool:
    cache_dir = get_weight_cache_dir(model_path)
    sentinel = osp.join(cache_dir, CACHE_SENTINEL)
    version_file = osp.join(cache_dir, '.version')
    if not osp.exists(sentinel) or not osp.exists(version_file):
        return False
    try:
        if Path(version_file).read_text().strip() != _WEIGHT_CACHE_VERSION:
            return False
    except Exception:
        return False
    return (osp.exists(osp.join(cache_dir, 'config.json'))
            and osp.exists(osp.join(cache_dir, 'rank00.safetensors')))


def _resolve_model_source(model_path: str) -> str:
    """v0.15 dropped the on-disk TurboMind workspace; a 'converted' model dir
    is now just a pointer to its HF source (written by
    convert_model_to_turbomind)."""
    source_file = osp.join(model_path, '.source_model')
    if osp.exists(source_file):
        candidate = Path(source_file).read_text().strip()
        if osp.exists(candidate):
            logger.info(f"Redirecting model load to HF source: {candidate}")
            return candidate
        logger.warning(f".source_model points to missing path: {candidate}")
    return model_path


# ---------------------------------------------------------------------------
# Model conversion  (load-time verification; no on-disk workspace in v0.15)
# ---------------------------------------------------------------------------

def convert_model_to_turbomind(model_path: str, output_dir: str, model_format: str, tp: int = 1):
    """Verify the model converts under the v0.15 engine and stage the output
    dir as a pointer to the HF source.

    v0.15 removed the on-disk workspace format: the engine converts HF
    checkpoints on the fly at load time. This keeps the TGWUI 'Convert' flow
    meaningful as an arch/format validation step and produces a model dir that
    from_pretrained() redirects back to the original source.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        logger.info(f"Validating {model_path} for the v0.15 engine (tp={tp})...")
        from lmdeploy.turbomind.converter import get_tm_config
        engine_config = TurbomindEngineConfig(tp=tp, model_format=model_format or None)
        model, _, _ = get_tm_config(str(model_path), engine_config, trust_remote_code=False)

        Path(osp.join(output_dir, '.source_model')).write_text(str(model_path))

        _TOKENIZER_FILES = {
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
            'vocab.json', 'merges.txt', 'tokenizer.model', 'added_tokens.json', 'spm.model',
        }
        import shutil
        for fname in _TOKENIZER_FILES:
            src = osp.join(model_path, fname)
            if osp.exists(src):
                shutil.copy2(src, osp.join(output_dir, fname))

        archs = getattr(getattr(model, '_config', None), 'architectures', None) or [type(model).__name__]
        with open(osp.join(output_dir, 'config.json'), 'w') as f:
            json.dump({'architectures': archs}, f)

        logger.info(f"Conversion verified. Model source staged at {output_dir}")
        return True
    except Exception:
        import traceback
        logger.error(f"Model conversion failed:\n{traceback.format_exc()}")
        return False


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class LMDeployModel:
    def __init__(self, pipeline, _tokenizer, _last_prompt_token_count: int):
        self.pipeline = pipeline
        self._tokenizer = _tokenizer
        self._last_prompt_token_count = _last_prompt_token_count

    def __getattr__(self, name):
        for obj in (self.pipeline, self._tokenizer):
            attr = getattr(obj, name, None)
            if attr is not None:
                return attr
        raise AttributeError(f"'LMDeployModel' has no attribute '{name}'")

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_pretrained(cls, path_to_model: str | Path):
        path_to_model = (Path(shared.args.model_dir) / path_to_model).resolve()
        model_path_str = _resolve_model_source(str(path_to_model))

        backend = shared.args.backend or 'turbomind'
        tp = shared.args.tensor_parallel or 1
        cache_type = (shared.args.cache_type or 'fp16').lower()
        ctx_size = getattr(shared.args, 'ctx_size', 4096) or 4096
        max_batch_size = getattr(shared.args, 'max_batch_size', 1) or 1
        cache_max_entry_count = getattr(shared.args, 'cache_max_entry_count', None) or 0.8
        cpu_realtime = getattr(shared.args, 'cpu_realtime_conversion', True)
        extra_flags = _parse_extra_flags(getattr(shared.args, 'extra_flags', None))

        quant_policy = {'q8': 8, 'q4': 4}.get(cache_type, 0)
        prefix_caching = not getattr(shared.args, 'disable_prefix_caching', False)

        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            f"[lmdeploy] from_pretrained: backend={backend} cpu_realtime={cpu_realtime} "
            f"prefix_caching={prefix_caching} session_len={ctx_size} max_batch_size={max_batch_size}"
        )

        try:
            if backend == 'turbomind':
                tm_kwargs = dict(
                    tp=tp,
                    session_len=ctx_size,
                    max_batch_size=max_batch_size,
                    quant_policy=quant_policy,
                    cache_max_entry_count=cache_max_entry_count,
                    max_prefill_token_num=2048,
                    num_tokens_per_iter=2048,
                    enable_prefix_caching=prefix_caching,
                )
                if prefix_caching:
                    tm_kwargs.update(
                        cache_prompt='all',
                        cache_generation='all',
                        cache_checkpoint_interval=4096,
                    )
                tm_kwargs.update(extra_flags)
                engine_config = TurbomindEngineConfig(**tm_kwargs)

                load_ctx = _cpu_checkpoint_load() if cpu_realtime else contextlib.nullcontext()
                with load_ctx:
                    pipeline = Pipeline(
                        model_path_str,
                        backend_config=engine_config,
                        log_level='INFO',
                    )

            elif backend == 'pytorch':
                pt_kwargs = dict(
                    tp=tp,
                    session_len=ctx_size,
                    max_batch_size=max_batch_size,
                    cache_max_entry_count=cache_max_entry_count,
                    eager_mode=True,
                )
                pt_kwargs.update(extra_flags)
                engine_config = PytorchEngineConfig(**pt_kwargs)
                pipeline = Pipeline(
                    model_path_str,
                    backend_config=engine_config,
                    log_level='INFO',
                )

            else:
                raise ValueError(f"Unknown backend '{backend}'")

            tokenizer = pipeline.async_engine.tokenizer

        except Exception:
            import traceback
            logger.error(f"Failed to load model:\n{traceback.format_exc()}")
            return None, None

        result = cls(pipeline, LMDeployTokenizerWrapper(tokenizer), 0)
        return result, result._tokenizer

    def is_multimodal(self):
        return False

    def generate_with_streaming(self, prompt, state):
        input_ids = self.encode(prompt, add_bos=False)
        prompt_token_count = len(input_ids)

        gen_config = self._prepare_generation_config(state)

        max_new_tokens = state['max_new_tokens']
        if state.get('auto_max_new_tokens'):
            max_new_tokens = state['truncation_length'] - prompt_token_count
        gen_config.max_new_tokens = max_new_tokens

        stop_event = state.get('stop_event')
        response_text = ""

        try:
            output_queue = asyncio.Queue()

            async def run_async():
                session = self.pipeline.session()
                async for out in self.pipeline.async_engine.generate(
                    messages=None,
                    session_id=session,
                    gen_config=gen_config,
                    input_ids=input_ids,
                    do_preprocess=False,
                    stream_response=True,
                ):
                    if shared.stop_everything or (stop_event and stop_event.is_set()):
                        break
                    await output_queue.put(out.response if out.response else '')

            loop = self.pipeline.internal_thread.loop
            future = asyncio.run_coroutine_threadsafe(run_async(), loop)

            while not future.done():
                try:
                    response_text += output_queue.get_nowait()
                    yield response_text
                except asyncio.QueueEmpty:
                    pass

        except (GeneratorExit, StopIteration, asyncio.CancelledError):
            pass
        except Exception:
            import traceback
            logger.error(f"Generation error:\n{traceback.format_exc()}")
            yield response_text

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass
        return output

    def _prepare_generation_config(self, state):
        temperature = state['temperature']
        if state.get('dynamic_temperature'):
            temperature = (state['dynatemp_low'] + state['dynatemp_high']) / 2

        return GenerationConfig(
            temperature=temperature,
            top_k=state['top_k'] if state['top_k'] > 0 else 128,
            top_p=state['top_p'],
            repetition_penalty=state['repetition_penalty'],
            ignore_eos=state.get('ban_eos_token', False),
            skip_special_tokens=state.get('skip_special_tokens', True),
            random_seed=state.get('seed') if (state.get('seed', -1) or -1) >= 0 else None,
        )

    def get_logits(self, token_ids, **kwargs):
        return None

    def encode(self, string, **kwargs):
        return self._tokenizer.encode(string, add_bos_token=kwargs.get('add_bos', True))

    def decode(self, ids, **kwargs):
        return self._tokenizer.decode(ids)

    @property
    def last_prompt_token_count(self):
        # Kept because external modules rely on it: text_generation.py:408 writes it
        # before each generation call, and chat.py / api/completions.py read it for
        # context-length hints and usage.prompt_tokens.
        return self._last_prompt_token_count

    @last_prompt_token_count.setter
    def last_prompt_token_count(self, value):
        self._last_prompt_token_count = value

    def unload(self):
        logger.info("Unloading LMDeploy model...")
        if getattr(self, 'pipeline', None) is not None:
            self.pipeline.close()
            self.pipeline = None
        self._tokenizer = None


class LMDeployTokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self._wrapped = tokenizer

    def __getattr__(self, name):
        return getattr(self._wrapped, name, getattr(self._wrapped.model, name, None))
