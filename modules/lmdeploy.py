import gc
import os
import threading
import queue
from pathlib import Path
from typing import Any, List, Optional, Tuple

import lmdeploy
from lmdeploy import Pipeline, PytorchEngineConfig, TurbomindEngineConfig, Tokenizer
from lmdeploy.messages import GenerationConfig

from modules import shared
from modules.logging_colors import logger


def convert_model_to_turbomind(model_path: str, output_dir: str, model_format: str, tp: int = 1):
    """Convert a HuggingFace model to TurboMind format and save to disk."""
    try:
        from lmdeploy.turbomind.deploy.converter import get_tm_model

        os.makedirs(output_dir, exist_ok=True)

        engine_config = TurbomindEngineConfig(
            tp=tp,
            model_format=model_format,
        )

        logger.info(f"Converting {model_path} to TurboMind format (tp={tp})...")
        tm_model = get_tm_model(
            model_path=model_path,
            model_name=None,
            chat_template_name=None,
            engine_config=engine_config,
            out_dir=output_dir,
        )
        logger.info("Exporting TurboMind model to disk...")
        tm_model.export()

        config_yaml_path = Path(output_dir) / 'config.yaml'
        if config_yaml_path.exists():
            import yaml
            with open(config_yaml_path) as f:
                cfg = yaml.safe_load(f)
            model_arch = cfg.get('model_config', {}).get('model_arch', '')
            model_cfg = cfg.get('model_config', {})
            arch_map = {
                'Qwen3_5ForConditionalGeneration': 'Qwen2ForCausalLM',
                'Qwen2ForCausalLM': 'Qwen2ForCausalLM',
                'LlamaForCausalLM': 'LlamaForCausalLM',
            }
            arch = arch_map.get(model_arch, 'Qwen2ForCausalLM')
            type_map = {
                'Qwen3_5ForConditionalGeneration': 'qwen2',
                'Qwen2ForCausalLM': 'qwen2',
                'LlamaForCausalLM': 'llama',
            }
            model_type = type_map.get(model_arch, 'qwen2')
            config_json = {
                'model_type': model_type,
                'architectures': [arch],
                'num_hidden_layers': model_cfg.get('num_layer', 64),
                'vocab_size': model_cfg.get('vocab_size', 151936),
            }
            with open(Path(output_dir) / 'config.json', 'w') as f:
                import json
                json.dump(config_json, f)

        logger.info(f"Conversion complete. Saved to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Model conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


class LMDeployModel:
    def __getattr__(self, name):
        attr = getattr(self.pipeline, name, None) or getattr(self.pipeline.tokenizer, name, None)
        if attr is not None:
            return attr
        raise AttributeError(f"'LMDeployModel' object has no attribute '{name}'")

    @property
    def device(self):
        import torch
        return lmdeploy.device.cuda.current_device() if hasattr(lmdeploy, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_pretrained(cls, path_to_model):
        # Resolve to absolute path so lmdeploy's internal loaders find weight files
        # regardless of the process's working directory.
        path_to_model = (Path(shared.args.model_dir) / path_to_model).resolve()

        backend = shared.args.backend or 'turbomind'
        tp = shared.args.tensor_parallel or 1
        cache_type = (shared.args.cache_type or 'fp16').lower()
        ctx_size = getattr(shared.args, 'ctx_size', 4096) or 4096
        max_batch_size = getattr(shared.args, 'max_batch_size', 1) or 1
        cache_max_entry_count = getattr(shared.args, 'cache_max_entry_count', None) or 0.05

        quant_policy = 0
        if cache_type in ('q8', 'int8'):
            quant_policy = 8
        elif cache_type in ('q4', 'int4'):
            quant_policy = 4

        model_format = None
        config_path = path_to_model / 'config.json'
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    config = json.load(f)
                quant_method = (
                    config.get('quantization_config', {}).get('quant_method', '') or
                    config.get('quant_method', '')
                ).lower()
                if 'awq' in quant_method:
                    model_format = 'awq'
                elif 'gptq' in quant_method:
                    model_format = 'gptq'
                elif 'compressed' in quant_method:
                    model_format = 'compressed-tensors'
                elif 'hqq' in quant_method:
                    model_format = 'hqq'
                elif quant_method:
                    model_format = 'hf'
            except Exception:
                pass

        if model_format is None:
            name_lower = path_to_model.name.lower()
            if 'awq' in name_lower:
                model_format = 'awq'
            elif 'gptq' in name_lower or 'gq' in name_lower:
                model_format = 'gptq'
            elif 'hqq' in name_lower:
                model_format = 'hqq'

        is_turbomind_format = (
            (path_to_model / 'config.yaml').exists() or
            (path_to_model / 'config.ini').exists()
        )
        if is_turbomind_format:
            # Pre-converted; lmdeploy detects the format internally.
            model_format = None

        if backend == 'turbomind':
            engine_config = TurbomindEngineConfig(
                tp=tp,
                session_len=ctx_size,
                max_batch_size=max_batch_size,
                quant_policy=quant_policy,
                model_format=model_format,
                cache_max_entry_count=cache_max_entry_count,
                max_prefill_token_num=256,
                enable_prefix_caching=False,
            )
        elif backend == 'pytorch':
            engine_config = PytorchEngineConfig(
                tp=tp,
                session_len=ctx_size,
                max_batch_size=max_batch_size,
                cache_max_entry_count=cache_max_entry_count,
                # Skips CUDA graph capture; no throughput cost in single-user scenarios.
                eager_mode=True,
            )
        else:
            logger.warning(f"Unknown backend '{backend}', defaulting to turbomind")
            engine_config = TurbomindEngineConfig(
                tp=tp,
                session_len=ctx_size,
                cache_max_entry_count=0.05,
                max_prefill_token_num=256,
            )

        gc.collect()
        import torch
        torch.cuda.empty_cache()

        try:
            pipeline = Pipeline(
                str(path_to_model),
                backend_config=engine_config,
                log_level='INFO',
            )
        except Exception as e:
            logger.error(f"Failed to load model with LMDeploy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

        result = cls()
        result.pipeline = pipeline
        result._tokenizer = LMDeployTokenizerWrapper(pipeline.async_engine.tokenizer)
        result._last_prompt_token_count = 0

        return result, result._tokenizer

    def is_multimodal(self) -> bool:
        return False

    def generate_with_streaming(self, prompt, state):
        gen_config = self._prepare_generation_config(state)

        max_new_tokens = state['max_new_tokens']
        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - self._last_prompt_token_count

        gen_config.max_new_tokens = max_new_tokens

        stop_event = state.get('stop_event')
        response_text = ""

        try:
            for chunk in self.pipeline.stream_infer([prompt], gen_config=gen_config):
                if shared.stop_everything or (stop_event and stop_event.is_set()):
                    break
                if chunk:
                    token = chunk.text.decode() if isinstance(chunk.text, bytes) else str(chunk.text)
                    response_text += token
                    yield response_text
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield response_text

    def generate(self, prompt, state):
        gen_config = self._prepare_generation_config(state)

        max_new_tokens = state['max_new_tokens']
        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - self._last_prompt_token_count

        gen_config.max_new_tokens = max_new_tokens

        try:
            response = self.pipeline.infer([prompt], gen_config=gen_config)
            if response:
                return response[0].text.decode() if isinstance(response[0].text, bytes) else str(response[0].text)
        except Exception as e:
            logger.error(f"Error during generation: {e}")

        return ""

    def _prepare_generation_config(self, state) -> GenerationConfig:
        temperature = state['temperature']
        if state['dynamic_temperature']:
            temperature = (state['dynatemp_low'] + state['dynatemp_high']) / 2

        top_k = state['top_k'] if state['top_k'] > 0 else 128
        top_p = state['top_p']
        repetition_penalty = state['repetition_penalty']

        ignore_eos = state.get('ban_eos_token', False)
        skip_special_tokens = state.get('skip_special_tokens', True)
        seed = state.get('seed', -1)

        return GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            ignore_eos=ignore_eos,
            skip_special_tokens=skip_special_tokens,
            random_seed=seed if seed >= 0 else None,
        )

    def get_logits(self, token_ids, **kwargs):
        return None

    def encode(self, string, **kwargs):
        add_bos = kwargs.get('add_bos', True)
        return self._tokenizer.encode(string, add_bos_token=add_bos)

    def decode(self, ids, **kwargs):
        return self._tokenizer.decode(ids)

    @property
    def last_prompt_token_count(self):
        return self._last_prompt_token_count

    @last_prompt_token_count.setter
    def last_prompt_token_count(self, value):
        self._last_prompt_token_count = value

    def unload(self):
        if hasattr(self, 'pipeline'):
            del self.pipeline
            self.pipeline = None
        if hasattr(self, '_tokenizer'):
            self._tokenizer = None


class LMDeployTokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self._wrapped = tokenizer

    def __getattr__(self, name):
        return getattr(self._wrapped, name, getattr(self._wrapped.model, name))