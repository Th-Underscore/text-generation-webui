import asyncio
import contextlib
import gc
import json
import os
import os.path as osp
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import lmdeploy
from lmdeploy import Pipeline, PytorchEngineConfig, TurbomindEngineConfig, Tokenizer
from lmdeploy.messages import GenerationConfig

from modules import shared
from modules.logging_colors import logger

import torch
import yaml

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
# Workspace loader
# ---------------------------------------------------------------------------

class _workspace_load_hook:
    """
    Context manager that patches TurboMind to load a pre-converted workspace
    without re-running the HF→TurboMind conversion.

    Critical dtype contract: _tofile() always saves floating-point weights as
    _weight_dtype_map(weight_type) — fp16 or bf16 — even when the TurboMind
    VRAM buffer for that weight is TYPE_FP32 (e.g. norm/embedding layers).
    _copy() must therefore probe the save dtype from weight_type, not from
    ref.dtype, and cast up to ref.dtype before the H2D copy.
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

    def __enter__(self):
        import lmdeploy.turbomind.turbomind as _tm_mod
        import lmdeploy.archs as _archs_mod
        import lmdeploy.tokenizer as _tok_mod
        from lmdeploy.turbomind.deploy.config import TurbomindModelConfig

        self._tm_mod = _tm_mod
        self._archs_mod = _archs_mod
        self._tok_mod = _tok_mod
        self._orig_from_hf = _tm_mod.TurboMind._from_hf
        self._orig_load_weights = _tm_mod.TurboMind._load_weights
        self._orig_autoget_backend = _archs_mod.autoget_backend
        self._orig_tokenizer_init = _tok_mod.HuggingFaceTokenizer.__init__
        wp = self.workspace_path

        source_model_path = None
        source_model_file = osp.join(wp, '.source_model')
        if osp.exists(source_model_file):
            candidate = Path(source_model_file).read_text().strip()
            if osp.exists(candidate):
                source_model_path = candidate
            else:
                logger.warning(f"[workspace_hook] .source_model points to missing path: {candidate}")

        _orig_tok_init = self._orig_tokenizer_init

        def _patched_tokenizer_init(self_tok, model_dir):
            if source_model_path and osp.abspath(model_dir) == osp.abspath(wp):
                logger.info(f"[workspace_hook] Tokenizer redirected to {source_model_path}")
                model_dir = source_model_path
            _orig_tok_init(self_tok, model_dir)

        def _autoget_backend(model_path):
            if osp.abspath(model_path) == osp.abspath(wp):
                return 'turbomind'
            return self._orig_autoget_backend(model_path)

        def _from_hf(self_tm, model_path, engine_config):
            import _turbomind as _tm_c
            with open(osp.join(wp, 'config.yaml')) as f:
                cfg_data = yaml.safe_load(f)
            logger.info(
                f"[workspace_hook] config.yaml: "
                f"model_arch={cfg_data.get('model_config', {}).get('model_arch')} "
                f"weight_type={cfg_data.get('model_config', {}).get('weight_type')}"
            )
            tm_config = TurbomindModelConfig.from_dict(cfg_data)
            self_tm._postprocess_config(tm_config, engine_config)

            # model_dir='' matches the standard in-memory path. Passing the
            # workspace path here causes the C++ engine to attempt its own disk
            # load, which conflicts with our Python-side weight filling.
            model_comm = _tm_c.TurboMind.create(
                model_dir='',
                config=yaml.safe_dump(self_tm.config_dict),
                weight_type=self_tm.config.model_config.weight_type,
            )
            self_tm._create_weight(model_comm)

            class _TmModelStub:
                tm_params = {}

            self_tm._tm_model = _TmModelStub()
            return model_comm

        def _load_weights_from_workspace(self_tm):
            import numpy as np

            weight_type = self_tm.config.model_config.weight_type

            _NP_DTYPE = {
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.int32:   np.int32,
                torch.int8:    np.int8,
                torch.uint8:   np.uint8,
            }

            def _read_file(fpath, dtype):
                """Read a raw weight file as written by BaseOutputModel._tofile()."""
                if dtype == torch.bfloat16:
                    # _tofile() views bf16 as float16 before handing to numpy.
                    arr = np.fromfile(fpath, dtype=np.uint16).copy()
                    return torch.from_numpy(arr).view(torch.bfloat16)
                np_dtype = _NP_DTYPE.get(dtype)
                if np_dtype is None:
                    raise ValueError(f"Unsupported dtype {dtype} for {fpath}")
                return torch.from_numpy(np.fromfile(fpath, dtype=np_dtype).copy())

            def _save_dtype_for(weight_type):
                """
                The dtype _tofile() used when saving floating-point weights.
                Always fp16/bf16 regardless of the VRAM buffer dtype. Determined
                by _weight_dtype_map, which maps weight_type → torch dtype.
                Reading the file with ref.dtype instead (e.g. fp32 for norm
                layers) reads half the elements and leaves buffers uninitialised.
                """
                from lmdeploy.turbomind.deploy.target_model.base import _weight_dtype_map
                return _weight_dtype_map(weight_type, torch.float16)

            save_dtype = _save_dtype_for(weight_type)

            def _copy(tm_tensor, fpath):
                try:
                    import _turbomind as _tm_c
                    if tm_tensor.type == _tm_c.DataType.TYPE_UINT32:
                        tm_tensor = tm_tensor.view(_tm_c.DataType.TYPE_INT32)
                except Exception:
                    pass

                try:
                    ref = torch.from_dlpack(tm_tensor)
                except Exception as exc:
                    logger.warning(f"[workspace_hook] from_dlpack failed ({osp.basename(fpath)}): {exc}")
                    return False

                try:
                    file_dtype = save_dtype if torch.is_floating_point(ref) else ref.dtype
                    data = _read_file(fpath, file_dtype)
                    if data.dtype != ref.dtype:
                        data = data.to(ref.dtype)
                    ref.copy_(data.reshape(ref.shape))
                    return True
                except Exception as exc:
                    logger.warning(f"[workspace_hook] copy failed ({osp.basename(fpath)}): {exc}")
                    return False

            def _device_id_of(tm_tensor):
                # __dlpack_device__ returns (type, id) without consuming the capsule.
                try:
                    _, dev_id = tm_tensor.__dlpack_device__()
                    return dev_id
                except Exception:
                    return 0

            self_tm._get_model_params()
            tm_params = self_tm._tm_model.tm_params
            missing = []
            n_copied = 0

            for name in list(tm_params.keys()):
                tm_tensors = tm_params.pop(name)
                bare_path = osp.join(wp, name)

                if osp.exists(bare_path):
                    # Replicated weight: same file written to every device buffer.
                    for tm_tensor in tm_tensors:
                        _copy(tm_tensor, bare_path)
                    n_copied += 1
                else:
                    # Split weight: per-rank files name.0, name.1, ...
                    per_rank = [osp.join(wp, f'{name}.{i}') for i in range(len(tm_tensors))]
                    if all(osp.exists(p) for p in per_rank):
                        if len(tm_tensors) > 1:
                            tm_tensors = sorted(tm_tensors, key=_device_id_of)
                        for tm_tensor, fpath in zip(tm_tensors, per_rank):
                            _copy(tm_tensor, fpath)
                        n_copied += 1
                    elif osp.exists(per_rank[0]):
                        for tm_tensor in tm_tensors:
                            _copy(tm_tensor, per_rank[0])
                        n_copied += 1
                    else:
                        missing.append(name)

            logger.info(f"[workspace_hook] Loaded {n_copied} weight tensors from workspace")
            if missing:
                logger.warning(f"[workspace_hook] {len(missing)} weights not found on disk: {missing[:5]}")

        _tm_mod.TurboMind._from_hf = _from_hf
        _tm_mod.TurboMind._load_weights = _load_weights_from_workspace
        _archs_mod.autoget_backend = _autoget_backend
        _tok_mod.HuggingFaceTokenizer.__init__ = _patched_tokenizer_init
        return self

    def __exit__(self, *_):
        self._tm_mod.TurboMind._from_hf = self._orig_from_hf
        self._tm_mod.TurboMind._load_weights = self._orig_load_weights
        self._archs_mod.autoget_backend = self._orig_autoget_backend
        self._tok_mod.HuggingFaceTokenizer.__init__ = self._orig_tokenizer_init


# ---------------------------------------------------------------------------
# CPU realtime conversion  (no VRAM peak during weight export)
# ---------------------------------------------------------------------------

class _cpu_realtime_conversion:
    """
    Patches the entire LMDeploy conversion pipeline to stay on CPU throughout.

    Normally each input policy (process_awq_gemm, to_cuda, etc.) sends the
    tensor to GPU before export_weight receives it. export_weight then
    allocates a second GPU buffer for the dtype cast. For a 27B AWQ model
    on V100 this double-buffer peak exceeds available VRAM.

    Patches applied:
      - All six input policy functions in lmdeploy.turbomind.deploy.policy
        are replaced with CPU equivalents (same logic, .cuda() removed).
      - export_weight's in-memory branch is replaced: instead of .cuda() +
        dtype cast on GPU + copy_from(), the patched version casts on CPU
        then writes via from_dlpack() + ref.copy_(). The only H2D transfer
        is the final DMA into the pre-allocated TurboMind VRAM buffer.

    The to_file path (used by convert_model_to_turbomind) is not affected.
    """

    def __enter__(self):
        import lmdeploy.turbomind.deploy.policy as _pol
        import lmdeploy.turbomind.deploy.target_model.base as _base_mod
        from lmdeploy.turbomind.deploy.policy import get_u4_slices, unpack_awq_gemm
        from lmdeploy.turbomind.deploy.target_model.base import _weight_dtype_map

        self._pol = _pol
        self._base_mod = _base_mod
        self._saved = {
            'to_cuda':                   _pol.to_cuda,
            'process_awq_gemm':          _pol.process_awq_gemm,
            'process_gptq':              _pol.process_gptq,
            'process_mxfp4':             _pol.process_mxfp4,
            'process_fp8':               _pol.process_fp8,
            'process_compressed_tensor': _pol.process_compressed_tensor,
        }
        self._orig_export = _base_mod.BaseOutputModel.export_weight

        def _to_cpu(x, *args):
            return x.cpu()

        def _awq_cpu(x, kind):
            x = x.cpu()
            if x.dtype == torch.int32:
                x = unpack_awq_gemm(x)
            if kind in ['qweight', 'qzeros', 'scales']:
                x = x.t()
            return x

        def _gptq_cpu(x, kind):
            x = x.cpu()
            if x.dtype == torch.int32:
                xs = get_u4_slices(x, torch.uint8)
                if kind == 'qweight':
                    x = torch.stack(xs, dim=1).view(-1, x.size(-1))
                else:
                    x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
            if kind in ['qweight', 'qzeros', 'scales']:
                x = x.t()
            return x

        def _mxfp4_cpu(x, kind):
            x = x.cpu()
            if kind == 'blocks':
                xs = get_u4_slices(torch.flatten(x, start_dim=-2), torch.uint8)
                x = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
            return x

        def _fp8_cpu(x, kind):
            x = x.cpu()
            if x.dtype == torch.float8_e4m3fn:
                return x.view(dtype=torch.uint8)
            return x.to(dtype=torch.bfloat16)

        def _compressed_cpu(x, kind):
            x = x.cpu()
            if x.dtype == torch.int32:
                xs = get_u4_slices(x, torch.uint8)
                if kind == 'weight_packed':
                    x = torch.stack(xs, dim=-1).view(*x.shape[:-1], -1)
                elif kind == 'weight_zero_point':
                    x = torch.stack(xs, dim=1).view(-1, x.size(-1))
            return x

        _orig_export = self._orig_export

        def _cpu_export_weight(self_model, param, name):
            if self_model.to_file:
                _orig_export(self_model, param, name)
                return

            tm_params = self_model.tm_params
            if not tm_params or name not in tm_params:
                return

            weight_type = self_model.model_config.weight_type
            data_type = self_model.model_config.data_type

            _FLOAT_TYPES = [torch.float, torch.half, torch.bfloat16]

            if torch.is_floating_point(param):
                save_dtype = _weight_dtype_map(weight_type, torch.float16)
                torch_tensor = param.detach().cpu().to(save_dtype).contiguous()
            else:
                torch_tensor = param.detach().cpu().contiguous()

            if torch_tensor.dtype in _FLOAT_TYPES:
                if weight_type == 'fp8':
                    if torch_tensor.dtype == torch.bfloat16 and data_type == 'float16':
                        torch_tensor = torch_tensor.half()
                elif weight_type in ['float16', 'int4']:
                    torch_tensor = torch_tensor.half()
                elif weight_type == 'bfloat16':
                    torch_tensor = torch_tensor.bfloat16()
                else:
                    torch_tensor = torch_tensor.half()

            if name in tm_params:
                try:
                    import _turbomind as _tm
                except ImportError:
                    _tm = None
                failed = 0
                for tm_tensor in tm_params[name]:
                    try:
                        tensor_for_copy = torch_tensor
                        if _tm is not None:
                            if tm_tensor.type == _tm.DataType.TYPE_FP32 and torch_tensor.dtype in [torch.float16, torch.bfloat16]:
                                tensor_for_copy = torch_tensor.float()
                            elif tm_tensor.type == _tm.DataType.TYPE_FP16 and torch_tensor.dtype == torch.float32:
                                tensor_for_copy = torch_tensor.half()
                        tm_tensor.copy_from(tensor_for_copy.cuda())
                    except Exception as exc:
                        failed += 1
                        logger.warning(f"[cpu_realtime] export failed ({name}): {exc}")

                if failed:
                    logger.warning(f"[cpu_realtime] {failed}/{len(tm_params[name])} copies failed for {name}")

                tm_params.pop(name)
            else:
                logger.warning(f"[cpu_realtime] weight not in tm_params: {name}")

        _pol.to_cuda                   = _to_cpu
        _pol.process_awq_gemm          = _awq_cpu
        _pol.process_gptq              = _gptq_cpu
        _pol.process_mxfp4             = _mxfp4_cpu
        _pol.process_fp8               = _fp8_cpu
        _pol.process_compressed_tensor = _compressed_cpu
        _base_mod.BaseOutputModel.export_weight = _cpu_export_weight
        return self

    def __exit__(self, *_):
        for name, fn in self._saved.items():
            setattr(self._pol, name, fn)
        self._base_mod.BaseOutputModel.export_weight = self._orig_export


# ---------------------------------------------------------------------------
# CPU cast patch for the to-file export path  (convert_model_to_turbomind)
# ---------------------------------------------------------------------------

class _cpu_cast_export_weight:
    """
    Patches export_weight's to_file path to cast tensors to target dtype on
    CPU before handing them to _tofile(). Prevents the param.to(torch_type)
    call inside export_weight from allocating a second CUDA buffer alongside
    an already-GPU-resident source tensor.
    Only active during convert_model_to_turbomind (to_file=True).
    """

    def __enter__(self):
        import lmdeploy.turbomind.deploy.target_model.base as _base_mod
        from lmdeploy.turbomind.deploy.target_model.base import _weight_dtype_map
        self._base_mod = _base_mod
        self._orig = _base_mod.BaseOutputModel.export_weight
        _orig = self._orig

        def _patched(self_model, param, name):
            if self_model.to_file and torch.is_floating_point(param):
                dtype = _weight_dtype_map(self_model.model_config.weight_type, torch.float16)
                if param.dtype != dtype or param.is_cuda:
                    param = param.cpu().to(dtype)
            _orig(self_model, param, name)

        _base_mod.BaseOutputModel.export_weight = _patched
        return self

    def __exit__(self, *_):
        self._base_mod.BaseOutputModel.export_weight = self._orig


# ---------------------------------------------------------------------------
# Workspace detection / cache helpers
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


def check_turbomind_workspace(model_path: str) -> bool:
    return osp.exists(osp.join(model_path, 'config.yaml'))


# ---------------------------------------------------------------------------
# Model conversion  (HF → TurboMind flat workspace on disk)
# ---------------------------------------------------------------------------

def convert_model_to_turbomind(model_path: str, output_dir: str, model_format: str, tp: int = 1):
    from lmdeploy.turbomind.deploy.converter import get_tm_model

    os.makedirs(output_dir, exist_ok=True)

    engine_config = TurbomindEngineConfig(tp=tp, model_format=model_format)

    try:
        logger.info(f"Converting {model_path} to TurboMind format (tp={tp})...")
        tm_model = get_tm_model(
            model_path=model_path,
            model_name=None,
            chat_template_name=None,
            engine_config=engine_config,
            out_dir=output_dir,
        )

        import lmdeploy.turbomind.deploy.target_model.base as _base_mod
        _orig_save_split = _base_mod.BaseOutputModel.save_split

        def _tp_aware_save_split(self_model, tensor, name, split_dim=None, split_num=1, copy=False):
            # Ensure per-rank files are always written (name.0, name.1, ...)
            # even for tp=1, so the workspace loader always finds rank-indexed files.
            effective = max(split_num, tp) if split_dim is not None else split_num
            _orig_save_split(self_model, tensor, name, split_dim, effective, copy)

        _base_mod.BaseOutputModel.save_split = _tp_aware_save_split
        try:
            cpu_cast = getattr(shared.args, 'cpu_cast_export', False)
            ctx = _cpu_cast_export_weight() if cpu_cast else contextlib.nullcontext()
            with ctx:
                logger.info("Exporting TurboMind model to disk...")
                tm_model.export()
        finally:
            _base_mod.BaseOutputModel.save_split = _orig_save_split

        Path(osp.join(output_dir, '.source_model')).write_text(model_path)

        config_yaml_path = osp.join(output_dir, 'config.yaml')
        if osp.exists(config_yaml_path):
            with open(config_yaml_path) as f:
                tm_cfg = yaml.safe_load(f)
            model_arch = tm_cfg.get('model_config', {}).get('model_arch', '')
            if model_arch:
                with open(osp.join(output_dir, 'config.json'), 'w') as f:
                    json.dump({'architectures': [model_arch]}, f)

        _TOKENIZER_FILES = {
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
            'vocab.json', 'merges.txt', 'tokenizer.model', 'added_tokens.json', 'spm.model',
        }
        import shutil
        for fname in _TOKENIZER_FILES:
            src = osp.join(model_path, fname)
            if osp.exists(src):
                shutil.copy2(src, osp.join(output_dir, fname))

        logger.info(f"Conversion complete. Saved to {output_dir}")
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
        model_path_str = str(path_to_model)

        backend = shared.args.backend or 'turbomind'
        tp = shared.args.tensor_parallel or 1
        cache_type = (shared.args.cache_type or 'fp16').lower()
        ctx_size = getattr(shared.args, 'ctx_size', 4096) or 4096
        max_batch_size = getattr(shared.args, 'max_batch_size', 1) or 1
        cache_max_entry_count = getattr(shared.args, 'cache_max_entry_count', None) or 0.05
        cpu_realtime = getattr(shared.args, 'cpu_realtime_conversion', True)
        extra_flags = _parse_extra_flags(getattr(shared.args, 'extra_flags', None))

        quant_policy = {'q8': 8, 'q4': 4}.get(cache_type, 0)

        gc.collect()
        torch.cuda.empty_cache()

        is_workspace = check_turbomind_workspace(model_path_str)
        if is_workspace:
            ws_cfg_path = osp.join(model_path_str, 'config.yaml')
            with open(ws_cfg_path) as f:
                ws_cfg = yaml.safe_load(f)
            if 'linear_attention' in ws_cfg.get('model_config', {}).get('layer_types', []):
                _floor = 0.05
                if cache_max_entry_count < _floor:
                    logger.warning(
                        f"Hybrid (linear_attention) model: raising cache_max_entry_count "
                        f"from {cache_max_entry_count} to {_floor}"
                    )
                    cache_max_entry_count = _floor

        try:
            if backend == 'turbomind':
                tm_kwargs = dict(
                    tp=tp,
                    session_len=ctx_size,
                    max_batch_size=max_batch_size,
                    quant_policy=quant_policy,
                    cache_max_entry_count=cache_max_entry_count,
                    max_prefill_token_num=256,
                    enable_prefix_caching=True,
                )
                tm_kwargs.update(extra_flags)
                engine_config = TurbomindEngineConfig(**tm_kwargs)

                if is_workspace:
                    logger.info(f"Loading pre-converted TurboMind workspace: {model_path_str}")
                    with _workspace_load_hook(model_path_str):
                        pipeline = Pipeline(
                            model_path_str,
                            backend_config=engine_config,
                            log_level='INFO',
                        )
                else:
                    if cpu_realtime:
                        logger.info("CPU realtime conversion enabled - no VRAM peak during weight export")
                    load_ctx = _cpu_realtime_conversion() if cpu_realtime else contextlib.nullcontext()
                    with load_ctx:
                        pipeline = Pipeline(
                            str(path_to_model),
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
                    str(path_to_model),
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
        gen_config = self._prepare_generation_config(state)

        max_new_tokens = state['max_new_tokens']
        if state.get('auto_max_new_tokens'):
            max_new_tokens = state['truncation_length'] - self._last_prompt_token_count
        gen_config.max_new_tokens = max_new_tokens

        input_ids = self.encode(prompt, add_bos=False)
        self._last_prompt_token_count = len(input_ids)

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