import json
import os
import pprint
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import requests
from transformers import AutoTokenizer

from modules import shared
from modules.logging_colors import logger


class vLLMServer:
    def __init__(self, model_name: str, server_port: Optional[int] = None):
        self.model_name = model_name
        self.model_path = Path(shared.args.model_dir) / model_name
        self.port = server_port or self._find_available_port()
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.session = requests.Session()
        self.vocabulary_size = None
        self.n_ctx = shared.args.ctx_size if shared.args.ctx_size > 0 else 8192
        self.bos_token = "<s>"
        self.bos_token_id = None
        self.eos_token_id = None
        self.last_prompt_token_count = 0

        self.model_id = str(self.model_path)

        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
            logger.info(f"Loaded local tokenizer from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load local tokenizer: {e}")

        self._start_server()
        self._wait_for_server()
        self._load_tokenizer_info()
    
    def _is_port_available(self, port):
        """Check if a port is available for use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return True
            except OSError:
                return False

    def _find_available_port(self):
        """Find an available port, preferring main port + 5."""
        preferred_port = shared.args.api_port + 5
        if self._is_port_available(preferred_port):
            return preferred_port

        # Fall back to OS-assigned random port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _get_engine_args(self) -> List[str]:
        """Build command-line arguments for 1Cat-vLLM engine."""
        tensor_parallel_size = getattr(shared.args, 'tensor_parallel_size', 1)
        gpu_memory_utilization = getattr(shared.args, 'gpu_memory_utilization', 0.85)
        max_num_seqs = getattr(shared.args, 'max_num_seqs', 1)
        max_num_batched_tokens = getattr(shared.args, 'max_num_batched_tokens', 4096)
        ctx_size = shared.args.ctx_size if shared.args.ctx_size > 0 else 8192
        gpu_devices = getattr(shared.args, 'gpu_devices', None)
        enforce_eager = getattr(shared.args, 'enforce_eager', False)
        extra_flags = getattr(shared.args, 'extra_flags', '')
        attention_backend = getattr(shared.args, 'attention_backend', 'auto')
        language_model_only = getattr(shared.args, 'language_model_only', False)
        kv_cache_dtype = getattr(shared.args, 'kv_cache_dtype', 'auto')
        block_size = getattr(shared.args, 'block_size', 16)
        cpu_offload_gb = getattr(shared.args, 'cpu_offload_gb', 0)
        enable_prefix_caching = getattr(shared.args, 'enable_prefix_caching', True)
        optimization_level = getattr(shared.args, 'optimization_level', 2)
        limit_mm_per_prompt = getattr(shared.args, 'limit_mm_per_prompt', '{"image":0,"video":0}')
        compilation_config = getattr(shared.args, 'compilation_config', '')
        speculative_config = getattr(shared.args, 'speculative_config', '')

        # Detect SM70 (Volta) and set SM70-specific flags
        capability = self._get_device_capability()
        is_sm70 = capability is not None and capability[0] == 7

        args = [
            "vllm",
            "serve",
            str(self.model_path),
            "--trust-remote-code",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-seqs", str(max_num_seqs),
            "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--max-model-len", str(ctx_size),
            "--disable-custom-all-reduce"
        ]

        # Tier 1 params
        if language_model_only:
            args.append("--language-model-only")

        if kv_cache_dtype != 'auto':
            args.extend(["--kv-cache-dtype", kv_cache_dtype])

        args.extend(["--block-size", str(block_size)])
        args.extend(["--dtype", "half"])

        # Mamba/linear-attention cache mode — only for hybrid models
        mamba_cache_mode = getattr(shared.args, 'mamba_cache_mode', 'none')
        if mamba_cache_mode != 'none':
            args.extend(["--mamba-cache-mode", mamba_cache_mode])
            args.extend(["--mamba-ssm-cache-dtype", "float32"])

        if cpu_offload_gb > 0:
            args.extend(["--cpu-offload-gb", str(cpu_offload_gb)])

        if enable_prefix_caching:
            args.append("--enable-prefix-caching")

        if optimization_level != 2:
            args.extend(["--optimization-level", str(optimization_level)])

        if speculative_config:
            args.append("--enforce-eager")
            args.extend(["--speculative-config", speculative_config])

        if enforce_eager:
            args.append("--enforce-eager")

        # Attention backend: user choice overrides SM70 default
        if attention_backend != 'auto':
            args.extend(["--attention-backend", attention_backend])

        # SM70-specific flags (respects user overrides above)
        if is_sm70:
            if attention_backend == 'auto':
                args.extend(["--attention-backend", "triton_attn"])
            args.extend(["--skip-mm-profiling"])
            args.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])
            if compilation_config:
                args.extend(["--compilation-config", compilation_config])
            # elif optimization_level > 0:
            #     args.extend(["--compilation-config", '{"mode":"none","cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'])
        # Note: don't add --quantization, let vLLM auto-detect

        # Use full path to vllm if available
        import shutil
        vllm_path = shutil.which("vllm")
        if vllm_path:
            args[0] = vllm_path

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

        if extra_flags:
            args.extend(extra_flags.split())

        return [arg for arg in args if arg]

    def _get_device_capability(self):
        """Get CUDA device capability, returns (major, minor) or None."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_capability(0)
        except Exception:
            pass
        return None

    def _start_server(self):
        """Start the 1Cat-vLLM server as a subprocess."""
        cmd = self._get_engine_args()
        logger.info(f"Starting 1Cat-vLLM server: {' '.join(cmd)}")

        env = os.environ.copy()
        custom_env = {}

        # Add PyTorch library path for 1Cat-vLLM
        import torch
        torch_lib = str(Path(torch.__file__).parent / "lib")
        if 'LD_LIBRARY_PATH' in env:
            custom_env['LD_LIBRARY_PATH'] = f"{torch_lib}:{env['LD_LIBRARY_PATH']}"
        else:
            custom_env['LD_LIBRARY_PATH'] = torch_lib

        custom_env.update({
            'PYTHONUNBUFFERED': '1',
            'TRANSFORMERS_VERBOSITY': 'error',
            'VLLM_WORKER_MULTIPROC_METHOD': 'fork',
            #'VLLM_SM70_LM_HEAD_TOP1': '0',
            'VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND': '1',
            'VLLM_SERVER_DEV_MODE': '1',
            'PYTORCH_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128'
        })
        
        env.update(custom_env)
        logger.info(f"Using custom env: {custom_env}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        threading.Thread(target=self._read_stderr, daemon=True).start()
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self):
        """Read stdout and log messages (vLLM engine stats go here)."""
        if self.process and self.process.stdout:
            for line in self.process.stdout:
                if line.strip():
                    logger.info(f"1Cat-vLLM: {line.strip()}")

    def _read_stderr(self):
        """Read stderr and log messages."""
        if self.process and self.process.stderr:
            for line in self.process.stderr:
                if line.strip():
                    if "ERROR" in line or "Traceback" in line:
                        logger.error(f"1Cat-vLLM: {line.strip()}")
                    elif "WARN" in line:
                        logger.warning(f"1Cat-vLLM: {line.strip()}")
                    else:
                        logger.info(f"1Cat-vLLM: {line.strip()}")

    def _wait_for_server(self, timeout: int = 1200):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data"):
                        self.model_id = data["data"][0]["id"]
                    logger.info(f"1Cat-vLLM server ready at {self.base_url} (model_id={self.model_id!r})")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        raise RuntimeError(f"1Cat-vLLM server failed to start within {timeout}s")

    def _load_tokenizer_info(self):
        """Fetch tokenizer info from the server."""
        if self.tokenizer is not None:
            try:
                self.bos_token_id = self.tokenizer.bos_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.bos_token = self.tokenizer.bos_token
                logger.info(f"Loaded tokenizer info from local tokenizer: bos={self.bos_token_id}, eos={self.eos_token_id}")
                return
            except Exception as e:
                logger.warning(f"Failed to get tokenizer info from local tokenizer: {e}")

        # 1Cat-vLLM exposes /tokenizers/info when --enable-tokenizer-info-endpoint is set
        try:
            response = self.session.get(f"{self.base_url}/tokenizers/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                bos = data.get("bos_token_id") or data.get("bos_token")
                eos = data.get("eos_token_id") or data.get("eos_token")
                if bos is not None:
                    self.bos_token_id = int(bos) if not isinstance(bos, int) else bos
                if eos is not None:
                    self.eos_token_id = int(eos) if not isinstance(eos, int) else eos
                if isinstance(bos, str):
                    self.bos_token = bos
                logger.info(f"1Cat-vLLM tokenizer info: bos={self.bos_token_id}, eos={self.eos_token_id}")
                return
        except Exception:
            pass

        # Fallback: try /v1/models settings
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    model_data = data["data"][0]
                    settings = model_data.get("settings", {})
                    bos = settings.get("bos_token_id")
                    eos = settings.get("eos_token_id")
                    if bos is not None:
                        self.bos_token_id = int(bos) if not isinstance(bos, int) else bos
                    if eos is not None:
                        self.eos_token_id = int(eos) if not isinstance(eos, int) else eos
        except Exception:
            pass

        if self.bos_token_id is None:
            self.bos_token_id = 1
        if self.eos_token_id is None:
            self.eos_token_id = 151643
        logger.warning(f"Using default tokenizer info: bos={self.bos_token_id}, eos={self.eos_token_id}")

    def encode(self, text: str, add_bos_token: bool = False, **kwargs) -> List[int]:
        if self.tokenizer is not None:
            try:
                inputs = self.tokenizer(text, add_special_tokens=add_bos_token, return_tensors="pt")
                token_ids = inputs["input_ids"].tolist()[0]
                # logger.debug(f"Encoded {text[:50]}... to {len(token_ids)} tokens: {token_ids[:20]}...")
                return token_ids
            except Exception as e:
                logger.warning(f"Local tokenizer encode failed: {e}, falling back to API")

        url = f"{self.base_url}/v1/tokenize"
        payload = {
            "model": self.model_id,
            "prompt": text,
            "add_special_tokens": add_bos_token,
        }

        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result.get("tokens", result.get("token_ids",[]))
        except Exception as e:
            logger.error(f"1Cat-vLLM encode error: {e}")
            return [0] * max(1, int(len(text) / 3.5))

    def decode(self, token_ids: List[int], **kwargs) -> str:
        if self.tokenizer is not None:
            try:
                text = self.tokenizer.decode(token_ids)
                logger.info(f"Decoded {token_ids[:20]}... to: {text[:100]}...")
                return text
            except Exception as e:
                logger.warning(f"Local tokenizer decode failed: {e}, falling back to API")

        url = f"{self.base_url}/v1/detokenize"
        payload = {
            "model": self.model_id,
            "tokens": token_ids,
        }

        try:
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result.get("prompt", result.get("content", ""))
        except Exception as e:
            logger.error(f"1Cat-vLLM decode error: {e}")
            return ""

    def convert_ids_to_tokens(self, ids, **kwargs):
        """Convert token IDs to tokens."""
        if isinstance(ids, int):
            return self.decode([ids], **kwargs).strip()
        return [self.decode([id], **kwargs).strip() for id in ids]

    def prepare_payload(self, state: dict, prompt_length: int) -> dict:
        """Prepare the request payload from generation state."""
        payload = {
            "model": self.model_id,
            "temperature": state.get("temperature", 1.0),
            "top_p": state.get("top_p", 1.0),
            "top_k": state.get("top_k", -1),
            "min_p": state.get("min_p", 0.0),
            "repetition_penalty": state.get("repetition_penalty", 1.0),
            "presence_penalty": state.get("presence_penalty", 0.0),
            "frequency_penalty": state.get("frequency_penalty", 0.0),
            "stop": [],
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }

        seed = state.get("seed", -1)
        if seed != -1:
            payload["seed"] = seed

        if state.get("no_repeat_ngram_size", 0) > 0:
            payload["no_repeat_ngram_size"] = state.get("no_repeat_ngram_size")

        if state.get("ban_eos_token"):
            payload["ignore_eos"] = True

        # Handle max_tokens correctly for vLLM's strict requirements
        max_new_tokens = state.get("max_new_tokens", 256)
        auto_max_new_tokens = state.get("auto_max_new_tokens", False)

        safety_buffer = 8
        max_possible = self.n_ctx - prompt_length - safety_buffer

        if auto_max_new_tokens:
            payload["max_tokens"] = max(max_possible, 1)
        else:
            if max_possible > 0:
                payload["max_tokens"] = min(max_new_tokens, max_possible)
            else:
                payload["max_tokens"] = 1

        return payload

    def generate_with_streaming(self, prompt: str, state: dict, **kwargs):
        """Generate text with streaming - uses /v1/completions API.

        The socket read runs on a background pump thread; the generator loop
        polls the line queue with a short timeout so it can react to a cancel
        (shared.stop_everything / stop_event) within ~0.1s instead of blocking
        for the whole prefill until the first token arrives."""
        url = f"{self.base_url}/v1/completions"

        prompt_token_count = len(self.encode(prompt))
        self.last_prompt_token_count = prompt_token_count

        logger.info(f"1Cat-vLLM request: {prompt_token_count} tokens")

        payload = self.prepare_payload(state, prompt_token_count)
        payload["prompt"] = prompt
        payload["stream"] = True

        if shared.args.verbose:
            logger.info("1CAT_VLLM_PARAMS=")
            pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(payload)
            print()

        logger.debug(f"1Cat-vLLM request: POST {url}")
        response = self.session.post(url, json=payload, stream=True)
        logger.debug(f"1Cat-vLLM request: POST complete")

        if response.status_code == 400:
            logger.error(f"1Cat-vLLM completions error: {response.json()}")
            response.close()
            return
        response.raise_for_status()

        q = queue.Queue()
        sentinel = object()

        def pump():
            try:
                for line in response.iter_lines():
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(sentinel)

        threading.Thread(target=pump, daemon=True).start()

        full_text = ""
        try:
            while True:
                # Short poll: lets a cancel land during a silent prefill
                if shared.stop_everything:
                    break
                stop_event = state.get("stop_event")
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    line = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is sentinel:
                    break
                if not line:
                    continue
                try:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    data = json.loads(line)
                    if data.get("choices"):
                        content = data["choices"][0].get("text", "")
                        if content:
                            full_text += content
                            yield full_text
                except json.JSONDecodeError:
                    continue
        finally:
            # Offload to a daemon thread: response.close() blocks until the
            # pump thread's socket read ends, which can be the whole prefill.
            threading.Thread(target=response.close, daemon=True).start()

    def generate(self, prompt: str, state: dict, **kwargs) -> str:
        """Generate without streaming."""
        output = ""
        for output in self.generate_with_streaming(prompt, state, **kwargs):
            pass
        return output

    def unload(self):
        """Stop the server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("1Cat-vLLM server stopped")

    def stop(self):
        """Alias for unload for compatibility."""
        self.unload()