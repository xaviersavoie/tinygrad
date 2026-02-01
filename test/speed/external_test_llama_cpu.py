#!/usr/bin/env python3
"""Llama 1B CPU decode speed test: tinygrad vs HuggingFace Transformers.

Single-core execution enforced for fair comparison.
"""
import os
# Single-core execution - must be set before imports
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["CPU_COUNT"] = "1"  # tinygrad: limit to 1 core
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable CUDA for torch

import unittest
import time
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)  # inter-op parallelism

from transformers import LlamaConfig, LlamaForCausalLM
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv, colorize_float
from tinygrad.apps.llm import Transformer
from tinygrad.uop.ops import UOp

# Verify single-threaded execution
assert torch.get_num_threads() == 1, f"torch using {torch.get_num_threads()} threads, expected 1"
assert torch.get_num_interop_threads() == 1, f"torch using {torch.get_num_interop_threads()} interop threads, expected 1"

class TestLlamaCPU(unittest.TestCase):
  def test_llama_1b_decode(self):
    # Llama 3.2 1B config
    DIM, HIDDEN, HEADS, N_KV_HEADS, LAYERS = 2048, 8192, 32, 8, 16
    VOCAB_SIZE, MAX_CONTEXT = 128256, 512

    # --- HuggingFace model (random weights, no download) ---
    config = LlamaConfig(
      hidden_size=DIM,
      intermediate_size=HIDDEN,
      num_attention_heads=HEADS,
      num_key_value_heads=N_KV_HEADS,
      num_hidden_layers=LAYERS,
      vocab_size=VOCAB_SIZE,
      max_position_embeddings=MAX_CONTEXT,
      rms_norm_eps=1e-5,
      rope_theta=500000,
      use_cache=True,
    )
    torch_model = LlamaForCausalLM(config).eval()

    torch_model = torch.compile(torch_model, backend="inductor", mode="max-autotune-no-cudagraphs")


    # --- Tinygrad model ---
    HEAD_DIM = DIM // HEADS
    tiny_model = Transformer(num_blocks=LAYERS, dim=DIM, hidden_dim=HIDDEN, n_heads=HEADS,
                             n_kv_heads=N_KV_HEADS, norm_eps=1e-5, vocab_size=VOCAB_SIZE,
                             head_dim=HEAD_DIM, rope_theta=500000, max_context=MAX_CONTEXT)
    v_start_pos = UOp.variable("start_pos", 1, MAX_CONTEXT-1)

    # Warmup: prefill + decode for HuggingFace
    with torch.no_grad():
      prefill_ids = torch.tensor([[1, 2, 3, 4, 5]])
      outputs = torch_model(prefill_ids, use_cache=True)
      past_kv = outputs.past_key_values
      for i in range(3):
        outputs = torch_model(torch.tensor([[1]]), past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values

    # Warmup: prefill + decode for tinygrad
    tiny_model(Tensor([[1, 2, 3, 4, 5]]), start_pos=0)
    Device[Device.DEFAULT].synchronize()
    for i in range(3):
      tiny_model(Tensor([[1]]), start_pos=v_start_pos.bind(5+i))
      Device[Device.DEFAULT].synchronize()

    # Benchmark decode
    N = getenv("CNT", 8)

    # Reset KV cache for benchmark
    with torch.no_grad():
      outputs = torch_model(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), use_cache=True)
      past_kv = outputs.past_key_values

    torch_times = []
    for i in range(N):
      with torch.no_grad():
        st = time.perf_counter()
        outputs = torch_model(torch.tensor([[1]]), past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        torch_times.append(time.perf_counter() - st)

    # Reset KV cache for tinygrad benchmark (same prefill length as torch)
    tiny_model(Tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), start_pos=0)
    Device[Device.DEFAULT].synchronize()

    tiny_times = []
    for i in range(N):
      Device[Device.DEFAULT].synchronize()
      st = time.perf_counter()
      tiny_model(Tensor([[1]]), start_pos=v_start_pos.bind(8+i))
      Device[Device.DEFAULT].synchronize()
      tiny_times.append(time.perf_counter() - st)

    et_torch, et_tiny = min(torch_times) * 1000, min(tiny_times) * 1000
    print(f"\nllama 1B decode: torch {et_torch:.2f}ms ({1000/et_torch:.2f} tok/s), tinygrad {et_tiny:.2f}ms ({1000/et_tiny:.2f} tok/s), " +
          f"{colorize_float(et_tiny/et_torch)} {'faster' if et_torch > et_tiny else 'slower'}")
    # Assert tinygrad is at least as fast as torch (with 5% tolerance for variance)
    assert et_tiny <= et_torch * 1.05, f"tinygrad {et_tiny:.2f}ms slower than torch {et_torch:.2f}ms"

if __name__ == '__main__':
  unittest.main()
