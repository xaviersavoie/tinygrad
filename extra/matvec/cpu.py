import functools, math, platform, subprocess
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler

# ** detect AVX-512 at import time (x86 only)
def _has_avx512() -> bool:
  if platform.machine() not in ('x86_64', 'AMD64'): return False
  try: return b'avx512f' in subprocess.check_output(['grep', '-m1', 'flags', '/proc/cpuinfo'])
  except Exception: return False
_avx512 = _has_avx512()

# ** C source generation — row formulation for (M, K) weight layout
# y[m] = sum_k W[m,k] * x[k], vectorized over K, unrolled M by 8

def _src_x86_avx512(name: str, M: int, K: int, half: bool) -> str:
  wtype = "__fp16" if half else "float"
  load_w = lambda off: f"_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(data1 + {off})))" if half else f"_mm512_loadu_ps(data1 + {off})"
  return f"""\
#include <immintrin.h>
void {name}(float* restrict data0, {wtype}* restrict data1, float* restrict data2) {{
  for (int m = 0; m < {M}; m += 8) {{
    __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps(), a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
    __m512 a4=_mm512_setzero_ps(), a5=_mm512_setzero_ps(), a6=_mm512_setzero_ps(), a7=_mm512_setzero_ps();
    for (int k = 0; k < {K}; k += 16) {{
      __m512 xv = _mm512_loadu_ps(data2 + k);
      a0 = _mm512_fmadd_ps({load_w(f"(m+0)*{K} + k")}, xv, a0);
      a1 = _mm512_fmadd_ps({load_w(f"(m+1)*{K} + k")}, xv, a1);
      a2 = _mm512_fmadd_ps({load_w(f"(m+2)*{K} + k")}, xv, a2);
      a3 = _mm512_fmadd_ps({load_w(f"(m+3)*{K} + k")}, xv, a3);
      a4 = _mm512_fmadd_ps({load_w(f"(m+4)*{K} + k")}, xv, a4);
      a5 = _mm512_fmadd_ps({load_w(f"(m+5)*{K} + k")}, xv, a5);
      a6 = _mm512_fmadd_ps({load_w(f"(m+6)*{K} + k")}, xv, a6);
      a7 = _mm512_fmadd_ps({load_w(f"(m+7)*{K} + k")}, xv, a7);
    }}
    data0[m+0] = _mm512_reduce_add_ps(a0); data0[m+1] = _mm512_reduce_add_ps(a1);
    data0[m+2] = _mm512_reduce_add_ps(a2); data0[m+3] = _mm512_reduce_add_ps(a3);
    data0[m+4] = _mm512_reduce_add_ps(a4); data0[m+5] = _mm512_reduce_add_ps(a5);
    data0[m+6] = _mm512_reduce_add_ps(a6); data0[m+7] = _mm512_reduce_add_ps(a7);
  }}
}}
"""

def _src_x86_avx2(name: str, M: int, K: int, half: bool) -> str:
  wtype = "__fp16" if half else "float"
  load_w = lambda off: f"_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(data1 + {off})))" if half else f"_mm256_loadu_ps(data1 + {off})"
  return f"""\
#include <immintrin.h>
static inline float hsum256(__m256 v) {{
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), hi);
  lo = _mm_add_ps(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(_mm_add_ss(lo, _mm_movehl_ps(lo, lo)));
}}
void {name}(float* restrict data0, {wtype}* restrict data1, float* restrict data2) {{
  for (int m = 0; m < {M}; m += 8) {{
    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
    __m256 a4=_mm256_setzero_ps(), a5=_mm256_setzero_ps(), a6=_mm256_setzero_ps(), a7=_mm256_setzero_ps();
    for (int k = 0; k < {K}; k += 8) {{
      __m256 xv = _mm256_loadu_ps(data2 + k);
      a0 = _mm256_fmadd_ps({load_w(f"(m+0)*{K} + k")}, xv, a0);
      a1 = _mm256_fmadd_ps({load_w(f"(m+1)*{K} + k")}, xv, a1);
      a2 = _mm256_fmadd_ps({load_w(f"(m+2)*{K} + k")}, xv, a2);
      a3 = _mm256_fmadd_ps({load_w(f"(m+3)*{K} + k")}, xv, a3);
      a4 = _mm256_fmadd_ps({load_w(f"(m+4)*{K} + k")}, xv, a4);
      a5 = _mm256_fmadd_ps({load_w(f"(m+5)*{K} + k")}, xv, a5);
      a6 = _mm256_fmadd_ps({load_w(f"(m+6)*{K} + k")}, xv, a6);
      a7 = _mm256_fmadd_ps({load_w(f"(m+7)*{K} + k")}, xv, a7);
    }}
    data0[m+0] = hsum256(a0); data0[m+1] = hsum256(a1);
    data0[m+2] = hsum256(a2); data0[m+3] = hsum256(a3);
    data0[m+4] = hsum256(a4); data0[m+5] = hsum256(a5);
    data0[m+6] = hsum256(a6); data0[m+7] = hsum256(a7);
  }}
}}
"""

def _src_arm(name: str, M: int, K: int, half: bool) -> str:
  wtype = "__fp16" if half else "float"
  load_w = lambda off: f"vcvt_f32_f16(vld1_f16(data1 + {off}))" if half else f"vld1q_f32(data1 + {off})"
  return f"""\
#include <arm_neon.h>
void {name}(float* restrict data0, {wtype}* restrict data1, float* restrict data2) {{
  for (int m = 0; m < {M}; m += 8) {{
    float32x4_t a0=vdupq_n_f32(0), a1=vdupq_n_f32(0), a2=vdupq_n_f32(0), a3=vdupq_n_f32(0);
    float32x4_t a4=vdupq_n_f32(0), a5=vdupq_n_f32(0), a6=vdupq_n_f32(0), a7=vdupq_n_f32(0);
    for (int k = 0; k < {K}; k += 4) {{
      float32x4_t xv = vld1q_f32(data2 + k);
      a0 = vfmaq_f32(a0, {load_w(f"(m+0)*{K} + k")}, xv);
      a1 = vfmaq_f32(a1, {load_w(f"(m+1)*{K} + k")}, xv);
      a2 = vfmaq_f32(a2, {load_w(f"(m+2)*{K} + k")}, xv);
      a3 = vfmaq_f32(a3, {load_w(f"(m+3)*{K} + k")}, xv);
      a4 = vfmaq_f32(a4, {load_w(f"(m+4)*{K} + k")}, xv);
      a5 = vfmaq_f32(a5, {load_w(f"(m+5)*{K} + k")}, xv);
      a6 = vfmaq_f32(a6, {load_w(f"(m+6)*{K} + k")}, xv);
      a7 = vfmaq_f32(a7, {load_w(f"(m+7)*{K} + k")}, xv);
    }}
    data0[m+0] = vaddvq_f32(a0); data0[m+1] = vaddvq_f32(a1);
    data0[m+2] = vaddvq_f32(a2); data0[m+3] = vaddvq_f32(a3);
    data0[m+4] = vaddvq_f32(a4); data0[m+5] = vaddvq_f32(a5);
    data0[m+6] = vaddvq_f32(a6); data0[m+7] = vaddvq_f32(a7);
  }}
}}
"""

def _get_src_fn():
  arch = platform.machine()
  if arch in ('x86_64', 'AMD64'): return _src_x86_avx512 if _avx512 else _src_x86_avx2
  if arch in ('arm64', 'aarch64'): return _src_arm
  return None

def _get_simd_k():
  arch = platform.machine()
  if arch in ('x86_64', 'AMD64'): return 16 if _avx512 else 8
  if arch in ('arm64', 'aarch64'): return 4
  return 0

# ** PROGRAM UOp builder

_compile_cache: dict[str, bytes] = {}
_compiler = ClangJITCompiler()

def _custom_cpu_matvec(C: UOp, A: UOp, B: UOp, dname: str) -> UOp:
  """Build PROGRAM UOp. C=out(1,M) float, A=W(M,K) contiguous, B=x(K,) float."""
  M, K = A.shape
  half = A.dtype.itemsize == 2
  name = f"matvec_{M}_{K}_{'f16' if half else 'f32'}"
  src = _get_src_fn()(name, M, K, half)
  if name not in _compile_cache: _compile_cache[name] = _compiler.compile_cached(src)
  binary = _compile_cache[name]
  wmem = M * K * (2 if half else 4)
  sink = UOp.sink(C.base, A.base, B.base, arg=KernelInfo(name=name, estimates=Estimates(ops=2*M*K, mem=wmem + K*4 + M*4)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                                UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

# ** user-facing API

def can_use_cpu_matvec(a: Tensor, b: Tensor) -> bool:
  """Called from dot(self=a, w=b). b is weight.T (K,M) PERMUTE view of contiguous (M,K) weight."""
  if a.device != "CPU" or b.device != "CPU": return False
  if a.ndim < 2 or b.ndim != 2: return False
  if math.prod(a.shape[:-1]) != 1: return False
  if b.dtype not in {dtypes.half, dtypes.float16, dtypes.float, dtypes.float32}: return False
  if b.uop.op is not Ops.PERMUTE: return False
  if _get_src_fn() is None: return False
  K, M = b.shape
  sk = _get_simd_k()
  if K % sk != 0 or M % 8 != 0: return False
  return True

def cpu_matvec(a: Tensor, b: Tensor) -> Tensor:
  """a=(...,K) @ b=(K,M) where b is PERMUTE view of contiguous (M,K) weight."""
  K, M = b.shape
  w_orig = Tensor(b.uop.src[0], device=b.device)
  out = Tensor.empty(1, M, dtype=dtypes.float, device=a.device)
  out = Tensor.custom_kernel(out, w_orig, a.reshape(K), fxn=functools.partial(_custom_cpu_matvec, dname=a.device))[0]
  return out.reshape(*a.shape[:-1], M)
