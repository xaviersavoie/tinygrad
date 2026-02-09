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

# ** C source generation — y[m] = sum_k W[m,k]*x[k], vectorized over K, M unrolled by 8
# each arch: (include, vec_type, simd_width, zero, load_f32, load_f16, fma(w,x,acc), hsum, helper_code)
_ARCHS = {
  'avx512': ('immintrin.h', '__m512', 16, '_mm512_setzero_ps()',
    lambda a: f'_mm512_loadu_ps({a})', lambda a: f'_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)({a})))',
    lambda w,x,c: f'_mm512_fmadd_ps({w},{x},{c})', lambda v: f'_mm512_reduce_add_ps({v})', ''),
  'avx2': ('immintrin.h', '__m256', 8, '_mm256_setzero_ps()',
    lambda a: f'_mm256_loadu_ps({a})', lambda a: f'_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)({a})))',
    lambda w,x,c: f'_mm256_fmadd_ps({w},{x},{c})', lambda v: f'hsum256({v})',
    'static inline float hsum256(__m256 v) {\n  __m128 hi = _mm256_extractf128_ps(v, 1);\n'
    '  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), hi);\n  lo = _mm_add_ps(lo, _mm_movehdup_ps(lo));\n'
    '  return _mm_cvtss_f32(_mm_add_ss(lo, _mm_movehl_ps(lo, lo)));\n}\n'),
  'neon': ('arm_neon.h', 'float32x4_t', 4, 'vdupq_n_f32(0)',
    lambda a: f'vld1q_f32({a})', lambda a: f'vcvt_f32_f16(vld1_f16({a}))',
    lambda w,x,c: f'vfmaq_f32({c},{w},{x})', lambda v: f'vaddvq_f32({v})', ''),
}

def _get_arch():
  m = platform.machine()
  if m in ('x86_64', 'AMD64'): return 'avx512' if _avx512 else 'avx2'
  if m in ('arm64', 'aarch64'): return 'neon'
  return None

def _gen_src(name: str, M: int, K: int, half: bool) -> str:
  inc, vec, w, zero, ld, ld16, fma, hsum, helper = _ARCHS[_get_arch()]
  wtype, ldw = ("__fp16", ld16) if half else ("float", ld)
  NL = '\n'
  return f"""\
#include <{inc}>
{helper}void {name}(float* restrict data0, {wtype}* restrict data1, float* restrict data2) {{
  for (int m = 0; m < {M}; m += 8) {{
    {vec} {', '.join(f'a{i}={zero}' for i in range(4))};
    {vec} {', '.join(f'a{i}={zero}' for i in range(4, 8))};
    for (int k = 0; k < {K}; k += {w}) {{
      {vec} xv = {ld('data2 + k')};
{NL.join(f'      a{i} = {fma(ldw(f"data1 + (m+{i})*{K} + k"), "xv", f"a{i}")};' for i in range(8))}
    }}
{NL.join(f'    data0[m+{i}] = {hsum(f"a{i}")};' for i in range(8))}
  }}
}}
"""

# ** PROGRAM UOp builder

_compile_cache: dict[str, bytes] = {}
_compiler = ClangJITCompiler()

def _custom_cpu_matvec(C: UOp, A: UOp, B: UOp, dname: str) -> UOp:
  """Build PROGRAM UOp. C=out(1,M) float, A=W(M,K) contiguous, B=x(K,) float."""
  M, K = A.shape
  half = A.dtype.itemsize == 2
  name = f"matvec_{M}_{K}_{'f16' if half else 'f32'}"
  src = _gen_src(name, M, K, half)
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
  arch = _get_arch()
  if arch is None: return False
  K, M = b.shape
  if K % _ARCHS[arch][2] != 0 or M % 8 != 0: return False
  return True

def cpu_matvec(a: Tensor, b: Tensor) -> Tensor:
  """a=(...,K) @ b=(K,M) where b is PERMUTE view of contiguous (M,K) weight."""
  K, M = b.shape
  w_orig = Tensor(b.uop.src[0], device=b.device)
  out = Tensor.empty(1, M, dtype=dtypes.float, device=a.device)
  out = Tensor.custom_kernel(out, w_orig, a.reshape(K), fxn=functools.partial(_custom_cpu_matvec, dname=a.device))[0]
  return out.reshape(*a.shape[:-1], M)
