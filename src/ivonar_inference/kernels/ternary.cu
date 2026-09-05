#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#ifndef MAX_CHUNKS
#define MAX_CHUNKS 70
#endif
#ifndef MAX_TOPK
#define MAX_TOPK 256
#endif
#define THREADS 256
#define PREFETCH 4
#define NARROW_PER_THREAD 4
#define WIDE_PER_THREAD ((MAX_CHUNKS * 64 + THREADS - 1) / THREADS)
#define ATTN_THREADS 64
#define HALF_DIM (HEAD_DIM / 2)
#define PARTIAL_STRIDE (HEAD_DIM + 2)
#define RECURRENT_MAX_THREADS 1024
#define RECURRENT_MAX_STATE 512
#ifndef TILE
#define TILE 32
#endif
#define TILE_STATE 16
#define TILE_HEAD 128
#define TILE_TAPS 8
#define TILE_PHASE 4
#ifndef GEMM_MIN_BLOCKS
#define GEMM_MIN_BLOCKS 2
#endif
#ifndef TILE_GROUP_NARROW
#define TILE_GROUP_NARROW 8
#endif
#ifndef TILE_GROUP_WIDE
#define TILE_GROUP_WIDE 8
#endif
#define NARROW_CHUNKS (NARROW_PER_THREAD * THREADS / 64)
#define SAMPLE_THREADS 1024
#define SAMPLE_WARPS (SAMPLE_THREADS / 32)
#define SAMPLE_BATCH 4
#define SAMPLE_DELTAS 6
#define NEG_INF __int_as_float(0xff800000)

typedef unsigned short half_t;
typedef unsigned char u8;

enum { MODE_PLAIN = 0, MODE_NORM = 1, MODE_ATTN = 2 };
enum { EPI_STORE = 0, EPI_RESIDUAL = 1, EPI_SILU_PAIR = 2 };

__device__ __forceinline__ int dp4a(int a, int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ float h2f(half_t h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

__device__ __forceinline__ half_t f2h(float f) {
  half_t h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_max(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

__device__ __forceinline__ int warp_sum_int(int v) {
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

__device__ __forceinline__ float block_max(float v, float* slots) {
  v = warp_max(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  if (lane == 0) slots[warp] = v;
  __syncthreads();
  float peak = slots[0];
  for (int w = 1; w < warps; ++w) peak = fmaxf(peak, slots[w]);
  return peak;
}

__device__ __forceinline__ float block_sum(float v, float* slots) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  if (lane == 0) slots[warp] = v;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < warps; ++w) total += slots[w];
  return total;
}

__device__ __forceinline__ int q8_offset(int i) {
  const int u = i >> 6;
  const int quad = (i >> 4) & 3;
  const int slot = (quad + (u >> 1)) & 3;
  return u * 64 + slot * 16 + (i & 3) * 4 + ((i >> 2) & 3);
}

template <int MODE>
__device__ __forceinline__ float source_value(
    const float* __restrict__ input, const float* __restrict__ aux, int i, int n, int splits) {
  if (MODE == MODE_ATTN) {
    const int head = i / HEAD_DIM, dim = i - head * HEAD_DIM;
    const float* base = input + (size_t)head * splits * PARTIAL_STRIDE;
    float peak = NEG_INF;
    for (int s = 0; s < splits; ++s) peak = fmaxf(peak, base[s * PARTIAL_STRIDE]);
    float total = 0.0f, value = 0.0f;
    for (int s = 0; s < splits; ++s) {
      const float* part = base + s * PARTIAL_STRIDE;
      const float weight = part[0] == NEG_INF ? 0.0f : expf(part[0] - peak);
      total += weight * part[1];
      value += weight * part[2 + dim];
    }
    return value / total;
  }
  return input[i];
}

template <int MODE, int PER_THREAD>
__device__ __forceinline__ float quantize_source(
    const float* __restrict__ input, const float* __restrict__ aux, int n, float eps, int splits,
    signed char* q8, int* xsum, float* slots) {
  const int tid = threadIdx.x;
  const int chunks = (n + 63) >> 6;
  float values[PER_THREAD];
  float weights[PER_THREAD];
  float squares = 0.0f;
#pragma unroll
  for (int k = 0; k < PER_THREAD; ++k) {
    const int i = tid + k * THREADS;
    float v = 0.0f;
    if (i < n) {
      v = source_value<MODE>(input, aux, i, n, splits);
      if (MODE == MODE_NORM) weights[k] = aux[i];
    }
    values[k] = v;
    squares += v * v;
  }
  for (int i = n + tid; i < chunks * 64; i += THREADS) q8[q8_offset(i)] = 0;
  if (MODE == MODE_NORM) {
    squares = block_sum(squares, slots);
    const float inv_rms = 1.0f / sqrtf(squares / (float)n + eps);
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
      values[k] = fminf(fmaxf(values[k] * inv_rms, -8.0f), 8.0f) * weights[k];
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < PER_THREAD; ++k) amax = fmaxf(amax, fabsf(values[k]));
  amax = block_max(amax, slots + 32);
  const float scale = fmaxf(amax / 127.0f, 1e-5f);
#pragma unroll
  for (int k = 0; k < PER_THREAD; ++k) {
    const int i = tid + k * THREADS;
    if (i < n) q8[q8_offset(i)] = (signed char)(int)rintf(values[k] / scale);
  }
  __syncthreads();
  if (tid < chunks) {
    const int* words = (const int*)q8 + tid * 16;
    int total = 0;
#pragma unroll
    for (int w = 0; w < 16; ++w) total = dp4a(words[w], 0x01010101, total);
    xsum[tid] = total;
  }
  __syncthreads();
  return scale;
}

__device__ __forceinline__ int quad_dot(unsigned int w, const int4 a, int dot) {
  dot = dp4a(w & 0x03030303, a.x, dot);
  dot = dp4a((w >> 2) & 0x03030303, a.y, dot);
  dot = dp4a((w >> 4) & 0x03030303, a.z, dot);
  dot = dp4a((w >> 6) & 0x03030303, a.w, dot);
  return dot;
}

__device__ __forceinline__ float chunk_dot(
    const uint4 w, const signed char* __restrict__ chunk, int rotation, int chunk_sum, half_t scale) {
  const int4* quads = (const int4*)chunk;
  int dot = 0;
  dot = quad_dot(w.x, quads[rotation], dot);
  dot = quad_dot(w.y, quads[(rotation + 1) & 3], dot);
  dot = quad_dot(w.z, quads[(rotation + 2) & 3], dot);
  dot = quad_dot(w.w, quads[(rotation + 3) & 3], dot);
  return (float)(dot - chunk_sum) * h2f(scale);
}

__device__ __forceinline__ void unpack_word(unsigned int w, int* codes) {
  codes[0] = w & 0x03030303;
  codes[1] = (w >> 2) & 0x03030303;
  codes[2] = (w >> 4) & 0x03030303;
  codes[3] = (w >> 6) & 0x03030303;
}

__device__ __forceinline__ void unpack_chunk(const uint4 w, int* codes) {
  unpack_word(w.x, codes);
  unpack_word(w.y, codes + 4);
  unpack_word(w.z, codes + 8);
  unpack_word(w.w, codes + 12);
}

__device__ __forceinline__ int quad_dot_unpacked(const int* codes, const int4 a, int dot) {
  dot = dp4a(codes[0], a.x, dot);
  dot = dp4a(codes[1], a.y, dot);
  dot = dp4a(codes[2], a.z, dot);
  dot = dp4a(codes[3], a.w, dot);
  return dot;
}

__device__ __forceinline__ float chunk_dot_unpacked(
    const int* codes, const signed char* __restrict__ chunk, int rotation, int chunk_sum, float scale) {
  const int4* quads = (const int4*)chunk;
  int dot = 0;
  dot = quad_dot_unpacked(codes, quads[rotation], dot);
  dot = quad_dot_unpacked(codes + 4, quads[(rotation + 1) & 3], dot);
  dot = quad_dot_unpacked(codes + 8, quads[(rotation + 2) & 3], dot);
  dot = quad_dot_unpacked(codes + 12, quads[(rotation + 3) & 3], dot);
  return (float)(dot - chunk_sum) * scale;
}

#define GEMV_ARGS                                                                                        \
  const float* __restrict__ input, const float* __restrict__ aux, const signed char* __restrict__ q8_in, \
      const int* __restrict__ xsum_in, const float* __restrict__ scale_in, const u8* __restrict__ packed, \
      const half_t* __restrict__ scales, const float* __restrict__ bias, float* __restrict__ output,    \
      int rows, int n, int row_bytes, int groups, int lanes, int splits, float eps
#define GEMV_PASS input, aux, q8_in, xsum_in, scale_in, packed, scales, bias, output, rows, n, row_bytes, groups, lanes, splits, eps

template <int MODE, int EPI, int PER_THREAD>
__device__ __forceinline__ void gemv_body(GEMV_ARGS) {
  __shared__ __align__(16) signed char q8[MAX_CHUNKS * 64];
  __shared__ int xsum[MAX_CHUNKS];
  __shared__ float slots[64];
  const int tid = threadIdx.x;
  const int chunks = (n + 63) >> 6;
  const int row = blockIdx.x * (THREADS / lanes) + tid / lanes;
  const int sub = tid & (lanes - 1);
  const bool active = row < rows;
  const uint4* wrow = (const uint4*)(packed + (size_t)(active ? row : 0) * row_bytes);
  uint4 ahead[PREFETCH];
#pragma unroll
  for (int k = 0; k < PREFETCH; ++k) {
    const int u = sub + k * lanes;
    ahead[k] = (active && u < chunks) ? __ldg(wrow + u) : make_uint4(0u, 0u, 0u, 0u);
  }
  float scale;
  if (PER_THREAD == 0) {
    for (int i = tid; i < chunks * 16; i += THREADS) ((int*)q8)[i] = ((const int*)q8_in)[i];
    if (tid < chunks) xsum[tid] = xsum_in[tid];
    scale = *scale_in;
    __syncthreads();
  } else {
    scale = quantize_source<MODE, (PER_THREAD > 0 ? PER_THREAD : 1)>(input, aux, n, eps, splits, q8, xsum, slots);
  }
  float acc = 0.0f;
  if (active) {
    const half_t* srow = scales + (size_t)row * groups;
#pragma unroll
    for (int k = 0; k < PREFETCH; ++k) {
      const int u = sub + k * lanes;
      if (u < chunks) acc += chunk_dot(ahead[k], q8 + u * 64, (u >> 1) & 3, xsum[u], srow[u >> 1]);
    }
#pragma unroll 2
    for (int u = sub + PREFETCH * lanes; u < chunks; u += lanes) {
      acc += chunk_dot(__ldg(wrow + u), q8 + u * 64, (u >> 1) & 3, xsum[u], srow[u >> 1]);
    }
  }
  for (int offset = 1; offset < lanes; offset <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, offset);
  float result = acc * scale;
  if (active && bias) result += bias[row];
  if (EPI == EPI_SILU_PAIR) {
    const float partner = __shfl_down_sync(0xffffffffu, result, lanes);
    if (active && sub == 0 && (row & 1) == 0) output[row >> 1] = silu_f(result) * partner;
  } else if (active && sub == 0) {
    if (EPI == EPI_RESIDUAL) output[row] += result;
    else output[row] = result;
  }
}

extern "C" __global__ void __launch_bounds__(THREADS) gemv_plain_store(GEMV_ARGS) {
  gemv_body<MODE_PLAIN, EPI_STORE, NARROW_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_plain_residual(GEMV_ARGS) {
  gemv_body<MODE_PLAIN, EPI_RESIDUAL, NARROW_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_plain_residual_wide(GEMV_ARGS) {
  gemv_body<MODE_PLAIN, EPI_RESIDUAL, WIDE_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_norm_store(GEMV_ARGS) {
  gemv_body<MODE_NORM, EPI_STORE, NARROW_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_norm_silu_pair(GEMV_ARGS) {
  gemv_body<MODE_NORM, EPI_SILU_PAIR, NARROW_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_attn_residual(GEMV_ARGS) {
  gemv_body<MODE_ATTN, EPI_RESIDUAL, NARROW_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_prequant_store(GEMV_ARGS) {
  gemv_body<MODE_PLAIN, EPI_STORE, 0>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_plain_store_wide(GEMV_ARGS) {
  gemv_body<MODE_PLAIN, EPI_STORE, WIDE_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_norm_store_wide(GEMV_ARGS) {
  gemv_body<MODE_NORM, EPI_STORE, WIDE_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_norm_silu_pair_wide(GEMV_ARGS) {
  gemv_body<MODE_NORM, EPI_SILU_PAIR, WIDE_PER_THREAD>(GEMV_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) gemv_attn_residual_wide(GEMV_ARGS) {
  gemv_body<MODE_ATTN, EPI_RESIDUAL, WIDE_PER_THREAD>(GEMV_PASS);
}

#define QUANT_ARGS                                                                                      \
  const float* __restrict__ input, const float* __restrict__ aux, int n, float eps, int splits,        \
      signed char* __restrict__ q8_out, int* __restrict__ xsum_out, float* __restrict__ scale_out
#define QUANT_PASS input, aux, n, eps, splits, q8_out, xsum_out, scale_out

template <int MODE, int PER_THREAD>
__device__ __forceinline__ void quantize_body(QUANT_ARGS) {
  __shared__ float slots[64];
  const float scale = quantize_source<MODE, PER_THREAD>(input, aux, n, eps, splits, q8_out, xsum_out, slots);
  if (threadIdx.x == 0) *scale_out = scale;
}

extern "C" __global__ void __launch_bounds__(THREADS) quantize_norm(QUANT_ARGS) {
  quantize_body<MODE_NORM, NARROW_PER_THREAD>(QUANT_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) quantize_norm_wide(QUANT_ARGS) {
  quantize_body<MODE_NORM, WIDE_PER_THREAD>(QUANT_PASS);
}

extern "C" __global__ void __launch_bounds__(THREADS) embed_token(
    const long long* __restrict__ token, const u8* __restrict__ packed, const half_t* __restrict__ scales,
    float* __restrict__ output, int n, int row_bytes, int groups) {
  const int i = blockIdx.x * THREADS + threadIdx.x;
  if (i >= n) return;
  const size_t row = (size_t)(*token);
  const int code = (packed[row * row_bytes + (i >> 2)] >> ((i & 3) * 2)) & 3;
  output[i] = (float)(code - 1) * h2f(scales[row * groups + (i >> 7)]);
}

extern "C" __global__ void __launch_bounds__(RECURRENT_MAX_THREADS) recurrent_step(
    const float* __restrict__ proj, const float* __restrict__ conv_weight, const float* __restrict__ conv_bias,
    float* __restrict__ conv_state, const float* __restrict__ decay, const float* __restrict__ theta,
    const float* __restrict__ skip, float* __restrict__ ssm_state, float* __restrict__ prev_force,
    float* __restrict__ output, int hidden, int state_dim, int head_dim, int conv_taps,
    float half_dt_min, float half_dt_range) {
  __shared__ float partial[RECURRENT_MAX_THREADS];
  __shared__ float values[RECURRENT_MAX_THREADS];
  __shared__ float coefficients[RECURRENT_MAX_STATE][5];
  const int head = blockIdx.x, tid = threadIdx.x;
  const int s = tid / head_dim, dim = tid - s * head_dim;
  const int c = head * head_dim + dim;
  const float* bcdt = proj + 2 * hidden + head * 3 * state_dim;
  const size_t slot = (size_t)(head * state_dim + s) * head_dim + dim;
  const float previous = prev_force[slot];
  float* state = ssm_state + slot * 2;
  const float sr = state[0], si = state[1];
  if (s == 0) {
    const float* taps = conv_weight + c * conv_taps;
    float* window = conv_state + c * (conv_taps - 1);
    const float incoming = proj[c];
    float conv = conv_bias[c];
    for (int k = 0; k < conv_taps - 1; ++k) conv += taps[k] * window[k];
    conv += taps[conv_taps - 1] * incoming;
    for (int k = 0; k < conv_taps - 2; ++k) window[k] = window[k + 1];
    window[conv_taps - 2] = incoming;
    values[dim] = silu_f(conv);
  }
  if (tid < state_dim) {
    const float half_dt = half_dt_min + sigmoid_f(bcdt[2 * state_dim + tid]) * half_dt_range;
    const int hs = head * state_dim + tid;
    const float magnitude = expf(2.0f * half_dt * decay[hs]);
    float sn, cs;
    sincosf(2.0f * half_dt * theta[hs], &sn, &cs);
    coefficients[tid][0] = tanhf(bcdt[tid]);
    coefficients[tid][1] = tanhf(bcdt[state_dim + tid]);
    coefficients[tid][2] = half_dt;
    coefficients[tid][3] = magnitude * cs;
    coefficients[tid][4] = magnitude * sn;
  }
  __syncthreads();
  const float value = values[dim];
  const float b = coefficients[s][0], cc = coefficients[s][1], half_dt = coefficients[s][2];
  const float ar = coefficients[s][3], ai = coefficients[s][4];
  const float force = b * value;
  const float dr = (force + ar * previous) * half_dt;
  const float di = (ai * previous) * half_dt;
  const float nr = sr * ar - si * ai + dr;
  const float ni = sr * ai + si * ar + di;
  state[0] = nr;
  state[1] = ni;
  prev_force[slot] = force;
  partial[tid] = cc * nr;
  __syncthreads();
  if (s == 0) {
    float total = 0.0f;
    for (int k = 0; k < state_dim; ++k) total += partial[k * head_dim + dim];
    output[c] = (total + skip[c] * value) * sigmoid_f(proj[hidden + c]);
  }
}

__device__ __forceinline__ void load_half_row(const half_t* __restrict__ src, float* dst) {
  if (HEAD_DIM % 8 == 0) {
    const uint4* vec = (const uint4*)src;
#pragma unroll
    for (int c = 0; c < HEAD_DIM / 8; ++c) {
      const uint4 w = __ldg(vec + c);
      dst[c * 8 + 0] = h2f((half_t)(w.x & 0xffffu));
      dst[c * 8 + 1] = h2f((half_t)(w.x >> 16));
      dst[c * 8 + 2] = h2f((half_t)(w.y & 0xffffu));
      dst[c * 8 + 3] = h2f((half_t)(w.y >> 16));
      dst[c * 8 + 4] = h2f((half_t)(w.z & 0xffffu));
      dst[c * 8 + 5] = h2f((half_t)(w.z >> 16));
      dst[c * 8 + 6] = h2f((half_t)(w.w & 0xffffu));
      dst[c * 8 + 7] = h2f((half_t)(w.w >> 16));
    }
  } else {
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) dst[d] = h2f(src[d]);
  }
}

__device__ __forceinline__ void attention_merge(
    float m, float l, const float* acc, float* lane_weight, float (*lane_acc)[HEAD_DIM + 1], float* slots,
    float* __restrict__ out);

extern "C" __global__ void __launch_bounds__(ATTN_THREADS) attention_step(
    const float* __restrict__ qlk, const float* __restrict__ kv, const float* __restrict__ rope,
    const long long* __restrict__ position, half_t* __restrict__ k_cache, half_t* __restrict__ v_cache,
    float* __restrict__ partial, int k_rope_offset, int nope_dim, int rope_dim, int v_offset, int max_len,
    int splits, float scale) {
  __shared__ float sq[HEAD_DIM];
  __shared__ float sk[HEAD_DIM];
  __shared__ float sv[HEAD_DIM];
  __shared__ float lane_weight[ATTN_THREADS];
  __shared__ float lane_acc[ATTN_THREADS][HEAD_DIM + 1];
  __shared__ float slots[64];
  const int head = blockIdx.x, split = blockIdx.y, tid = threadIdx.x;
  const int pos = (int)(*position);
  for (int pair = tid; pair < HALF_DIM; pair += ATTN_THREADS) {
    const int d = 2 * pair;
    const size_t rope_slot = ((size_t)pos * HALF_DIM + pair) * 2;
    const float cr = rope[rope_slot], ci = rope[rope_slot + 1];
    const float q0 = qlk[head * HEAD_DIM + d], q1 = qlk[head * HEAD_DIM + d + 1];
    float k0, k1;
    if (d < nope_dim) {
      k0 = kv[head * nope_dim + d];
      k1 = kv[head * nope_dim + d + 1];
    } else {
      const int r = d - nope_dim;
      k0 = qlk[k_rope_offset + head * rope_dim + r];
      k1 = qlk[k_rope_offset + head * rope_dim + r + 1];
    }
    const float qr0 = q0 * cr - q1 * ci, qr1 = q0 * ci + q1 * cr;
    const float kr0 = k0 * cr - k1 * ci, kr1 = k0 * ci + k1 * cr;
    const float v0 = kv[v_offset + head * HEAD_DIM + d], v1 = kv[v_offset + head * HEAD_DIM + d + 1];
    sq[d] = qr0 * scale;
    sq[d + 1] = qr1 * scale;
    sk[d] = kr0;
    sk[d + 1] = kr1;
    sv[d] = v0;
    sv[d + 1] = v1;
    if (split == 0) {
      const size_t slot = ((size_t)head * max_len + pos) * HEAD_DIM + d;
      k_cache[slot] = f2h(kr0);
      k_cache[slot + 1] = f2h(kr1);
      v_cache[slot] = f2h(v0);
      v_cache[slot + 1] = f2h(v1);
    }
  }
  __syncthreads();
  const int total = pos + 1;
  const int per_split = (total + splits - 1) / splits;
  const int start = split * per_split;
  const int end = min(start + per_split, total);
  float m = NEG_INF, l = 0.0f;
  float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;
  float row[HEAD_DIM];
  for (int j = start + tid; j < end; j += ATTN_THREADS) {
    float score = 0.0f;
    if (j == pos) {
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) score += sq[d] * sk[d];
    } else {
      load_half_row(k_cache + ((size_t)head * max_len + j) * HEAD_DIM, row);
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) score += sq[d] * row[d];
    }
    const float peak = fmaxf(m, score);
    const float correction = expf(m - peak);
    const float weight = expf(score - peak);
    l = l * correction + weight;
    if (j == pos) {
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) row[d] = sv[d];
    } else {
      load_half_row(v_cache + ((size_t)head * max_len + j) * HEAD_DIM, row);
    }
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * correction + weight * row[d];
    m = peak;
  }
  attention_merge(m, l, acc, lane_weight, lane_acc, slots, partial + ((size_t)head * splits + split) * PARTIAL_STRIDE);
}

__device__ __forceinline__ void attention_merge(
    float m, float l, const float* acc, float* lane_weight, float (*lane_acc)[HEAD_DIM + 1], float* slots,
    float* __restrict__ out) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) lane_acc[tid][d] = acc[d];
  const float peak = block_max(m, slots);
  const float weight = m == NEG_INF ? 0.0f : expf(m - peak);
  lane_weight[tid] = weight;
  const float total_weight = block_sum(weight * l, slots + 32);
  for (int d = tid; d < HEAD_DIM; d += ATTN_THREADS) {
    float value = 0.0f;
#pragma unroll 8
    for (int t = 0; t < ATTN_THREADS; ++t) value += lane_weight[t] * lane_acc[t][d];
    out[2 + d] = value;
  }
  if (tid == 0) {
    out[0] = peak;
    out[1] = total_weight;
  }
}

extern "C" __global__ void __launch_bounds__(THREADS) embed_tile(
    const long long* __restrict__ tokens, const u8* __restrict__ packed, const half_t* __restrict__ scales,
    float* __restrict__ output, int n, int row_bytes, int groups, int m) {
  const int t = blockIdx.y;
  const int i = blockIdx.x * THREADS + threadIdx.x;
  if (i >= n || t >= m) return;
  const size_t row = (size_t)tokens[t];
  const int code = (packed[row * row_bytes + (i >> 2)] >> ((i & 3) * 2)) & 3;
  output[(size_t)t * n + i] = (float)(code - 1) * h2f(scales[row * groups + (i >> 7)]);
}

extern "C" __global__ void __launch_bounds__(THREADS) gather_token(
    const float* __restrict__ tile, float* __restrict__ output, int n, const long long* __restrict__ valid) {
  const int i = blockIdx.x * THREADS + threadIdx.x;
  if (i >= n) return;
  const int t = max((int)(*valid) - 1, 0);
  output[i] = tile[(size_t)t * n + i];
}

#define QUANT_TILE_ARGS                                                                                 \
  const float* __restrict__ input, int in_stride, const float* __restrict__ aux, int n, float eps,     \
      int splits, signed char* __restrict__ q8_out, int* __restrict__ xsum_out, float* __restrict__ scale_out
#define QUANT_TILE_PASS input, in_stride, aux, n, eps, splits, q8_out, xsum_out, scale_out

template <int MODE, int PER_THREAD>
__device__ __forceinline__ void quantize_tile_body(QUANT_TILE_ARGS) {
  __shared__ float slots[64];
  const int t = blockIdx.x;
  const int chunks = (n + 63) >> 6;
  const float scale = quantize_source<MODE, PER_THREAD>(
      input + (size_t)t * in_stride, aux, n, eps, splits, q8_out + (size_t)t * chunks * 64, xsum_out + (size_t)t * chunks,
      slots);
  if (threadIdx.x == 0) scale_out[t] = scale;
}

extern "C" __global__ void __launch_bounds__(THREADS) quantize_tile_plain(QUANT_TILE_ARGS) {
  quantize_tile_body<MODE_PLAIN, NARROW_PER_THREAD>(QUANT_TILE_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) quantize_tile_plain_wide(QUANT_TILE_ARGS) {
  quantize_tile_body<MODE_PLAIN, WIDE_PER_THREAD>(QUANT_TILE_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) quantize_tile_norm(QUANT_TILE_ARGS) {
  quantize_tile_body<MODE_NORM, NARROW_PER_THREAD>(QUANT_TILE_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) quantize_tile_norm_wide(QUANT_TILE_ARGS) {
  quantize_tile_body<MODE_NORM, WIDE_PER_THREAD>(QUANT_TILE_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) quantize_tile_attn(QUANT_TILE_ARGS) {
  quantize_tile_body<MODE_ATTN, NARROW_PER_THREAD>(QUANT_TILE_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS) quantize_tile_attn_wide(QUANT_TILE_ARGS) {
  quantize_tile_body<MODE_ATTN, WIDE_PER_THREAD>(QUANT_TILE_PASS);
}

#define GEMM_ARGS                                                                                        \
  const signed char* __restrict__ q8_in, const int* __restrict__ xsum_in, const float* __restrict__ scale_in, \
      const u8* __restrict__ packed, const half_t* __restrict__ scales, const float* __restrict__ bias,    \
      float* __restrict__ output, int rows, int n, int row_bytes, int groups, int lanes, int out_stride, int m
#define GEMM_PASS q8_in, xsum_in, scale_in, packed, scales, bias, output, rows, n, row_bytes, groups, lanes, out_stride, m

template <int GROUP, int CAP>
__device__ __forceinline__ void load_group(
    signed char (*buffer)[CAP * 64], const signed char* __restrict__ q8_in, int first, int m, int chunks) {
  const int quads = chunks * 4;
  for (int i = threadIdx.x; i < GROUP * quads; i += THREADS) {
    const int g = i / quads, w = i - g * quads;
    const int t = first + g;
    if (t < m) ((int4*)buffer[g])[w] = ((const int4*)(q8_in + (size_t)t * chunks * 64))[w];
  }
}

template <int EPI, int GROUP, int CAP>
__device__ __forceinline__ void gemm_tile_body(GEMM_ARGS) {
  __shared__ __align__(16) signed char q8[GROUP][CAP * 64];
  __shared__ int xsum[TILE * CAP];
  __shared__ float token_scale[TILE];
  const int tid = threadIdx.x;
  const int chunks = (n + 63) >> 6;
  const int row = blockIdx.x * (THREADS / lanes) + tid / lanes;
  const int sub = tid & (lanes - 1);
  const bool active = row < rows;
  const uint4* wrow = (const uint4*)(packed + (size_t)(active ? row : 0) * row_bytes);
  const half_t* srow = scales + (size_t)(active ? row : 0) * groups;
  const float row_bias = active && bias ? bias[row] : 0.0f;
  uint4 ahead[PREFETCH];
#pragma unroll
  for (int k = 0; k < PREFETCH; ++k) {
    const int u = sub + k * lanes;
    ahead[k] = (active && u < chunks) ? __ldg(wrow + u) : make_uint4(0u, 0u, 0u, 0u);
  }
  for (int i = tid; i < m * chunks; i += THREADS) xsum[i] = xsum_in[i];
  if (tid < m) token_scale[tid] = scale_in[tid];
  for (int base = 0; base < m; base += GROUP) {
    __syncthreads();
    load_group<GROUP, CAP>(q8, q8_in, base, m, chunks);
    float acc[GROUP];
    float residual[GROUP];
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      acc[g] = 0.0f;
      const int t = base + g;
      residual[g] = EPI == EPI_RESIDUAL && t < m && active && sub == 0 ? output[(size_t)t * out_stride + row] : 0.0f;
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int k = 0; k < PREFETCH; ++k) {
        const int u = sub + k * lanes;
        if (u < chunks) {
          int codes[16];
          unpack_chunk(ahead[k], codes);
          const float scale = h2f(srow[u >> 1]);
          const int rotation = (u >> 1) & 3;
#pragma unroll
          for (int g = 0; g < GROUP; ++g) {
            acc[g] += chunk_dot_unpacked(codes, q8[g] + u * 64, rotation, xsum[(base + g) * chunks + u], scale);
          }
        }
      }
      for (int u = sub + PREFETCH * lanes; u < chunks; u += lanes) {
        int codes[16];
        unpack_chunk(__ldg(wrow + u), codes);
        const float scale = h2f(srow[u >> 1]);
        const int rotation = (u >> 1) & 3;
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
          acc[g] += chunk_dot_unpacked(codes, q8[g] + u * 64, rotation, xsum[(base + g) * chunks + u], scale);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      for (int offset = 1; offset < lanes; offset <<= 1) acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], offset);
    }
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      const int t = base + g;
      const float result = acc[g] * token_scale[t < m ? t : 0] + row_bias;
      if (EPI == EPI_SILU_PAIR) {
        const float partner = __shfl_down_sync(0xffffffffu, result, lanes);
        if (t < m && active && sub == 0 && (row & 1) == 0) {
          output[(size_t)t * out_stride + (row >> 1)] = silu_f(result) * partner;
        }
      } else if (t < m && active && sub == 0) {
        output[(size_t)t * out_stride + row] = EPI == EPI_RESIDUAL ? residual[g] + result : result;
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(THREADS, GEMM_MIN_BLOCKS) gemm_tile_store(GEMM_ARGS) {
  gemm_tile_body<EPI_STORE, TILE_GROUP_NARROW, NARROW_CHUNKS>(GEMM_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS, GEMM_MIN_BLOCKS) gemm_tile_residual(GEMM_ARGS) {
  gemm_tile_body<EPI_RESIDUAL, TILE_GROUP_NARROW, NARROW_CHUNKS>(GEMM_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS, GEMM_MIN_BLOCKS) gemm_tile_silu_pair(GEMM_ARGS) {
  gemm_tile_body<EPI_SILU_PAIR, TILE_GROUP_NARROW, NARROW_CHUNKS>(GEMM_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS, GEMM_MIN_BLOCKS) gemm_tile_store_wide(GEMM_ARGS) {
  gemm_tile_body<EPI_STORE, TILE_GROUP_WIDE, MAX_CHUNKS>(GEMM_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS, GEMM_MIN_BLOCKS) gemm_tile_residual_wide(GEMM_ARGS) {
  gemm_tile_body<EPI_RESIDUAL, TILE_GROUP_WIDE, MAX_CHUNKS>(GEMM_PASS);
}
extern "C" __global__ void __launch_bounds__(THREADS, GEMM_MIN_BLOCKS) gemm_tile_silu_pair_wide(GEMM_ARGS) {
  gemm_tile_body<EPI_SILU_PAIR, TILE_GROUP_WIDE, MAX_CHUNKS>(GEMM_PASS);
}

extern "C" __global__ void __launch_bounds__(RECURRENT_MAX_THREADS) recurrent_tile(
    const float* __restrict__ proj, int proj_stride, const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias, float* __restrict__ conv_state, const float* __restrict__ decay,
    const float* __restrict__ theta, const float* __restrict__ skip, float* __restrict__ ssm_state,
    float* __restrict__ prev_force, float* __restrict__ output, int hidden, int state_dim, int head_dim,
    int conv_taps, float half_dt_min, float half_dt_range, int m, const long long* __restrict__ valid_ptr) {
  __shared__ float coefficients[TILE][TILE_STATE * 5];
  __shared__ float values[TILE][TILE_HEAD];
  __shared__ float partial[TILE_PHASE][RECURRENT_MAX_THREADS];
  const int head = blockIdx.x, tid = threadIdx.x;
  const int s = tid / head_dim, dim = tid - s * head_dim;
  const int c = head * head_dim + dim;
  const int valid = min((int)(*valid_ptr), m);
  const size_t slot = (size_t)(head * state_dim + s) * head_dim + dim;
  float force_prev = prev_force[slot];
  float sr = ssm_state[slot * 2], si = ssm_state[slot * 2 + 1];
  for (int index = tid; index < head_dim * valid; index += blockDim.x) {
    const int t = index / head_dim, d = index - t * head_dim;
    values[t][d] = proj[(size_t)t * proj_stride + head * head_dim + d];
  }
  for (int index = tid; index < state_dim * valid; index += blockDim.x) {
    const int t = index / state_dim, ss = index - t * state_dim;
    const float* bcdt = proj + (size_t)t * proj_stride + 2 * hidden + head * 3 * state_dim;
    const float half_dt = half_dt_min + sigmoid_f(bcdt[2 * state_dim + ss]) * half_dt_range;
    const int hs = head * state_dim + ss;
    const float magnitude = expf(2.0f * half_dt * decay[hs]);
    float sn, cs;
    sincosf(2.0f * half_dt * theta[hs], &sn, &cs);
    float* co = coefficients[t] + ss * 5;
    co[0] = tanhf(bcdt[ss]);
    co[1] = tanhf(bcdt[state_dim + ss]);
    co[2] = half_dt;
    co[3] = magnitude * cs;
    co[4] = magnitude * sn;
  }
  __syncthreads();
  if (s == 0) {
    const float* taps = conv_weight + c * conv_taps;
    float* window = conv_state + c * (conv_taps - 1);
    const float bias_c = conv_bias[c];
    float ring[TILE_TAPS];
#pragma unroll
    for (int k = 0; k < TILE_TAPS; ++k) ring[k] = k < conv_taps - 1 ? window[k] : 0.0f;
    for (int t = 0; t < valid; ++t) {
      const float incoming = values[t][dim];
      float conv = bias_c;
#pragma unroll
      for (int k = 0; k < TILE_TAPS; ++k) {
        if (k < conv_taps - 1) conv += taps[k] * ring[k];
      }
      conv += taps[conv_taps - 1] * incoming;
#pragma unroll
      for (int k = 0; k < TILE_TAPS - 1; ++k) {
        if (k < conv_taps - 2) ring[k] = ring[k + 1];
      }
#pragma unroll
      for (int k = 0; k < TILE_TAPS; ++k) {
        if (k == conv_taps - 2) ring[k] = incoming;
      }
      values[t][dim] = silu_f(conv);
    }
#pragma unroll
    for (int k = 0; k < TILE_TAPS; ++k) {
      if (k < conv_taps - 1) window[k] = ring[k];
    }
  }
  __syncthreads();
  float gates[TILE_PHASE];
  if (s == 0) {
#pragma unroll
    for (int k = 0; k < TILE_PHASE; ++k) gates[k] = k < valid ? proj[(size_t)k * proj_stride + hidden + c] : 0.0f;
  }
  for (int base = 0; base < valid; base += TILE_PHASE) {
#pragma unroll
    for (int k = 0; k < TILE_PHASE; ++k) {
      const int t = base + k;
      if (t < valid) {
        const float* co = coefficients[t] + s * 5;
        const float force = co[0] * values[t][dim];
        const float dr = (force + co[3] * force_prev) * co[2];
        const float di = (co[4] * force_prev) * co[2];
        const float nr = sr * co[3] - si * co[4] + dr;
        const float ni = sr * co[4] + si * co[3] + di;
        sr = nr;
        si = ni;
        force_prev = force;
        partial[k][tid] = co[1] * nr;
      }
    }
    float next_gates[TILE_PHASE];
    if (s == 0) {
#pragma unroll
      for (int k = 0; k < TILE_PHASE; ++k) {
        const int t = base + TILE_PHASE + k;
        next_gates[k] = t < valid ? proj[(size_t)t * proj_stride + hidden + c] : 0.0f;
      }
    }
    __syncthreads();
    if (s == 0) {
#pragma unroll
      for (int k = 0; k < TILE_PHASE; ++k) {
        const int t = base + k;
        if (t < valid) {
          float total = 0.0f;
          for (int j = 0; j < state_dim; ++j) total += partial[k][j * head_dim + dim];
          output[(size_t)t * hidden + c] = (total + skip[c] * values[t][dim]) * sigmoid_f(gates[k]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE_PHASE; ++k) gates[k] = next_gates[k];
  }
  ssm_state[slot * 2] = sr;
  ssm_state[slot * 2 + 1] = si;
  prev_force[slot] = force_prev;
}

extern "C" __global__ void __launch_bounds__(ATTN_THREADS) kv_write_tile(
    const float* __restrict__ qlk, int qlk_stride, const float* __restrict__ kv, int kv_stride,
    const float* __restrict__ rope, const long long* __restrict__ position, half_t* __restrict__ k_cache,
    half_t* __restrict__ v_cache, int k_rope_offset, int nope_dim, int rope_dim, int v_offset, int max_len,
    int m, const long long* __restrict__ valid_ptr) {
  const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  if (t >= min((int)(*valid_ptr), m)) return;
  const int pos = (int)(*position) + t;
  const float* q_row = qlk + (size_t)t * qlk_stride;
  const float* kv_row = kv + (size_t)t * kv_stride;
  for (int pair = tid; pair < HALF_DIM; pair += ATTN_THREADS) {
    const int d = 2 * pair;
    const size_t rope_slot = ((size_t)pos * HALF_DIM + pair) * 2;
    const float cr = rope[rope_slot], ci = rope[rope_slot + 1];
    float k0, k1;
    if (d < nope_dim) {
      k0 = kv_row[head * nope_dim + d];
      k1 = kv_row[head * nope_dim + d + 1];
    } else {
      const int r = d - nope_dim;
      k0 = q_row[k_rope_offset + head * rope_dim + r];
      k1 = q_row[k_rope_offset + head * rope_dim + r + 1];
    }
    const size_t slot = ((size_t)head * max_len + pos) * HEAD_DIM + d;
    k_cache[slot] = f2h(k0 * cr - k1 * ci);
    k_cache[slot + 1] = f2h(k0 * ci + k1 * cr);
    v_cache[slot] = f2h(kv_row[v_offset + head * HEAD_DIM + d]);
    v_cache[slot + 1] = f2h(kv_row[v_offset + head * HEAD_DIM + d + 1]);
  }
}

extern "C" __global__ void __launch_bounds__(ATTN_THREADS) attention_tile(
    const float* __restrict__ qlk, int qlk_stride, const float* __restrict__ rope,
    const long long* __restrict__ position, const half_t* __restrict__ k_cache, const half_t* __restrict__ v_cache,
    float* __restrict__ partial, int partial_stride, int max_len, int splits, float scale, int m,
    const long long* __restrict__ valid_ptr) {
  __shared__ float sq[HEAD_DIM];
  __shared__ float lane_weight[ATTN_THREADS];
  __shared__ float lane_acc[ATTN_THREADS][HEAD_DIM + 1];
  __shared__ float slots[64];
  const int head = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  if (t >= min((int)(*valid_ptr), m)) return;
  const int pos = (int)(*position) + t;
  const float* q_row = qlk + (size_t)t * qlk_stride + head * HEAD_DIM;
  for (int pair = tid; pair < HALF_DIM; pair += ATTN_THREADS) {
    const int d = 2 * pair;
    const size_t rope_slot = ((size_t)pos * HALF_DIM + pair) * 2;
    const float cr = rope[rope_slot], ci = rope[rope_slot + 1];
    const float q0 = q_row[d], q1 = q_row[d + 1];
    sq[d] = (q0 * cr - q1 * ci) * scale;
    sq[d + 1] = (q0 * ci + q1 * cr) * scale;
  }
  __syncthreads();
  const int total = pos + 1;
  const int per_split = (total + splits - 1) / splits;
  const int start = split * per_split;
  const int end = min(start + per_split, total);
  float m_run = NEG_INF, l = 0.0f;
  float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;
  float row[HEAD_DIM];
  for (int j = start + tid; j < end; j += ATTN_THREADS) {
    load_half_row(k_cache + ((size_t)head * max_len + j) * HEAD_DIM, row);
    float score = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) score += sq[d] * row[d];
    const float peak = fmaxf(m_run, score);
    const float correction = expf(m_run - peak);
    const float weight = expf(score - peak);
    l = l * correction + weight;
    load_half_row(v_cache + ((size_t)head * max_len + j) * HEAD_DIM, row);
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * correction + weight * row[d];
    m_run = peak;
  }
  float* out = partial + (size_t)t * partial_stride + ((size_t)head * splits + split) * PARTIAL_STRIDE;
  attention_merge(m_run, l, acc, lane_weight, lane_acc, slots, out);
}

__device__ __forceinline__ unsigned int sortable_key(float v) {
  const unsigned int u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float sampling_score(
    const float* __restrict__ logits, const u8* __restrict__ seen, int i, float inverse_temperature,
    float penalty, float inverse_penalty) {
  float score = logits[i] * inverse_temperature;
  if (seen[i]) score *= score > 0.0f ? inverse_penalty : penalty;
  return score;
}

__device__ __forceinline__ void sampling_scores(
    const float* __restrict__ logits, const u8* __restrict__ seen, int base, int vocab, float inverse_temperature,
    float penalty, float inverse_penalty, float* scores) {
  if (base + 3 < vocab) {
    const float4 four = *(const float4*)(logits + base);
    const uchar4 flags = *(const uchar4*)(seen + base);
    scores[0] = four.x * inverse_temperature;
    scores[1] = four.y * inverse_temperature;
    scores[2] = four.z * inverse_temperature;
    scores[3] = four.w * inverse_temperature;
    if (flags.x) scores[0] *= scores[0] > 0.0f ? inverse_penalty : penalty;
    if (flags.y) scores[1] *= scores[1] > 0.0f ? inverse_penalty : penalty;
    if (flags.z) scores[2] *= scores[2] > 0.0f ? inverse_penalty : penalty;
    if (flags.w) scores[3] *= scores[3] > 0.0f ? inverse_penalty : penalty;
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      scores[e] = base + e < vocab
                      ? sampling_score(logits, seen, base + e, inverse_temperature, penalty, inverse_penalty)
                      : NEG_INF;
    }
  }
}

struct SampleSelection {
  unsigned int prefix;
  int remaining;
  int bin_count;
};

__device__ __forceinline__ void count_key(unsigned int* histogram, int warp, unsigned int digit) {
  atomicAdd(&histogram[warp * 128 + (digit >> 1)], 1u << (16 * (digit & 1)));
}

__device__ __forceinline__ unsigned int read_count(const unsigned int* histogram, int warp, int digit) {
  return (histogram[warp * 128 + (digit >> 1)] >> (16 * (digit & 1))) & 0xffffu;
}

__device__ __forceinline__ void select_digit(
    unsigned int* histogram, unsigned int* suffix, unsigned int* totals, SampleSelection* selection,
    unsigned int prefix, int remaining, int shift) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < 256) {
    unsigned int total = 0u;
    for (int w = 0; w < SAMPLE_WARPS; ++w) total += read_count(histogram, w, tid);
    unsigned int running = total;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const unsigned int other = __shfl_down_sync(0xffffffffu, running, offset);
      if (lane + offset < 32) running += other;
    }
    totals[tid] = total;
    suffix[tid] = running;
  }
  __syncthreads();
  if (tid < 256) {
    unsigned int above = suffix[tid] - totals[tid];
    for (int w = warp + 1; w < 8; ++w) above += suffix[w * 32];
    const unsigned int count = totals[tid];
    if ((int)above < remaining && remaining <= (int)(above + count)) {
      selection->prefix = prefix | ((unsigned int)tid << shift);
      selection->remaining = remaining - (int)above;
      selection->bin_count = (int)count;
    }
  }
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS) sample_token(
    const float* __restrict__ logits, const float* __restrict__ temperature, const float* __restrict__ penalty,
    const float* __restrict__ inverse_penalty, u8* __restrict__ seen, const long long* __restrict__ top_k,
    const float* __restrict__ uniform, long long* __restrict__ token, int vocab) {
  __shared__ unsigned int histogram[SAMPLE_WARPS * 128];
  __shared__ unsigned int suffix[256];
  __shared__ unsigned int totals[256];
  __shared__ SampleSelection selection;
  __shared__ int definite_shared;
  __shared__ int candidate_shared[2];
  __shared__ int chosen_shared;
  __shared__ unsigned int candidate_key[2][SAMPLE_THREADS];
  __shared__ int candidate_index[2][SAMPLE_THREADS];
  __shared__ int pick_index[MAX_TOPK];
  __shared__ float slots[64];
  __shared__ float warp_totals[SAMPLE_WARPS];
  __shared__ float cdf[SAMPLE_THREADS];
  __shared__ int delta_counts[SAMPLE_DELTAS][SAMPLE_WARPS];
  __shared__ float threshold_shared;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float inverse_temperature = 1.0f / *temperature;
  const float pen = *penalty, inv_pen = *inverse_penalty;
  const int k = max(1, min((int)(*top_k), min(MAX_TOPK, vocab)));
  const int stride = SAMPLE_THREADS * 4;
  const float deltas[SAMPLE_DELTAS] = {4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 32.0f};
  unsigned int prefix = 0u;
  unsigned int resolved = 0u;
  int remaining = k;
  int shift = 32;
  float scores[SAMPLE_BATCH][4];
  if (tid == 0) {
    definite_shared = 0;
    candidate_shared[0] = 0;
    candidate_shared[1] = 0;
    chosen_shared = -1;
    threshold_shared = NEG_INF;
  }
  float local_peak = NEG_INF;
  for (int base = tid * 4; base < vocab; base += stride * SAMPLE_BATCH) {
#pragma unroll
    for (int b = 0; b < SAMPLE_BATCH; ++b) {
      sampling_scores(logits, seen, base + b * stride, vocab, inverse_temperature, pen, inv_pen, scores[b]);
#pragma unroll
      for (int e = 0; e < 4; ++e) local_peak = fmaxf(local_peak, scores[b][e]);
    }
  }
  const float peak_score = block_max(local_peak, slots);
  int counts[SAMPLE_DELTAS];
#pragma unroll
  for (int d = 0; d < SAMPLE_DELTAS; ++d) counts[d] = 0;
  for (int base = tid * 4; base < vocab; base += stride * SAMPLE_BATCH) {
#pragma unroll
    for (int b = 0; b < SAMPLE_BATCH; ++b) {
      sampling_scores(logits, seen, base + b * stride, vocab, inverse_temperature, pen, inv_pen, scores[b]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int d = 0; d < SAMPLE_DELTAS; ++d) counts[d] += scores[b][e] >= peak_score - deltas[d] ? 1 : 0;
      }
    }
  }
#pragma unroll
  for (int d = 0; d < SAMPLE_DELTAS; ++d) {
    const int total = warp_sum_int(counts[d]);
    if (lane == 0) delta_counts[d][warp] = total;
  }
  __syncthreads();
  if (tid == 0) {
    for (int d = SAMPLE_DELTAS - 1; d >= 0; --d) {
      int total = 0;
      for (int w = 0; w < SAMPLE_WARPS; ++w) total += delta_counts[d][w];
      if (total >= k && total <= SAMPLE_THREADS) {
        threshold_shared = peak_score - deltas[d];
        break;
      }
    }
  }
  __syncthreads();
  const float threshold = threshold_shared;
  const bool filtered = threshold != NEG_INF;
  if (!filtered) {
    while (shift > 0) {
      shift -= 8;
      for (int bin = tid; bin < SAMPLE_WARPS * 128; bin += SAMPLE_THREADS) histogram[bin] = 0u;
      __syncthreads();
      for (int base = tid * 4; base < vocab; base += stride * SAMPLE_BATCH) {
#pragma unroll
        for (int b = 0; b < SAMPLE_BATCH; ++b) {
          sampling_scores(logits, seen, base + b * stride, vocab, inverse_temperature, pen, inv_pen, scores[b]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = base + b * stride + e;
            const unsigned int key = sortable_key(scores[b][e]);
            if (i < vocab && (key & resolved) == prefix) count_key(histogram, warp, (key >> shift) & 0xffu);
          }
        }
      }
      __syncthreads();
      select_digit(histogram, suffix, totals, &selection, prefix, remaining, shift);
      prefix = selection.prefix;
      remaining = selection.remaining;
      resolved = 0xffffffffu << shift;
      if (selection.bin_count <= SAMPLE_THREADS) break;
    }
  }
  int phase = 0;
  for (int base = tid * 4; base < vocab; base += stride * SAMPLE_BATCH) {
#pragma unroll
    for (int b = 0; b < SAMPLE_BATCH; ++b) {
      sampling_scores(logits, seen, base + b * stride, vocab, inverse_temperature, pen, inv_pen, scores[b]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = base + b * stride + e;
        if (i >= vocab) continue;
        const unsigned int key = sortable_key(scores[b][e]);
        if (filtered) {
          if (scores[b][e] >= threshold) {
            const int slot = atomicAdd(&candidate_shared[0], 1);
            if (slot < SAMPLE_THREADS) {
              candidate_key[0][slot] = key;
              candidate_index[0][slot] = i;
            }
          }
          continue;
        }
        const unsigned int masked = key & resolved;
        if (masked > prefix) {
          const int slot = atomicAdd(&definite_shared, 1);
          if (slot < MAX_TOPK) pick_index[slot] = i;
        } else if (masked == prefix) {
          const int slot = atomicAdd(&candidate_shared[0], 1);
          if (slot < SAMPLE_THREADS) {
            candidate_key[0][slot] = key;
            candidate_index[0][slot] = i;
          }
        }
      }
    }
  }
  __syncthreads();
  int candidates = min(candidate_shared[0], SAMPLE_THREADS);
  while (candidates > remaining && shift > 0) {
    shift -= 8;
    for (int bin = tid; bin < SAMPLE_WARPS * 128; bin += SAMPLE_THREADS) histogram[bin] = 0u;
    if (tid == 0) candidate_shared[phase ^ 1] = 0;
    __syncthreads();
    if (tid < candidates) count_key(histogram, warp, (candidate_key[phase][tid] >> shift) & 0xffu);
    __syncthreads();
    select_digit(histogram, suffix, totals, &selection, 0u, remaining, shift);
    const unsigned int chosen_digit = selection.prefix >> shift;
    if (tid < candidates) {
      const unsigned int key = candidate_key[phase][tid];
      const unsigned int digit = (key >> shift) & 0xffu;
      if (digit > chosen_digit) {
        const int slot = atomicAdd(&definite_shared, 1);
        if (slot < MAX_TOPK) pick_index[slot] = candidate_index[phase][tid];
      } else if (digit == chosen_digit) {
        const int slot = atomicAdd(&candidate_shared[phase ^ 1], 1);
        candidate_key[phase ^ 1][slot] = key;
        candidate_index[phase ^ 1][slot] = candidate_index[phase][tid];
      }
    }
    __syncthreads();
    phase ^= 1;
    candidates = candidate_shared[phase];
    remaining = selection.remaining;
  }
  const int definite = min(definite_shared, MAX_TOPK);
  const int taken = min(remaining, candidates);
  if (tid < taken && definite + tid < MAX_TOPK) pick_index[definite + tid] = candidate_index[phase][tid];
  __syncthreads();
  const int count = min(definite + taken, MAX_TOPK);
  const float mine = tid < count
                         ? sampling_score(logits, seen, pick_index[tid], inverse_temperature, pen, inv_pen)
                         : NEG_INF;
  const float peak = block_max(mine, slots);
  const float probability = tid < count ? expf(mine - peak) : 0.0f;
  float running = probability;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float other = __shfl_up_sync(0xffffffffu, running, offset);
    if (lane >= offset) running += other;
  }
  if (lane == 31) warp_totals[warp] = running;
  __syncthreads();
  float carry = 0.0f;
  for (int w = 0; w < warp; ++w) carry += warp_totals[w];
  cdf[tid] = carry + running;
  __syncthreads();
  const float total = cdf[count - 1];
  const float target = fminf(*uniform, 0.99999994f) * total;
  const float lower = tid > 0 ? cdf[tid - 1] : 0.0f;
  if (tid < count && lower <= target && target < cdf[tid]) chosen_shared = tid;
  __syncthreads();
  if (tid == 0) {
    int chosen = chosen_shared;
    if (chosen < 0) chosen = count - 1;
    const int index = pick_index[chosen];
    token[0] = (long long)index;
    seen[index] = 1;
  }
}
