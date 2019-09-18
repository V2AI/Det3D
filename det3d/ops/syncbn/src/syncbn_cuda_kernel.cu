#include <vector>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/TensorAccessor.h>

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct SumOp {
  __device__ SumOp(const PTA& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PTA& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % WARP_SIZE == 0) {
    shared[tid / WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t, typename accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
    at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output,
    const at::PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, at::RestrictPtrTraits, index_t> mean_,
    const at::PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, at::RestrictPtrTraits, index_t> var_or_std,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> weight,
    const at::PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> bias,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd = 1.0 / var_or_std[plane];

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> input,
    at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> save_mean,
    at::PackedTensorAccessor<accscalar_t, 1, at::RestrictPtrTraits, index_t> save_mean2) {

  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  accscalar_t* shared_avg_var = (accscalar_t*) &shared_n[WARP_SIZE];

  // first the reductions each thread does separately
  accscalar_t avg = 0;
  accscalar_t avg2 = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      accscalar_t v = input[batch][plane][x];
      accscalar_t d1 = v - avg;
      accscalar_t d2 = (v * v) - avg2;
      n++;
      avg += d1 / n;
      avg2 += d2 / n;
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    accscalar_t o_avg2 = WARP_SHFL_XOR(avg2, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    // var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg2 = (n * avg2 + o_n * o_avg2) * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = avg2;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid] : 0);
    avg2 = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : 0);
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    accscalar_t o_avg2 = WARP_SHFL_XOR(avg2, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    // var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg2 = (n * avg2 + o_n * o_avg2) * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    /*
    accscalar_t invstd = 0;
    if (var_n != static_cast<accscalar_t>(0) || epsilon != static_cast<accscalar_t>(0)) {
      invstd = static_cast<accscalar_t>(1) / device_sqrt(var_n / N + epsilon);
    }
    */
    save_mean[plane] = avg;
    save_mean2[plane] = avg2;

  }

}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_collect_grad_statistics_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> input,
    const at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> grad_output,
    at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_weight,
    at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_bias,
    at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_ex,
    at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_exs,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> weight,
    const at::PackedTensorAccessor<accscalar_t, 1, at::DefaultPtrTraits, index_t> save_mean,
    const at::PackedTensorAccessor<accscalar_t, 1, at::DefaultPtrTraits, index_t> save_std,
    accscalar_t epsilon,
    accscalar_t cf) {

    index_t plane = blockIdx.x;
    index_t N = grad_output.size(0) * grad_output.size(2);

    accscalar_t mean, invstd;
    mean = save_mean[plane];
    invstd = 1.0 / save_std[plane];
    /*
    if (train) {
        mean = save_mean[plane];
        invstd = 1.0 / save_std[plane];
    } else {
        mean = static_cast<accscalar_t>(running_mean[plane]);
        invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(running_var[plane]) + epsilon);
    }
    */
    accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
    // accscalar_t norm = accscalar_t(1) / N;

    // Compute two values across (batch, x/y/z) in one pass:
    // 1. Sum(grad_output)
    // 2. DotProduct(input - mean, grad_output)
    GradOp<scalar_t, accscalar_t, at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t>> g(mean, input, grad_output);
    Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t,
                                                                                       at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t>>>(g, grad_output, plane);
    accscalar_t grad_output_sum = res.v1;
    accscalar_t dot_p = res.v2;
    /*
    accscalar_t grad_mean = grad_output_sum * norm;
    accscalar_t proj_scale = dot_p * norm * invstd * invstd;
    accscalar_t grad_scale = invstd * weight_val;

    if (grad_input.data() != NULL) {
        for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
            for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
                scalar_t go = grad_output[batch][plane][x];
                if (train) {
                    scalar_t inp = input[batch][plane][x];
                    accscalar_t proj = (inp - mean) * proj_scale;
                    grad_input[batch][plane][x] = static_cast<scalar_t>((go - proj - grad_mean) * grad_scale);
                } else {
                    grad_input[batch][plane][x] = static_cast<scalar_t>(go * grad_scale);
                }
            }
        }
    }
    */
    if (threadIdx.x == 0) {
        grad_exs[plane] = static_cast<scalar_t>(dot_p * weight_val * (-0.5) * pow(invstd, 3) * cf);
        grad_ex[plane] = static_cast<scalar_t>(grad_output_sum * weight_val * (-1.0) * invstd + \
                dot_p * weight_val * pow(invstd, 3) * mean * cf);
    }
    if (grad_weight.size(0) > 0) {
        if (threadIdx.x == 0) {
            // printf("dot_p = %f, invstd = %f\n", dot_p, invstd);
            grad_weight[plane] = static_cast<scalar_t>(dot_p * invstd);
        }
    }

    if (grad_bias.size(0) > 0) {
        if (threadIdx.x == 0) {
            grad_bias[plane] = static_cast<scalar_t>(grad_output_sum);
        }
    }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> input,
    const at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> grad_output,
    at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> grad_input,
    at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_weight,
    at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_bias,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> weight,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> running_mean,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> running_var,
    const at::PackedTensorAccessor<accscalar_t, 1, at::DefaultPtrTraits, index_t> save_mean,
    const at::PackedTensorAccessor<accscalar_t, 1, at::DefaultPtrTraits, index_t> save_invstd,
    bool train,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<accscalar_t>(running_mean[plane]);
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(running_var[plane]) + epsilon);
  }

  accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
  accscalar_t norm = accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<scalar_t, accscalar_t, at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t>> g(mean, input, grad_output);
  Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t,
                                                                                   at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t>>>(g, grad_output, plane);
  accscalar_t grad_output_sum = res.v1;
  accscalar_t dot_p = res.v2;

  accscalar_t grad_mean = grad_output_sum * norm;
  accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        scalar_t go = grad_output[batch][plane][x];
        if (train) {
          scalar_t inp = input[batch][plane][x];
          accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<scalar_t>(grad_output_sum);
    }
  }
}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = at::DefaultPtrTraits, typename index_t = int64_t>
static at::PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const at::Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return at::PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

std::vector<at::Tensor> batch_norm_collect_statistics_cuda(
    const at::Tensor input) {

    // const auto batch_size = input.size(0);
    const auto channel_size = input.size(1);
    // const auto dim_size = input.size(2);

    auto input_reshaped = input.reshape({input.size(0), input.size(1), -1}); // internally we merge the feature dimensions

    auto ex   = at::empty({channel_size}, input.options());
    auto exs  = at::empty({channel_size}, input.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 blocks(input_reshaped.size(1));
    int tf = getNumThreads(input_reshaped.size(2));
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "batch_norm_collect_statistics_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        if (at::cuda::detail::canUse32BitIndexMath(input)) {
            batch_norm_collect_statistics_kernel<scalar_t, accscalar_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>(),
                ex.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int32_t>(),
                exs.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int32_t>());
        } else {
            batch_norm_collect_statistics_kernel<scalar_t, accscalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int64_t>(),
                ex.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int64_t>(),
                exs.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int64_t>());
        }
    }));
    THCudaCheck(cudaGetLastError());
    return {ex, exs};
}

at::Tensor batch_norm_transform_input_cuda(
    const at::Tensor input,
    const at::Tensor gamma,
    const at::Tensor beta,
    const at::Tensor ex,
    const at::Tensor exs,
    float eps,
    float cf) {
    const auto channel_size = input.size(1);

    auto input_reshaped = input.reshape({input.size(0), input.size(1), -1}); // internally we merge the feature dimensions
    auto output_reshaped = at::empty_like(input_reshaped);
    auto std = (cf * (exs - ex * ex) + eps).sqrt();

    auto stream = at::cuda::getCurrentCUDAStream();

    int tf = std::max<int>(getNumThreads(input_reshaped.size(2)/4),
                           std::min<int>(getNumThreads(input_reshaped.size(2)), 64));
    int tb = std::max<int>(64/tf, 1);
    dim3 blocks_trans(input_reshaped.size(1), std::max<int>(1, std::min<int>((256*1024)/input_reshaped.size(1),
                                                                    (input_reshaped.size(0)+tb-1)/tb)));
    dim3 threads_trans(tf, tb);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "batch_norm_transform_input_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        if (at::cuda::detail::canUse32BitIndexMath(input)) {
            batch_norm_transform_input_kernel<scalar_t, accscalar_t, true, int32_t><<<blocks_trans, threads_trans, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>(),
                output_reshaped.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>(),
                ex.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int32_t>(),
                std.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int32_t>(),
                packed_accessor_or_dummy<scalar_t, 1, at::RestrictPtrTraits, int32_t>(gamma),
                packed_accessor_or_dummy<scalar_t, 1, at::RestrictPtrTraits, int32_t>(beta),
                eps);
        } else {
            batch_norm_transform_input_kernel<scalar_t, accscalar_t, true, int64_t><<<blocks_trans, threads_trans, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int64_t>(),
                output_reshaped.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int64_t>(),
                ex.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int64_t>(),
                std.packed_accessor<accscalar_t, 1, at::RestrictPtrTraits, int64_t>(),
                packed_accessor_or_dummy<scalar_t, 1, at::RestrictPtrTraits, int64_t>(gamma),
                packed_accessor_or_dummy<scalar_t, 1, at::RestrictPtrTraits, int64_t>(beta),
                eps);
        }
    }));
    THCudaCheck(cudaGetLastError());
    return output_reshaped.view(input.sizes());
}

std::vector<at::Tensor> batch_norm_collect_grad_statistics_cuda(
    const at::Tensor input,
    const at::Tensor grad_output,
    const at::Tensor weight,
    const at::Tensor ex,
    const at::Tensor exs,
    float eps,
    float cf) {
    const auto channel_size = input.size(1);

    auto input_reshaped = input.reshape({input.size(0), input.size(1), -1}); // internally we merge the feature dimensions
    auto grad_output_reshaped = grad_output.reshape(input_reshaped.sizes());
    auto std = (cf * (exs - ex * ex) + eps).sqrt();

    auto grad_weight = at::empty_like(weight);
    auto grad_bias = at::empty_like(weight);
    auto grad_ex = at::empty_like(ex);
    auto grad_exs = at::empty_like(exs);

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocks(input_reshaped.size(1));
    int tf = getNumThreads(input_reshaped.size(2));
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "batch_norm_collect_grad_statistics_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        if (at::cuda::detail::canUse32BitIndexMath(input)) {
            batch_norm_collect_grad_statistics_kernel<scalar_t, accscalar_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int32_t>(),
                grad_output_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int32_t>(),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int32_t>(grad_weight),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int32_t>(grad_bias),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int32_t>(grad_ex),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int32_t>(grad_exs),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int32_t>(weight),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int32_t>(ex),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int32_t>(std),
                eps,
                cf);
        } else {
            batch_norm_collect_grad_statistics_kernel<scalar_t, accscalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int64_t>(),
                grad_output_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int64_t>(),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int64_t>(grad_weight),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int64_t>(grad_bias),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int64_t>(grad_ex),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int64_t>(grad_exs),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int64_t>(weight),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int64_t>(ex),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int64_t>(std),
                eps,
                cf);
        }
    }));
    THCudaCheck(cudaGetLastError());
    return {grad_weight, grad_bias, grad_ex, grad_exs};
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_input_backward_kernel(
    const at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> input,
    const at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> grad_output,
    at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t> grad_input,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_ex,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> grad_exs,
    const at::PackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> weight,
    const at::PackedTensorAccessor<accscalar_t, 1, at::DefaultPtrTraits, index_t> save_mean,
    const at::PackedTensorAccessor<accscalar_t, 1, at::DefaultPtrTraits, index_t> save_invstd,
    accscalar_t epsilon) {

    index_t plane = blockIdx.x;
    index_t N = grad_output.size(0) * grad_output.size(2);

    // accscalar_t mean, invstd;
    // mean = save_mean[plane];
    accscalar_t invstd;
    invstd = 1.0 / save_invstd[plane];

    accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
    accscalar_t norm = accscalar_t(1) / N;
    /*
    // Compute two values across (batch, x/y/z) in one pass:
    // 1. Sum(grad_output)
    // 2. DotProduct(input - mean, grad_output)
    GradOp<scalar_t, accscalar_t, at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t>> g(mean, input, grad_output);
    Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t,
                                                                                   at::PackedTensorAccessor<scalar_t, 3, at::DefaultPtrTraits, index_t>>>(g, grad_output, plane);
    accscalar_t grad_output_sum = res.v1;
    accscalar_t dot_p = res.v2;

    accscalar_t grad_mean = grad_output_sum * norm;
    accscalar_t proj_scale = dot_p * norm * invstd * invstd;
    accscalar_t grad_scale = invstd * weight_val;
    */
    if (grad_input.data() != NULL) {
        for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
            for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
                grad_input[batch][plane][x] =
                        static_cast<scalar_t>(grad_output[batch][plane][x] * invstd * weight_val + grad_exs[plane] * 2.0 * input[batch][plane][x] * norm + \
                        grad_ex[plane] * norm);
            }
        }
    }
}

at::Tensor batch_norm_input_backward_cuda(
    const at::Tensor input,
    const at::Tensor grad_output,
    const at::Tensor weight,
    const at::Tensor ex,
    const at::Tensor exs,
    const at::Tensor grad_ex,
    const at::Tensor grad_exs,
    float eps,
    float cf) {
    auto input_reshaped = input.reshape({input.size(0), input.size(1), -1}); // internally we merge the feature dimensions
    auto grad_output_reshaped = grad_output.reshape(input_reshaped.sizes());
    auto std = (cf * (exs - ex * ex) + eps).sqrt();

    auto grad_input = at::empty_like(input);
    auto grad_input_reshaped = grad_input.view(input_reshaped.sizes());

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 blocks(input_reshaped.size(1));
    int tf = getNumThreads(input_reshaped.size(2));
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "batch_norm_input_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        if (at::cuda::detail::canUse32BitIndexMath(input)) {
            batch_norm_input_backward_kernel<scalar_t, accscalar_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int32_t>(),
                grad_output_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int32_t>(),
                packed_accessor_or_dummy<scalar_t, 3, at::DefaultPtrTraits, int32_t>(grad_input_reshaped),
                grad_ex.packed_accessor<scalar_t, 1, at::DefaultPtrTraits, int32_t>(),
                grad_exs.packed_accessor<scalar_t, 1, at::DefaultPtrTraits, int32_t>(),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int32_t>(weight),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int32_t>(ex),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int32_t>(std),
                eps);
        } else {
            batch_norm_input_backward_kernel<scalar_t, accscalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                input_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int64_t>(),
                grad_output_reshaped.packed_accessor<scalar_t, 3, at::DefaultPtrTraits, int64_t>(),
                packed_accessor_or_dummy<scalar_t, 3, at::DefaultPtrTraits, int64_t>(grad_input_reshaped),
                grad_ex.packed_accessor<scalar_t, 1, at::DefaultPtrTraits, int64_t>(),
                grad_exs.packed_accessor<scalar_t, 1, at::DefaultPtrTraits, int64_t>(),
                packed_accessor_or_dummy<scalar_t, 1, at::DefaultPtrTraits, int64_t>(weight),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int64_t>(ex),
                packed_accessor_or_dummy<accscalar_t, 1, at::DefaultPtrTraits, int64_t>(std),
                eps);
        }
    }));
    THCudaCheck(cudaGetLastError());
    return grad_input;
}