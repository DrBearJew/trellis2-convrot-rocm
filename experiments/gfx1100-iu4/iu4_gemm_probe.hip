#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                                    \
    do {                                                                                   \
        const hipError_t error__ = (expr);                                                  \
        if (error__ != hipSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__,             \
                         hipGetErrorString(error__));                                       \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (0)

using v2i32 = int __attribute__((ext_vector_type(2)));
using v4i32 = int __attribute__((ext_vector_type(4)));
using v8i32 = int __attribute__((ext_vector_type(8)));

static constexpr int kTile = 16;

static __device__ __forceinline__ v8i32 wmma_i4_signed(v2i32 a, v2i32 b, v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
}

static __device__ __forceinline__ v8i32 wmma_i8_signed(v4i32 a, v4i32 b, v8i32 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

static __device__ __forceinline__ int dot4_i8_signed(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

static __device__ __forceinline__ int load_i4_nibble(const uint8_t* row, int k) {
    return (row[k >> 1] >> (4 * (k & 1))) & 0x0f;
}

static __device__ __forceinline__ int pack_i4_tail_word(
        const uint8_t* row, int k0, int valid_k) {
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = k0 + i;
        const uint32_t nibble = k < valid_k ? uint32_t(load_i4_nibble(row, k)) : 0u;
        word |= nibble << (4 * i);
    }
    return int(word);
}

static __device__ __forceinline__ int pack_i8_tail_word(
        const int8_t* row, int k0, int valid_k) {
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = k0 + i;
        const uint32_t byte = k < valid_k ? uint32_t(uint8_t(row[k])) : 0u;
        word |= byte << (8 * i);
    }
    return int(word);
}

extern "C" __global__ __launch_bounds__(32) void iu4_gemm_candidate(
        const uint8_t* __restrict__ a,
        const uint8_t* __restrict__ b,
        int32_t* __restrict__ c,
        int m,
        int n,
        int k,
        int a_stride_bytes,
        int b_stride_bytes) {
    const int lane = int(threadIdx.x);
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int tile_m = int(blockIdx.y) * kTile;
    const int tile_n = int(blockIdx.x) * kTile;
    const int a_row = tile_m + lane_lo;
    const int b_row = tile_n + lane_lo;

    v8i32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < k; k0 += kTile) {
        v2i32 a_frag = {0, 0};
        v2i32 b_frag = {0, 0};
        if (a_row < m) {
            const uint8_t* row = a + size_t(a_row) * size_t(a_stride_bytes);
            if (k0 + kTile <= k) {
                const uint32_t* words = reinterpret_cast<const uint32_t*>(row + (k0 >> 1));
                a_frag[0] = int(words[0]);
                a_frag[1] = int(words[1]);
            } else {
                a_frag[0] = pack_i4_tail_word(row, k0, k);
                a_frag[1] = pack_i4_tail_word(row, k0 + 8, k);
            }
        }
        if (b_row < n) {
            const uint8_t* row = b + size_t(b_row) * size_t(b_stride_bytes);
            if (k0 + kTile <= k) {
                const uint32_t* words = reinterpret_cast<const uint32_t*>(row + (k0 >> 1));
                b_frag[0] = int(words[0]);
                b_frag[1] = int(words[1]);
            } else {
                b_frag[0] = pack_i4_tail_word(row, k0, k);
                b_frag[1] = pack_i4_tail_word(row, k0 + 8, k);
            }
        }
        acc = wmma_i4_signed(a_frag, b_frag, acc);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = tile_m + 2 * i + lane_hi;
        const int col = tile_n + lane_lo;
        if (row < m && col < n) {
            c[size_t(row) * size_t(n) + size_t(col)] = acc[i];
        }
    }
}

extern "C" __global__ __launch_bounds__(32) void iu8_gemm_control(
        const int8_t* __restrict__ a,
        const int8_t* __restrict__ b,
        int32_t* __restrict__ c,
        int m,
        int n,
        int k,
        int a_stride_bytes,
        int b_stride_bytes) {
    const int lane = int(threadIdx.x);
    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int tile_m = int(blockIdx.y) * kTile;
    const int tile_n = int(blockIdx.x) * kTile;
    const int a_row = tile_m + lane_lo;
    const int b_row = tile_n + lane_lo;

    v8i32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < k; k0 += kTile) {
        v4i32 a_frag = {0, 0, 0, 0};
        v4i32 b_frag = {0, 0, 0, 0};
        if (a_row < m) {
            const int8_t* row = a + size_t(a_row) * size_t(a_stride_bytes);
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                if (k0 + kTile <= k) {
                    a_frag[word] = reinterpret_cast<const int32_t*>(row + k0)[word];
                } else {
                    a_frag[word] = pack_i8_tail_word(row, k0 + 4 * word, k);
                }
            }
        }
        if (b_row < n) {
            const int8_t* row = b + size_t(b_row) * size_t(b_stride_bytes);
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                if (k0 + kTile <= k) {
                    b_frag[word] = reinterpret_cast<const int32_t*>(row + k0)[word];
                } else {
                    b_frag[word] = pack_i8_tail_word(row, k0 + 4 * word, k);
                }
            }
        }
        acc = wmma_i8_signed(a_frag, b_frag, acc);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = tile_m + 2 * i + lane_hi;
        const int col = tile_n + lane_lo;
        if (row < m && col < n) {
            c[size_t(row) * size_t(n) + size_t(col)] = acc[i];
        }
    }
}

extern "C" __global__ void dot4_i8_control(
        const int8_t* __restrict__ a,
        const int8_t* __restrict__ b,
        int32_t* __restrict__ c,
        int m,
        int n,
        int k,
        int a_stride_bytes,
        int b_stride_bytes) {
    const int col = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int row = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
    if (row >= m || col >= n) {
        return;
    }
    const int8_t* a_row = a + size_t(row) * size_t(a_stride_bytes);
    const int8_t* b_row = b + size_t(col) * size_t(b_stride_bytes);
    int acc = 0;
    for (int k0 = 0; k0 < k; k0 += 4) {
        const int a_word = k0 + 4 <= k
            ? *reinterpret_cast<const int32_t*>(a_row + k0)
            : pack_i8_tail_word(a_row, k0, k);
        const int b_word = k0 + 4 <= k
            ? *reinterpret_cast<const int32_t*>(b_row + k0)
            : pack_i8_tail_word(b_row, k0, k);
        acc = dot4_i8_signed(a_word, b_word, acc);
    }
    c[size_t(row) * size_t(n) + size_t(col)] = acc;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t size) : bytes(size) { HIP_CHECK(hipMalloc(&ptr, size)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
        other.ptr = nullptr;
        other.bytes = 0;
    }
    ~DeviceBuffer() {
        if (ptr) {
            (void)hipFree(ptr);
        }
    }
};

struct HostMatrices {
    int m;
    int n;
    int k;
    int i4_stride;
    int i8_stride;
    std::vector<int8_t> a_values;
    std::vector<int8_t> b_values;
    std::vector<uint8_t> a_i4;
    std::vector<uint8_t> b_i4;
    std::vector<int8_t> a_i8;
    std::vector<int8_t> b_i8;
};

static int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static std::vector<uint8_t> pack_i4_rows(
        const std::vector<int8_t>& values, int rows, int k, int stride) {
    std::vector<uint8_t> packed(size_t(rows) * size_t(stride), 0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < k; ++col) {
            const uint8_t nibble = uint8_t(values[size_t(row) * size_t(k) + size_t(col)]) & 0x0f;
            packed[size_t(row) * size_t(stride) + size_t(col >> 1)] |=
                uint8_t(nibble << (4 * (col & 1)));
        }
    }
    return packed;
}

static std::vector<int8_t> pad_i8_rows(
        const std::vector<int8_t>& values, int rows, int k, int stride) {
    std::vector<int8_t> padded(size_t(rows) * size_t(stride), 0);
    for (int row = 0; row < rows; ++row) {
        std::memcpy(padded.data() + size_t(row) * size_t(stride),
                    values.data() + size_t(row) * size_t(k), size_t(k));
    }
    return padded;
}

static HostMatrices make_matrices(int m, int n, int k, uint32_t seed) {
    HostMatrices host{};
    host.m = m;
    host.n = n;
    host.k = k;
    host.i4_stride = round_up(k, 16) / 2;
    host.i8_stride = round_up(k, 16);
    host.a_values.resize(size_t(m) * size_t(k));
    host.b_values.resize(size_t(n) * size_t(k));

    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-7, 7);
    for (int8_t& value : host.a_values) {
        value = int8_t(distribution(generator));
    }
    for (int8_t& value : host.b_values) {
        value = int8_t(distribution(generator));
    }
    host.a_i4 = pack_i4_rows(host.a_values, m, k, host.i4_stride);
    host.b_i4 = pack_i4_rows(host.b_values, n, k, host.i4_stride);
    host.a_i8 = pad_i8_rows(host.a_values, m, k, host.i8_stride);
    host.b_i8 = pad_i8_rows(host.b_values, n, k, host.i8_stride);
    return host;
}

static std::vector<int32_t> cpu_reference(const HostMatrices& host) {
    std::vector<int32_t> output(size_t(host.m) * size_t(host.n), 0);
    for (int row = 0; row < host.m; ++row) {
        for (int col = 0; col < host.n; ++col) {
            int32_t sum = 0;
            for (int inner = 0; inner < host.k; ++inner) {
                sum += int32_t(host.a_values[size_t(row) * size_t(host.k) + size_t(inner)]) *
                       int32_t(host.b_values[size_t(col) * size_t(host.k) + size_t(inner)]);
            }
            output[size_t(row) * size_t(host.n) + size_t(col)] = sum;
        }
    }
    return output;
}

static dim3 grid_for(int m, int n) {
    return dim3(unsigned(round_up(n, 16) / 16), unsigned(round_up(m, 16) / 16), 1);
}

static void launch_i4(const DeviceBuffer& a, const DeviceBuffer& b, DeviceBuffer& c,
                      int m, int n, int k, int stride, hipStream_t stream = nullptr) {
    hipLaunchKernelGGL(iu4_gemm_candidate, grid_for(m, n), dim3(32), 0, stream,
                       static_cast<const uint8_t*>(a.ptr), static_cast<const uint8_t*>(b.ptr),
                       static_cast<int32_t*>(c.ptr), m, n, k, stride, stride);
}

static void launch_i8(const DeviceBuffer& a, const DeviceBuffer& b, DeviceBuffer& c,
                      int m, int n, int k, int stride, hipStream_t stream = nullptr) {
    hipLaunchKernelGGL(iu8_gemm_control, grid_for(m, n), dim3(32), 0, stream,
                       static_cast<const int8_t*>(a.ptr), static_cast<const int8_t*>(b.ptr),
                       static_cast<int32_t*>(c.ptr), m, n, k, stride, stride);
}

static void launch_dot4_i8(const DeviceBuffer& a, const DeviceBuffer& b, DeviceBuffer& c,
                           int m, int n, int k, int stride, hipStream_t stream = nullptr) {
    const dim3 block(16, 16, 1);
    const dim3 grid(unsigned(round_up(n, int(block.x)) / int(block.x)),
                    unsigned(round_up(m, int(block.y)) / int(block.y)), 1);
    hipLaunchKernelGGL(dot4_i8_control, grid, block, 0, stream,
                       static_cast<const int8_t*>(a.ptr), static_cast<const int8_t*>(b.ptr),
                       static_cast<int32_t*>(c.ptr), m, n, k, stride, stride);
}

static bool compare_outputs(const char* label, const std::vector<int32_t>& got,
                            const std::vector<int32_t>& expected) {
    size_t mismatches = 0;
    int32_t max_abs_error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const int64_t delta = int64_t(got[i]) - int64_t(expected[i]);
        max_abs_error = std::max(max_abs_error, int32_t(std::llabs(delta)));
        if (delta != 0) {
            if (mismatches < 8) {
                std::fprintf(stderr, "%s mismatch[%zu]: got=%d expected=%d\n",
                             label, i, got[i], expected[i]);
            }
            ++mismatches;
        }
    }
    std::printf("correctness format=%s elements=%zu mismatches=%zu max_abs_error=%d\n",
                label, got.size(), mismatches, max_abs_error);
    return mismatches == 0;
}

static bool run_correctness_case(int m, int n, int k, uint32_t seed) {
    const HostMatrices host = make_matrices(m, n, k, seed);
    const std::vector<int32_t> reference = cpu_reference(host);
    DeviceBuffer a_i4(host.a_i4.size());
    DeviceBuffer b_i4(host.b_i4.size());
    DeviceBuffer a_i8(host.a_i8.size());
    DeviceBuffer b_i8(host.b_i8.size());
    DeviceBuffer c(size_t(m) * size_t(n) * sizeof(int32_t));
    HIP_CHECK(hipMemcpy(a_i4.ptr, host.a_i4.data(), host.a_i4.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_i4.ptr, host.b_i4.data(), host.b_i4.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(a_i8.ptr, host.a_i8.data(), host.a_i8.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_i8.ptr, host.b_i8.data(), host.b_i8.size(), hipMemcpyHostToDevice));

    std::vector<int32_t> got(reference.size());
    HIP_CHECK(hipMemset(c.ptr, 0xa5, c.bytes));
    launch_i4(a_i4, b_i4, c, m, n, k, host.i4_stride);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(got.data(), c.ptr, c.bytes, hipMemcpyDeviceToHost));
    const bool i4_ok = compare_outputs("iu4", got, reference);

    HIP_CHECK(hipMemset(c.ptr, 0xa5, c.bytes));
    launch_i8(a_i8, b_i8, c, m, n, k, host.i8_stride);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(got.data(), c.ptr, c.bytes, hipMemcpyDeviceToHost));
    const bool i8_ok = compare_outputs("iu8", got, reference);

    HIP_CHECK(hipMemset(c.ptr, 0xa5, c.bytes));
    launch_dot4_i8(a_i8, b_i8, c, m, n, k, host.i8_stride);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(got.data(), c.ptr, c.bytes, hipMemcpyDeviceToHost));
    const bool dot4_ok = compare_outputs("dot4_i8", got, reference);
    return i4_ok && i8_ok && dot4_ok;
}

static double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() & 1) {
        return values[middle];
    }
    return 0.5 * (double(values[middle - 1]) + double(values[middle]));
}

template <typename Launch>
static double time_kernel(Launch launch, int warmups, int iterations, int repeats) {
    for (int i = 0; i < warmups; ++i) {
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> samples;
    samples.reserve(size_t(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        HIP_CHECK(hipEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch();
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(elapsed_ms / float(iterations));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return median(samples);
}

static void run_benchmark_case(int m, int n, int k, int iterations, int repeats) {
    const HostMatrices host = make_matrices(m, n, k, 0x51a4u);
    DeviceBuffer a_i4(host.a_i4.size());
    DeviceBuffer b_i4(host.b_i4.size());
    DeviceBuffer a_i8(host.a_i8.size());
    DeviceBuffer b_i8(host.b_i8.size());
    DeviceBuffer c(size_t(m) * size_t(n) * sizeof(int32_t));
    HIP_CHECK(hipMemcpy(a_i4.ptr, host.a_i4.data(), host.a_i4.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_i4.ptr, host.b_i4.data(), host.b_i4.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(a_i8.ptr, host.a_i8.data(), host.a_i8.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_i8.ptr, host.b_i8.data(), host.b_i8.size(), hipMemcpyHostToDevice));

    const double i4_ms = time_kernel(
        [&] { launch_i4(a_i4, b_i4, c, m, n, k, host.i4_stride); }, 3, iterations, repeats);
    HIP_CHECK(hipGetLastError());
    const double i8_ms = time_kernel(
        [&] { launch_i8(a_i8, b_i8, c, m, n, k, host.i8_stride); }, 3, iterations, repeats);
    HIP_CHECK(hipGetLastError());
    const double dot4_i8_ms = time_kernel(
        [&] { launch_dot4_i8(a_i8, b_i8, c, m, n, k, host.i8_stride); }, 3, iterations, repeats);
    HIP_CHECK(hipGetLastError());

    const double operations = 2.0 * double(m) * double(n) * double(k);
    const double i4_tops = operations / (i4_ms * 1.0e9);
    const double i8_tops = operations / (i8_ms * 1.0e9);
    const double dot4_i8_tops = operations / (dot4_i8_ms * 1.0e9);
    std::printf(
        "benchmark m=%d n=%d k=%d iu4_ms=%.6f iu8_wmma_ms=%.6f dot4_i8_ms=%.6f "
        "iu4_vs_iu8_speedup=%.4f iu4_vs_dot4_speedup=%.4f "
        "iu4_tops=%.4f iu8_wmma_tops=%.4f dot4_i8_tops=%.4f iterations=%d repeats=%d\n",
        m, n, k, i4_ms, i8_ms, dot4_i8_ms, i8_ms / i4_ms, dot4_i8_ms / i4_ms,
        i4_tops, i8_tops, dot4_i8_tops, iterations, repeats);
}

static void usage(const char* program) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --correctness\n"
        "  %s --shape M N K [--iterations I] [--repeats R]\n"
        "  %s --trellis-shapes [--iterations I] [--repeats R]\n",
        program, program, program);
}

int main(int argc, char** argv) {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    std::printf("device name=%s arch=%s total_global_mem=%zu\n",
                properties.name, properties.gcnArchName, size_t(properties.totalGlobalMem));
    if (std::strncmp(properties.gcnArchName, "gfx1100", 7) != 0) {
        std::fprintf(stderr, "This probe is restricted to gfx1100; found %s\n", properties.gcnArchName);
        return 2;
    }

    bool correctness = false;
    bool trellis_shapes = false;
    bool have_shape = false;
    int m = 0;
    int n = 0;
    int k = 0;
    int iterations = 10;
    int repeats = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--correctness") {
            correctness = true;
        } else if (argument == "--trellis-shapes") {
            trellis_shapes = true;
        } else if (argument == "--shape" && i + 3 < argc) {
            m = std::atoi(argv[++i]);
            n = std::atoi(argv[++i]);
            k = std::atoi(argv[++i]);
            have_shape = true;
        } else if (argument == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (argument == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (iterations <= 0 || repeats <= 0 || m < 0 || n < 0 || k < 0) {
        usage(argv[0]);
        return 2;
    }

    bool ok = true;
    if (correctness) {
        ok = run_correctness_case(16, 16, 16, 1u) && ok;
        ok = run_correctness_case(32, 48, 64, 2u) && ok;
        ok = run_correctness_case(31, 29, 47, 3u) && ok;
    }
    if (have_shape) {
        if (m <= 0 || n <= 0 || k <= 0) {
            usage(argv[0]);
            return 2;
        }
        run_benchmark_case(m, n, k, iterations, repeats);
    }
    if (trellis_shapes) {
        const int shapes[][3] = {
            {4096, 1536, 1536},
            {8398, 1536, 8192},
            {8398, 8192, 1536},
            {2875, 1536, 8192},
            {2875, 8192, 1536},
        };
        for (const auto& shape : shapes) {
            run_benchmark_case(shape[0], shape[1], shape[2], iterations, repeats);
        }
    }
    if (!correctness && !have_shape && !trellis_shapes) {
        usage(argv[0]);
        return 2;
    }
    return ok ? 0 : 1;
}
