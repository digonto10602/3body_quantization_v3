// gpu_varsize_batched_inverse.cu
//
// Build (demo):
//   nvcc -O3 -std=c++17 -DBUILD_DEMO_MAIN -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
//        -Xcompiler -fopenmp -lineinfo -I/usr/include/eigen3 \
//        gpu_varsize_batched_inverse.cu -lcublas -lcudart -o test_varsize_inv
//
// Build (object):
//   nvcc -O3 -std=c++17 -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
//        -Xcompiler -fopenmp -lineinfo -I/usr/include/eigen3 \
//        -c gpu_varsize_batched_inverse.cu -o gpu_varsize_batched_inverse.o

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Eigen/Dense>

#include <omp.h>
#include <vector>
#include <map>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>

// --------------------------- error checks ---------------------------

static inline void throwOnCuda(cudaError_t st, const char* msg, const char* file, int line) {
  if (st != cudaSuccess) {
    std::ostringstream os;
    os << "CUDA error: " << msg << " | " << cudaGetErrorString(st)
       << " @ " << file << ":" << line;
    throw std::runtime_error(os.str());
  }
}
static inline void throwOnCublas(cublasStatus_t st, const char* msg, const char* file, int line) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream os;
    os << "cuBLAS error: " << msg << " | status=" << int(st)
       << " @ " << file << ":" << line;
    throw std::runtime_error(os.str());
  }
}

#define CUDA_CHECK(x, msg)   throwOnCuda((x), (msg), __FILE__, __LINE__)
#define CUBLAS_CHECK(x, msg) throwOnCublas((x), (msg), __FILE__, __LINE__)

static inline void cudaSyncCheck(const char* where) {
  CUDA_CHECK(cudaDeviceSynchronize(), where);
  CUDA_CHECK(cudaGetLastError(), where);
}

// --------------------------- helpers ---------------------------

static inline size_t bytesForOneMatWorstCase(int n) {
  const size_t cplxBytes = sizeof(cuDoubleComplex);
  size_t perMat = 2ull * size_t(n) * size_t(n) * cplxBytes
                + size_t(n) * sizeof(int)
                + sizeof(int);
  return perMat;
}

static inline void packEigenToHostCublas(const Eigen::MatrixXcd& A, cuDoubleComplex* dst) {
  const std::complex<double>* src = A.data();
  const int elems = int(A.size());
  for (int i = 0; i < elems; ++i) {
    dst[i] = make_cuDoubleComplex(src[i].real(), src[i].imag());
  }
}

static inline void unpackHostCublasToEigen(const cuDoubleComplex* src, Eigen::MatrixXcd& A) {
  std::complex<double>* dst = A.data();
  const int elems = int(A.size());
  for (int i = 0; i < elems; ++i) {
    dst[i] = std::complex<double>(cuCreal(src[i]), cuCimag(src[i]));
  }
}

static inline void makeIdentityBatchHost(int n, int batchCount, std::vector<cuDoubleComplex>& Bhost) {
  Bhost.assign(size_t(batchCount) * size_t(n) * size_t(n), make_cuDoubleComplex(0.0, 0.0));
  for (int b = 0; b < batchCount; ++b) {
    cuDoubleComplex* Bb = Bhost.data() + size_t(b) * size_t(n) * size_t(n);
    for (int i = 0; i < n; ++i) Bb[i + i * n] = make_cuDoubleComplex(1.0, 0.0);
  }
}

// --------------------------- GPU core: same-size group inversion ---------------------------
//
// getrfBatched: infoArray is DEVICE -> dInfo_getrf
// getrsBatched: info is HOST        -> hInfo_getrs
//
static void invertSameSizeGroup_cublasBatched(
    cublasHandle_t handle,
    int n,
    const std::vector<Eigen::MatrixXcd>& A_group,
    std::vector<Eigen::MatrixXcd>& Ainv_group_out)
{
  const int batchCount = int(A_group.size());
  if (batchCount == 0) return;

  // Host packed buffers
  std::vector<cuDoubleComplex> Ahost(size_t(batchCount) * size_t(n) * size_t(n));
  std::vector<cuDoubleComplex> Bhost;
  makeIdentityBatchHost(n, batchCount, Bhost);

  for (int b = 0; b < batchCount; ++b) {
    const auto& A = A_group[b];
    if (A.rows() != n || A.cols() != n) throw std::runtime_error("Size-group contains wrong-sized matrix.");
    packEigenToHostCublas(A, Ahost.data() + size_t(b) * size_t(n) * size_t(n));
  }

  // Device allocations
  cuDoubleComplex* dA = nullptr;
  cuDoubleComplex* dB = nullptr;
  int* dPivots = nullptr;       // device
  int* dInfo_getrf = nullptr;   // device
  std::vector<int> hInfo_getrf(batchCount, 0);
  std::vector<int> hInfo_getrs(batchCount, 0); // host

  CUDA_CHECK(cudaMalloc((void**)&dA, Ahost.size() * sizeof(cuDoubleComplex)), "cudaMalloc dA");
  CUDA_CHECK(cudaMalloc((void**)&dB, Bhost.size() * sizeof(cuDoubleComplex)), "cudaMalloc dB");
  CUDA_CHECK(cudaMalloc((void**)&dPivots, size_t(batchCount) * size_t(n) * sizeof(int)), "cudaMalloc dPivots");
  CUDA_CHECK(cudaMalloc((void**)&dInfo_getrf, size_t(batchCount) * sizeof(int)), "cudaMalloc dInfo_getrf");

  CUDA_CHECK(cudaMemcpy(dA, Ahost.data(), Ahost.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D A");
  CUDA_CHECK(cudaMemcpy(dB, Bhost.data(), Bhost.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D B");
  CUDA_CHECK(cudaMemset(dInfo_getrf, 0, size_t(batchCount) * sizeof(int)), "memset dInfo_getrf");

  // Device pointer arrays
  cuDoubleComplex** dA_array = nullptr;
  cuDoubleComplex** dB_array = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&dA_array, size_t(batchCount) * sizeof(cuDoubleComplex*)), "cudaMalloc dA_array");
  CUDA_CHECK(cudaMalloc((void**)&dB_array, size_t(batchCount) * sizeof(cuDoubleComplex*)), "cudaMalloc dB_array");

  std::vector<cuDoubleComplex*> hAptr(batchCount), hBptr(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    hAptr[b] = dA + size_t(b) * size_t(n) * size_t(n);
    hBptr[b] = dB + size_t(b) * size_t(n) * size_t(n);
  }
  CUDA_CHECK(cudaMemcpy(dA_array, hAptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D A ptrs");
  CUDA_CHECK(cudaMemcpy(dB_array, hBptr.data(), size_t(batchCount) * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice), "H2D B ptrs");

  // LU
  CUBLAS_CHECK(cublasZgetrfBatched(handle, n, dA_array, n, dPivots, dInfo_getrf, batchCount),
               "cublasZgetrfBatched");
  cudaSyncCheck("after cublasZgetrfBatched");

  CUDA_CHECK(cudaMemcpy(hInfo_getrf.data(), dInfo_getrf, size_t(batchCount) * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "D2H info_getrf");
  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrf[b] != 0) {
      std::ostringstream os;
      os << "LU failed: info[" << b << "]=" << hInfo_getrf[b]
         << " (singular if >0, illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Solve A*X=I (B overwritten with X)
  std::fill(hInfo_getrs.begin(), hInfo_getrs.end(), 0);
  CUBLAS_CHECK(cublasZgetrsBatched(handle, CUBLAS_OP_N, n, n,
                                   (const cuDoubleComplex**)dA_array, n,
                                   dPivots,
                                   dB_array, n,
                                   hInfo_getrs.data(), batchCount),
               "cublasZgetrsBatched");
  cudaSyncCheck("after cublasZgetrsBatched");

  for (int b = 0; b < batchCount; ++b) {
    if (hInfo_getrs[b] != 0) {
      std::ostringstream os;
      os << "Solve failed: info[" << b << "]=" << hInfo_getrs[b]
         << " (illegal param if <0).";
      throw std::runtime_error(os.str());
    }
  }

  // Copy back
  std::vector<cuDoubleComplex> Xhost(Bhost.size());
  CUDA_CHECK(cudaMemcpy(Xhost.data(), dB, Xhost.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H X");

  Ainv_group_out.resize(batchCount);
  for (int b = 0; b < batchCount; ++b) {
    Ainv_group_out[b].resize(n, n);
    unpackHostCublasToEigen(Xhost.data() + size_t(b) * size_t(n) * size_t(n), Ainv_group_out[b]);
  }

  // Cleanup
  cudaFree(dA_array);
  cudaFree(dB_array);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dPivots);
  cudaFree(dInfo_getrf);
}

// --------------------------- High-level pipeline ---------------------------

// Drop-in change: replace the std::function-based API with a templated Builder API.
// Everything else (chunking, grouping, cuBLAS inversion core) remains identical.
//
// Usage stays the same for lambdas/functions, but now it inlines and avoids std::function overhead.
//
// Example call:
//   auto buildA = [&](int i, double En, Eigen::MatrixXcd& A){ ... };
//   build_and_invert_energy_sweep_varsize_batched_gpu(En_i, En_f, N, buildA, Ainv, 0.8, 0);

#include <type_traits>
#include <utility>

// ---- replace ONLY this function signature+body in your file ----

// NOTE: You can keep the same function name.
// We template on Builder and forward it.
// Requirement: build_A(i, En, A) must be a valid expression returning void (or convertible to void).
template <class Builder>
void build_and_invert_energy_sweep_varsize_batched_gpu(
    double En_initial,
    double En_final,
    int total_problem_size,
    Builder&& build_A,                         // <--- templated callable
    std::vector<Eigen::MatrixXcd>& Ainv_vec,
    double vram_safety_fraction = 0.80,
    int omp_threads = 0)
{
  // Optional compile-time check (gives clearer errors if signature mismatches)
  static_assert(
      std::is_invocable_v<Builder&, int, double, Eigen::MatrixXcd&>,
      "Builder must be callable as: build_A(int i, double En, Eigen::MatrixXcd& A)");
  // If you want to enforce void return:
  // static_assert(std::is_void_v<std::invoke_result_t<Builder&, int, double, Eigen::MatrixXcd&>>,
  //              "Builder must return void");

  if (total_problem_size <= 0) throw std::runtime_error("total_problem_size must be > 0");
  if (!(vram_safety_fraction > 0.0 && vram_safety_fraction <= 0.95))
    throw std::runtime_error("vram_safety_fraction should be in (0, 0.95]");

  if (omp_threads > 0) omp_set_num_threads(omp_threads);

  const double del_En = (total_problem_size == 1)
                        ? 0.0
                        : (En_final - En_initial) / double(total_problem_size - 1);

  Ainv_vec.assign(size_t(total_problem_size), Eigen::MatrixXcd());

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle), "cublasCreate");

  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
  const size_t budget = size_t(double(freeB) * vram_safety_fraction);

  const int blockGrow = 64;

  int start = 0;
  while (start < total_problem_size) {
    int end = start;

    std::vector<Eigen::MatrixXcd> Avec_chunk;
    std::vector<int> n_chunk;

    while (end < total_problem_size) {
      const int trialEnd = std::min(total_problem_size, end + blockGrow);
      const int trialCount = trialEnd - start;

      // NVCC-safe sizing
      const size_t trialCountSz = static_cast<size_t>(trialCount);

      std::vector<Eigen::MatrixXcd> Avec_trial(trialCountSz);
      std::vector<int>              n_trial(trialCountSz, -1);

      // IMPORTANT: capture build_A by reference safely, even though Builder is a forwarding ref.
      // We create a reference alias to avoid accidental copies in OpenMP region.
      auto& buildA_ref = build_A;

      #pragma omp parallel for schedule(dynamic)
      for (int k = 0; k < trialCount; ++k) {
        const int i = start + k;
        const double En = En_initial + double(i) * del_En;

        Eigen::MatrixXcd A;
        buildA_ref(i, En, A);

        if (A.rows() == A.cols() && A.rows() > 0) {
          n_trial[size_t(k)] = int(A.rows());
          Avec_trial[size_t(k)] = std::move(A);
        }
      }

      for (int k = 0; k < trialCount; ++k) {
        if (n_trial[size_t(k)] <= 0) {
          cublasDestroy(handle);
          throw std::runtime_error("build_A produced an invalid (non-square or empty) matrix.");
        }
      }

      size_t est = 0;
      for (int k = 0; k < trialCount; ++k) est += bytesForOneMatWorstCase(n_trial[size_t(k)]);
      est += 32ull * 1024ull * 1024ull;

      if (est <= budget || trialCount == 1) {
        Avec_chunk.swap(Avec_trial);
        n_chunk.swap(n_trial);
        end = trialEnd;
        if (est > budget && trialCount == 1) break;
      } else {
        break;
      }
    }

    const int chunkCount = end - start;
    if (chunkCount <= 0) {
      cublasDestroy(handle);
      throw std::runtime_error("Chunking failure: chunkCount <= 0");
    }

    std::map<int, std::vector<int>> groups;
    for (int k = 0; k < chunkCount; ++k) groups[n_chunk[size_t(k)]].push_back(k);

    for (const auto& kv : groups) {
      const int n = kv.first;
      const auto& locs = kv.second;

      std::vector<Eigen::MatrixXcd> A_group;
      A_group.reserve(locs.size());
      for (int local : locs) A_group.push_back(Avec_chunk[size_t(local)]);

      std::vector<Eigen::MatrixXcd> Ainv_group;
      invertSameSizeGroup_cublasBatched(handle, n, A_group, Ainv_group);

      for (size_t t = 0; t < locs.size(); ++t) {
        const int global_i = start + locs[t];
        Ainv_vec[size_t(global_i)] = std::move(Ainv_group[t]);
      }
    }

    start = end;
  }

  cublasDestroy(handle);
}


// --------------------------- Demo main ---------------------------
#ifdef BUILD_DEMO_MAIN
int main() {
  try {
    const double En_i = 0.1;
    const double En_f = 1.0;
    const int totalN = 20000;

    auto buildA = [](int i, double En, Eigen::MatrixXcd& A) {
      int n = 64 + (i % 3) * 16; // 64, 80, 96
      A.resize(n, n);
      A.setZero();
      for (int r = 0; r < n; ++r) {
        A(r, r) = std::complex<double>(2.0 + 0.1 * En, 0.0);
        if (r + 1 < n) A(r, r + 1) = std::complex<double>(0.01, 0.02);
        if (r - 1 >= 0) A(r, r - 1) = std::complex<double>(0.01, -0.02);
      }
    };

    std::vector<Eigen::MatrixXcd> Ainv;
    build_and_invert_energy_sweep_varsize_batched_gpu(
      En_i, En_f, totalN, buildA, Ainv, 0.80, 0
    );

    std::cout << "Done. Ainv size = " << Ainv.size() << "\n";
    for( int i=0; i<Ainv.size(); ++i)
    {
      std::cout << "Ainv[" << i << "] dims = " << Ainv[i].rows() << "x" << Ainv[i].cols() << "\n";

    }
    std::cout<<std::endl; 
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
#endif