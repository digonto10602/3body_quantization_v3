#!/bin/bash
set -euo pipefail

OUT="test_varsize_inv"
SRC="gpu_varsize_batched_inverse.cu"
LOG="build_nvcc.log"

# If you ONLY use Eigen on the host (true for this code), disable Eigen's CUDA path:
EIGEN_DEFS="-DEIGEN_NO_CUDA"

# If Eigen still triggers constexpr warnings under nvcc, this relaxes the restriction:
NVCC_EXPT="--expt-relaxed-constexpr"

# Optional: suppress noisy diagnostic ids (keep if you want less spam)
DIAG_SUPPRESS="-diag-suppress 20013 -diag-suppress 20015 -diag-suppress 550 -diag-suppress 177"

# Build
nvcc -O3 -std=c++17 -DBUILD_DEMO_MAIN -lineinfo \
  ${NVCC_EXPT} ${EIGEN_DEFS} ${DIAG_SUPPRESS} \
  -Xcompiler -fopenmp \
  -I/usr/include/eigen3 \
  "${SRC}" -o "${OUT}" \
  -lcublas -lcudart \
  2>&1 | tee "${LOG}"

echo "Built: ${OUT}"
echo "Log  : ${LOG}"