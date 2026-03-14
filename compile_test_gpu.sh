#!/bin/bash
set -euo pipefail

OUT="test_gpu"
SRC="./source/test_gpu.cpp"
LOG="build_nvcc.log"
FADDEEVACC="./source/Faddeeva.cc"
OOUT="printer.o"
FOUT="Faddeeva.o"

# If you ONLY use Eigen on the host (true for this code), disable Eigen's CUDA path:
EIGEN_DEFS="-DEIGEN_NO_CUDA"

# If Eigen still triggers constexpr warnings under nvcc, this relaxes the restriction:
NVCC_EXPT="--expt-relaxed-constexpr"

# Optional: suppress noisy diagnostic ids (keep if you want less spam)
DIAG_SUPPRESS="-diag-suppress 20013 -diag-suppress 20015 -diag-suppress 550 -diag-suppress 177"

# Build
nvcc ${FADDEEVACC} -c -o ${FOUT} -Xcompiler -fopenmp 

nvcc -g -c -O3 -std=c++17 -lineinfo \
  ${NVCC_EXPT} ${EIGEN_DEFS} ${DIAG_SUPPRESS} \
  -Xcompiler -fopenmp \
  -I/usr/include/eigen3 \
  "${SRC}" -o "${OOUT}" \
  -lcublas -lcudart \
  2>&1 | tee "${LOG}"

nvcc -g -O3 "${OOUT}" "${FOUT}" -std=c++17 -lineinfo \
  ${NVCC_EXPT} ${EIGEN_DEFS} ${DIAG_SUPPRESS} \
  -Xcompiler -fopenmp \
  -I/usr/include/eigen3 \
  -o "${OUT}" \
  -lcublas -lcudart \
  2>&1 | tee "${LOG}"

echo "Built: ${OUT}"
echo "Log  : ${LOG}"