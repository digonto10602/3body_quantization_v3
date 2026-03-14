#!/usr/bin/env bash
set -euo pipefail

OUT="test_gpu"
SRC="./source/test_gpu.cpp"
FADDEEVACC="./source/Faddeeva.cc"

OBJ_MAIN="test_gpu.o"
OBJ_FAD="Faddeeva.o"

LOG="build_nvcc.log"

# -------- Modes --------
# Usage:
#   ./build.sh perf
#   ./build.sh debug
MODE="${1:-perf}"

# -------- Common flags --------
EIGEN_DEFS="-DEIGEN_NO_CUDA"
NVCC_EXPT="--expt-relaxed-constexpr"
DIAG_SUPPRESS="-diag-suppress 20013 -diag-suppress 20015 -diag-suppress 550 -diag-suppress 177"

INCLUDES="-I/usr/include/eigen3"
OMP_FLAGS="-Xcompiler -fopenmp"

# -------- Mode-dependent flags --------
# perf: what you want for Nsight Systems/Compute
# debug: heavier symbols; still avoid -G unless debugging kernels
if [[ "$MODE" == "perf" ]]; then
  NVCC_FLAGS="-O3 -g -lineinfo -std=c++17"
  CXX_FLAGS="-O3 -g -std=c++17"
elif [[ "$MODE" == "debug" ]]; then
  NVCC_FLAGS="-O0 -g -lineinfo -std=c++17"
  CXX_FLAGS="-O0 -g -std=c++17"
else
  echo "Unknown mode: $MODE (use perf|debug)"
  exit 1
fi

# -------- Clean log --------
: > "${LOG}"

echo "[1/3] Compile Faddeeva (CPU) -> ${OBJ_FAD}" | tee -a "${LOG}"
g++ ${CXX_FLAGS} -c "${FADDEEVACC}" -o "${OBJ_FAD}" -fopenmp 2>&1 | tee -a "${LOG}"

echo "[2/3] Compile main with nvcc -> ${OBJ_MAIN}" | tee -a "${LOG}"
nvcc -c ${NVCC_FLAGS} \
  ${NVCC_EXPT} ${EIGEN_DEFS} ${DIAG_SUPPRESS} \
  ${OMP_FLAGS} ${INCLUDES} \
  "${SRC}" -o "${OBJ_MAIN}" \
  2>&1 | tee -a "${LOG}"

echo "[3/3] Link -> ${OUT}" | tee -a "${LOG}"
nvcc ${NVCC_FLAGS} \
  ${NVCC_EXPT} ${EIGEN_DEFS} ${DIAG_SUPPRESS} \
  ${OMP_FLAGS} ${INCLUDES} \
  "${OBJ_MAIN}" "${OBJ_FAD}" -o "${OUT}" \
  -lcublas -lcudart \
  2>&1 | tee -a "${LOG}"

echo "Built: ${OUT}"
echo "Log  : ${LOG}"