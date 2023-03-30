#!/bin/bash

export HIP_PLATFORM=$(eval "${ROCM_HOME}/hip/bin/hipconfig --platform")
TIME_STAMP=$(date +"%Y%m%d_%H_%M")

INPUT_SIZES="500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000"
# replacement for INPUT_SIZES used when testing
INPUT_SIZE="500"

. ./env/bin/activate

cd benchmarks

pip install -r requirements.txt --quiet

# @param -k: only gpu based tools
# @param --benchmark-disable-gc: disable garbage collection during benchmarks
py.test --benchmark-only \
	-k 'cupy or millipyde or opencv_cuda' \
	--rounds 1 \
	--warmup-rounds 0 \
	--benchmark-disable-gc \
	--input-sizes ${INPUT_SIZE} \
	--benchmark-save="${HIP_PLATFORM}_${TIME_STAMP}"

cd ..
