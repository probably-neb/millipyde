#!/bin/bash

HIP_PLATFORM=$(eval "${ROCM_HOME}/hip/bin/hipconfig --platform")
TIME_STAMP=$(date +"%Y%m%d_%H_%M")

INPUT_SIZES="500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000"

. ./env/bin/activate

cd benchmarks

pip install -r /requirements.txt --quiet
py.test --benchmark-only \
	-k 'cupy or millipyde or opencv_cuda' \
	--rounds 1 \
	--warmup-rounds 10 \
	--benchmark-disable-gc \
	--input-sizes ${INPUT_SIZES} \
	--benchmark-save="${HIP_PLATFORM}_${TIME_STAMP}"

cd ..
