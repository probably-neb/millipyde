#!/bin/bash

. ./env/bin/activate
pip install -r ./benchmarking-requirements.txt --quiet
cd benchmarks
python bench.py
