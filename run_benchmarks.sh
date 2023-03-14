#!/bin/bash

. ./env/bin/activate
pip install -r ./benchmarking-requirements.txt --quiet
py.test --benchmark-only
