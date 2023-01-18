#!/bin/bash

# 5
# https://docs.amd.com/bundle/HIP-Installation-Guide-v5.3/page/Introduction_to_HIP_Installation_Guide.html

# 4.5
# https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html#understanding-amdgpu-and-rocm-stack-repositories-on-linux-distributions

#./install_cuda.sh
./add_rocm_repos.sh
sudo apt install hip-runtime-nvidia hip-dev

# verify
/opt/rocm-4.5.0/bin/hipconfig --full
