#!/bin/bash

# network repo and install 
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation-network

./uninstall_cuda.sh
./uninstall_nvidia.sh

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
# add cuda-keyring
distro=ubuntu2004
arch=x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

sudo apt update
sudo apt install cuda
sudo apt install nvidia-gds

echo "POST INSTALL"
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

./should_reboot.sh

