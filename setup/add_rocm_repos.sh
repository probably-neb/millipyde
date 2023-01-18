#!/bin/bash

# add rocm gpg key
if [ ! -f "./rocm.gpg.key" ]; then
	wget -q -o - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
fi

# replace *_latest with rocm_version for explicit version
#rocm_version=5.5
#amdgpu_latest=latest
#rocm_latest=debian/
#amdgpu_version=$rocm_version
rocm_version=4.5/
amdgpu_base_url="https://repo.radeon.com/amdgpu/${amdgpu_version}/ubuntu"
rocm_base_url="https://repo.radeon.com/rocm/apt/${rocm_version}"

echo "deb [arch=amd64] ${rocm_base_url} ubuntu main" | sudo tee /etc/apt/sources.list.d/rocm.list
echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" | sudo tee /etc/apt/preferences.d/rocm-pin-600

# load new repos
sudo apt update
