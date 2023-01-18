#!/bin/bash

#Prerequisites
# For the AMD platform, see Prerequisite Actions in the ROCm Installation Guide at https://docs.amd.com.
# For details about the NVIDIA platform, check the system requirements in the NVIDIA CUDA Installation Guide at https://docs.nvidia.com/cuda/cuda-installation-guide-linux.

######################################
# Installing HIP on the AMD Platform #
######################################

# Install ROCm packages or install prebuilt binary packages using the package manager. For details on ROCm installation, refer to the ROCm Installation Guide at, https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.3/page/Introduction_to_ROCm_Installation_Guide_for_Linux.html

# By default, HIP is installed in /opt/rocm/hip.

#########################################
# Installing HIP on the NVIDIA Platform #
#########################################

# Install the NVIDIA driver and prebuilt packages. Follow these steps:
  
# 1.    Ensure you install the NVIDIA driver using the following instructions:
  
# sudo apt-get install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall
# sudo reboot
  
# Or you may download the latest CUDA-toolkit to install the driver automatically,
# https://developer.nvidia.com/cuda-downloads
  
# 2.    Add the ROCm package server to your system. Refer to the OS-specific instructions in the ROCm Installation Guide.
  
# 3.    Install the "hip-runtime-nvidia" and "hip-dev" package. This will install CUDA SDK and the HIP porting layer.
  
# apt-get install hip-runtime-nvidia hip-dev
  
# Default Paths
# •       By default, HIP looks for CUDA SDK in /usr/local/cuda.
# •       By default, HIP is installed in /opt/rocm/hip.
# •       You may add /opt/rocm/bin to your path.

############################
# Verify Your Installation #
############################
# Run hipconfig using the instructions below (assuming the default installation path):
# 
# /opt/rocm/bin/hipconfig --full

function should_reboot () {
	echo "============="
	echo "SHOULD REBOOT"
	echo "============="
}

function verify () {
	echo "========="
	printf "VERIFYING ${@}"
	echo "========="
}

case $1 in
    prereqs)
        # add to groups
        # render required for ubuntu v20.04
        sudo usermod -a -G render $LOGNAME
        sudo usermod -a -G video $LOGNAME

        # install prereqs
		sudo apt install wget gnupg2
		wget -q -o - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

        # install kernel headers
        sudo apt install linux-headers-`uname -r` linux-modules-extra-`uname -r`

        # add repos
        # replace *_latest with rocm_version for explicit version
        rocm_version=5.3.3
        amdgpu_latest=latest
        rocm_latest=debian/
        amdgpu_version=$amdgpu_latest
        rocm_version=$rocm_latest
        amdgpu_base_url="https://repo.radeon.com/amdgpu/${amdgpu_version}/ubuntu"
        rocm_base_url="https://repo.radeon.com/rocm/apt/${rocm_version}"

        echo "deb [arch=amd64] ${rocm_base_url} focal main" | sudo tee /etc/apt/sources.list.d/rocm.list
        echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" | sudo tee /etc/apt/preferences.d/rocm-pin-600

        # load new repos
        sudo apt update
        ;;
	install-cuda)
		# also installs nvidia driver
		version="11.8.0"
		file="cuda_${version}_520.61.05_linux.run"
		if [[ ! -f $file ]]; then
			wget "https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${file}" 
		fi
		chmod +x ./$file
		sudo sh ./$file
		should_reboot
		;;
	install-hip-nvidia)
		sudo apt install hip-runtime-nvidia hip-dev
    		# By default HIP looks for CUDA SDK in /usr/local/cuda.
    		# By default HIP is installed into /opt/rocm/hip.
    		# Optionally, consider adding /opt/rocm/bin to your path to make it easier to use the tools.
		/opt/rocm/bin/hipconfig --full
		;;
	*)
		echo "invalid option specified"
		;;
esac


