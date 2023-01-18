# add to groups
# render required for ubuntu v20.04
sudo usermod -a -G render $LOGNAME
sudo usermod -a -G video $LOGNAME

# install prereqs
sudo apt install wget gnupg2

# install kernel headers
sudo apt install linux-headers-`uname -r` linux-modules-extra-`uname -r`

sudo apt install git openssh-server htop

# install python
sudo apt install python3 python-is-python3 python3-pip python3-venv
