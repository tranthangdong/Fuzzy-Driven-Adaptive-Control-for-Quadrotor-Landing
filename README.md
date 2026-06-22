[![Build Status](https://github.com/tranthangdong/Fuzzy-Driven-Adaptive-Control-for-Quadrotor-Landing/blob/main/docs/badge.svg)](https://github.com/tranthangdong/Fuzzy-Driven-Adaptive-Control-for-Quadrotor-Landing)
<p align="center">
<img src="imgs/iaedtu.jpg" alt="" width="20%" />
</p>

# Fuzzy-Driven-Adaptive-Control-for-Quadrotor-Landing

An enhanced version of the precision landing algorithm uses a 3x3 fuzzy controller 

## 1. 🛠️ Requirements
This setup is tested on Ubuntu 22.04
### a. ROS2
This package was written using ROS2 Humble under Ubuntu 22.04. If it's not installed, check the ROS2 documentation or run the commands below.
```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt upgrade -y
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
```
Don't forget to source your installation in the terminal or in your bashrc (See example below).
```bash
source /opt/ros/humble/setup.bash
```
### b. PX4 Autopilot
We used the PX4 Autopilot which uses the Micro XRCE-DDS Agent & Client as middleware. Here are the installation commands to run in your terminal, you can also check the PX4 documentation:

```bash
cd
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot/
make px4_sitl

pip install --user -U empy==3.3.4 pyros-genmsg setuptools

git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```
### c. Gazebo Transport (Gx-Transport)
```bash
# Install Gazebo Harmonic
sudo apt install gz-harmonic
sudo apt install libgz-transport13-dev
sudo apt install python3-gz-transport13
```
### d. OpenCV & Python Dependencies
```bash
# OpenCV 4.7+ 
pip3 install --user opencv-python==4.7.0.72
pip3 install --user opencv-contrib-python==4.7.0.72
# Install pip và Python libs
sudo apt install python3-pip python3-numpy python3-opencv

# Install others Python libs
pip3 install --user numpy opencv-python opencv-contrib-python
pip3 install --user rclpy
pip3 install --user pyyaml
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep2
sudo rosdep init
rosdep update

```
### e. Install and build the package
Create a ROS2 workspace (Skip if already done)
```bash
cd
mkdir -p px4_ws/src
```
Then clone this repository (project package) in px4_ws/src and rename the directory as prj_DroneLanding
```bash
cd px4_ws/src
git clone https://github.com/tranthangdong/Fuzzy-Driven-Adaptive-Control-for-Quadrotor-Landing.git
mv prj_DroneLanding
```
## 2. 🚀 How to run
.
## 3. ✍️ Paper:
.
## 4. 🔗 References
