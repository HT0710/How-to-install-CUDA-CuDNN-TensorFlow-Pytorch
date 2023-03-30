# <p align="center">How to install CUDA, CuDNN, TensorFlow and Pytorch</p>


## Table of Contents
- [How to install CUDA, CuDNN, TensorFlow and Pytorch](#how-to-install-cuda-cudnn-tensorflow-and-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [**Requirements**](#requirements)
    - [System](#system)
    - [Software](#software)
  - [**Preparation (IMPORTANT)**](#preparation-important)
    - [1. NVIDIA Driver](#1-nvidia-driver)
      - [Verification](#verification)
    - [2. Miniconda](#2-miniconda)
      - [Verification](#verification-1)
  - [**Install CUDA**](#install-cuda)
      - [Verification](#verification-2)
  - [**Install CuDNN**](#install-cudnn)
      - [Verification](#verification-3)
  - [**Install TensorFlow**](#install-tensorflow)
      - [Verification](#verification-4)
  - [**Install Pytorch**](#install-pytorch)
      - [Verification](#verification-5)
  - [License](#license)
  - [References](#references)
  - [Contact](#contact)


## Introduction
Having trouble getting your deep learning model to run on GPU. Please follow the instructions.

This is a step by step instructions of how to install:
```
- CUDA 11.8
- CuDNN 8.6.0
- TensorFlow 2.12.*
- Pytorch 2.0
```

> **Note:**
> - You can skip TensorFlow or Pytorch if don't use it.
> - Pytorch come with it own CuDNN so you can skip CuDNN installation if use Pytorch only.

## **Requirements**
### System
```
- Ubuntu 16.04 or higher (64-bit)
- NVIDIA Graphics Card *
```

> **Note:**
> - I don't recommend trying to use GPU on Windows, believe me it's not worth the effort.
> - TensorFlow only officially support Ubuntu. However, the following instructions may also work for other Linux distros.
> - \* AMD doesn't have CUDA cores. *CUDA is proprietary framework created by Nvidia and it's used only on Nvidia cards.*
> - Personally I am using Zorin OS and it works fine.

### Software
```
- Python 3.8–3.11
- NVIDIA GPU drivers version 450.80.02 or higher.
- Miniconda (Recommended) *
```

> **Note:**
> - I will also include how to install the NVIDIA Driver and Miniconda in this instructions if you don't already have it.
> - \* Miniconda is the recommended approach for installing TensorFlow with GPU support. It creates a separate environment to avoid changing any installed software in your system. This is also the easiest way to install the required software especially for the GPU setup.


## **Preparation (IMPORTANT)**
### 1. NVIDIA Driver
#### Verification
Check if you already have it by run this on your terminal:
```bash
nvidia-smi 
```

If you got the output, the NVIDIA Driver is already installed. Then go to the next step.

![smi](https://user-images.githubusercontent.com/95120444/228067789-0d42c6f4-daaa-4c9c-aa4c-bc95359f4a5e.png)

If not, follow those step bellow:
1. Go to NVIDIA Driver Downloads site: https://www.nvidia.com/download/index.aspx?lang=en-us
2. Search for your GPU and then download it. Remember to choose `Linux 64-bit` Operating System
    ![driver](https://user-images.githubusercontent.com/95120444/228067841-b66b38aa-8d27-4f9d-8ad1-0c87bb524fc7.png)
3. Blacklist nouveau
   > **Note:** It does not work with CUDA and must be disabled
    ```bash
    sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf" 
    sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf" 
    ```
4. Remove old NVIDIA driver (optional)
    > **Note:** Desktop maybe temporary at lower resolution after this step
    ```bash
    sudo apt-get remove '^nvidia-*' 
    sudo apt autoremove 
    reboot
    ```
5. Install any pending updates and all required packages
    ```bash
    sudo apt update && sudo apt upgrade -y 
    sudo apt install build-essential libglvnd-dev pkg-config 
    ```
6. Install the driver:
    > **Note:** Your driver version may higher than this instructions, those following command is an example. **Please use `Tab` to autocomplete the file name.**
    1. Stop current display server
        > **Note:** For the smoothest installation
        ```bash
        sudo telinit 3
        ```
    2. Enter terminal mode, press: **`CTRL + ALT + F1`** and login with your username and password
    3. Navigate to your directory containing the driver
        ```bash
        # Example
        cd Downloads/
        ls
        # It must contain: NVIDIA-Linux-x86_64-5xx.x.x.run
        ```
    4. Give execution permission
        ```bash
        # Example
        sudo chmod -x NVIDIA-Linux-x86_64-5xx.x.x.run 
        ```
    5. Run the installation
        ```bash
        # Example
        sudo ./NVIDIA-Linux-x86_64-5xx.x.x.run 
        ```
    6. Following the wizard and search google if unsure
        > **Note:** Usually you just need to press Enter the whole thing
    7. The Nvidia driver is now installed. Reboot your system
        ```bash
        reboot 
        ```
7.  [Verification](#verification)
> **Note:** It has many other way to install it, but in my experience this way cause less errors or simple to fix compare to other methods.

### 2. Miniconda
You can use the following command to install Miniconda

Download
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh 
```
Install
1. Run bellow:
    ```bash
    sh Miniconda3-latest-Linux-x86_64.sh 
    ```
2. Press Enter to continue
3. Press q to skip the License Agreement detail
4. Type `yes` and press Enter
5. Press Enter to confirm the installation location
6. Reopen your terminal or:
   ```bash
   source ~/.bashrc
   ```
7. Disable conda auto activate base
    ```bash
    conda config --set auto_activate_base false 
    ```

#### Verification
```bash
conda -V 
```
> **Note:** Miniconda is a free minimal installer for conda. Is a package and environment manager that helps you install, update, and remove packages from your command-line interface. You can use it to write your own packages and maintain different versions of them in separate environments.

## **Install CUDA**
- The installation bellow is **CUDA Toolkit 11.8**
- It automatically recognize the distro and install the appropriate version.

Download:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run 
```

Install:
1. Run bellow, it will take some minutes please be patient.
    ```bash
    sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit 
    ```
2. Add CUDA to path:
    ```bash
    echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc 
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc 
    ```
3. Reopen your terminal or:
    ```bash
    source ~/.bashrc
    ```

> **Note:** Same as the driver, it has many other way to install it but with this way you can install and use multiple version of CUDA by simply change the version of CUDA in path (~/.bashrc).

#### Verification
```bash
nvcc --version 
```
Output:

![cuda](https://user-images.githubusercontent.com/95120444/228067950-c72ee4e5-fcf7-488b-9ff4-7739b0bd405b.png)

## **Install CuDNN**
The installation bellow is **cuDNN v8.6.0**

1. Go to this site: https://developer.nvidia.com/rdp/cudnn-archive
2. You'll have to log in, answer a few questions then you will be redirected to download
3. Select **Download cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x**
4. Select **Local Installer for Linux x86_64 (Tar)**
    ![cudnn1](https://user-images.githubusercontent.com/95120444/228067998-c7738898-aad4-43a6-8fb4-e9830a022b84.png)
5. Open terminal and then navigate to your directory containing the cuDNN tar file
6. Unzip the CuDNN package
    ```bash
    tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz 
    ```
7. Copy those files into the CUDA toolkit directory
    ```bash
    sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* 
    ```

> **Note:** You need to have a developer account to get CuDNN there are no direct links to download files. Why? *Ask Nvidia*.

#### Verification
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 
```
Output:

![cudnn](https://user-images.githubusercontent.com/95120444/228068029-a4e9fabb-b1aa-4b30-bb6f-0a3c7f27ce0f.png)

## **Install TensorFlow**
Please read the [Requirements](#requirements) and the [Preparation](#preparation-important) sections before continue the installation bellow.

1. Create a conda environment
    - Create a new conda environment named `tf` and `python 3.9`:
        ```bash
        conda create --name tf python=3.9 
        ```
    - You can deactivate and activate it:
        ```bash
        conda deactivate 
        conda activate tf 
        ```
    > **Note:** Make sure it is activated for the rest of the installation.

2. GPU setup
    - You can skip this section if you only run TensorFlow on the CPU.
    - Make sure the NVIDIA GPU driver is installed. Use the following command to verify it:
        ```bash
        nvidia-smi 
        ```
    - Then install CUDA and cuDNN with conda and pip.
        ```bash
        conda install -c conda-forge cudatoolkit=11.8.0 
        pip install nvidia-cudnn-cu11==8.6.0.163 
        ```
    - Configure the system paths.
        ```bash
        mkdir -p $CONDA_PREFIX/etc/conda/activate.d 
        CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) 
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh 
        ```

3. Install TensorFlow
    - TensorFlow requires a recent version of pip, so upgrade your pip installation to be sure you're running the latest version.
        ```bash
        pip install --upgrade pip 
        ```
    - Then, install TensorFlow with pip.
        ```bash
        pip install tensorflow==2.12.* 
        ```
    > **Note:** Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.

#### Verification
- Verify the CPU setup:
    ```bash
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" 
    ```
    If a tensor is returned, you've installed TensorFlow successfully.
- Verify the GPU setup:
    ```bash
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 
    ```
    If a list of GPU devices is returned, you've installed TensorFlow successfully.

## **Install Pytorch**
Please read the [Requirements](#requirements) and the [Preparation](#preparation-important) sections before continue the installation bellow.

> **Note:** Pytorch come with it own CuDNN so you can skip CuDNN installation if use Pytorch only.

1. Create a conda environment
    - Create a new conda environment named `torch` and `python 3.9`:
        ```bash
        conda create --name torch python=3.9 
        ```
    - You can deactivate and activate it:
        ```bash
        conda deactivate 
        conda activate torch 
        ```
    > **Note:** Make sure it is activated for the rest of the installation.

2. Install TensorFlow
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
    ```



#### Verification
```bash
# Check CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"
# CUDA device count
python3 -c "import torch; print(torch.cuda.device_count())"
# Current CUDA device
python3 -c "import torch; print(torch.cuda.current_device())"
# Get device 0 name
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## License
This project is licensed under the MIT License. See [LICENSE](https://github.com/HT0710/How-to-install-CUDA-CuDNN-TensorFlow-Pytorch/blob/main/LICENSE) for more details.

## References
- NVIDIA Driver: https://www.nvidia.com/download/index.aspx?lang=en-us
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
- CuDNN: https://developer.nvidia.com/rdp/cudnn-archive
- TensorFlow: https://www.tensorflow.org/install/pip
- Pytorch: https://pytorch.org/get-started/locally/

## Contact
Open an issue: [New issue](https://github.com/HT0710/How-to-install-CUDA-CuDNN-TensorFlow-Pytorch/issues/new)

Mail: pthung7102002@gmail.com