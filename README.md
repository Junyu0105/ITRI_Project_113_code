# ITRI_Project_113_code
# Instruction

NVIDIA Modulus is a powerful library and platform to implement Physics-informed Neural Network(PINN). Here, we demonstrate how to install the Modulus environement and how to use it. The all materials are referenced from https://github.com/NVIDIA/modulus-sym.



## System Requirements

- **Operating System**
  - Ubuntu 20.04 or Linux 5.13 kernel

- **Driver and GPU Requirements**

  - `pip`: NVIDIA driver that is compatible with local PyTorch installation.

  - Docker container: Modulus container is based on CUDA 11.8, which requires NVIDIA which requires NVIDIA Driver release 520 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 450.51 (or later R450), 470.57 (or later R470), 510.47 (or later R510), or 515.65 (or later R515).The CUDA driver’s compatibility package only supports particular drivers. Thus, users should upgrade from all R418, R440, and R460 drivers, which are not forward-compatible with CUDA 11.8. 3Driver release 515 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 450.51 (or later R450), 470.57 (or later R470), or 510.47 (or later R510). However, any drivers older than 465 will not support the SDF library. For additional support details, see [PyTorch NVIDIA Container](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12).

- **Required installations for pip install**

  - Python 3.8

- **Recommended Hardware**

  - 64-bit x86
  - [NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus):
    - NVIDIA Ampere GPUs - A100, A30, A4000
    - Volta GPUs - V100
    - Turing GPUs - T4
  - Other Supported GPUs:
    - NVIDIA Ampere GPUs - RTX 30xx
    - Volta GPUs - Titan V, Quadro GV100
  - For others, please reach us out at [Modulus Forums](https://forums.developer.nvidia.com/t/welcome-to-the-modulus-physics-ml-model-framework-forum)



## Installation

We install the latest version of Modulus Symbolic by using PyPi:

```bash
pip install nvidia-modulus.sym
```



## Implementation

After installing the packages, you can try to run the codes in the example folders.  For example, if I want to run the 1D wave equation:

```bash
cd wave_1d
python wave_1d.py
```



## Contact Information

If you have any problem, don't hestitate to contact us.

> ######  NTHU Extreme Events Computational Lab 黃琮暉老師實驗室

- Lab contact Info: #33760(R530)
