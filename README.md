# ePASS: Enhanced Physical-layer Authentication with Super-resolution for Satellites

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Communications%20Letters-blue)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2025a-orange.svg)](https://www.mathworks.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"Lightweight Physical-Layer Authentication via IQ Sample Super-Resolution in LEO Satellite Networks"**

> **Authors:** Ivy Selorm Dogbey†, Yongjae Lee†, Jihwan Moon, Taehoon Kim, and Inkyu Bang
>
> † These authors contributed equally to this work.
>
> **Submitted in:** IEEE Communications Letters

---

## Overview

ePASS is a practical and lightweight **physical-layer authentication (PLA)** framework for LEO satellite communication networks. It leverages **Enhanced Deep Super-Resolution (EDSR)** techniques to enhance the resolution of IQ sample-based images, enabling accurate satellite authentication with a minimal number of received packets.

### Key Features

- **IQ Sample to Image Conversion**: Transform raw IQ data into 2D histogram images
  - Cartesian coordinates (224x224) for standard processing
  - Polar coordinates (256x256) for AMC-style comparison
- **Super-Resolution Enhancement**: EDSR x4 upscaling (224x224 → 896x896)
- **Multi-Architecture Support**:
  - CNN: Modified ResNet-18 for 66-class satellite classification
  - RNN: LSTM and GRU for sequential feature extraction
- **Spoofing Attack Detection**: Robust authentication using sparse autoencoder

### Performance Highlights

- **90.3% authentication accuracy** with only 500 IQ samples (~5 packets)
- **94.77% authentication accuracy** with 1,000 IQ samples
- Significantly outperforms Raw ResNet baseline and AMC methods
- Suitable for real-time operation on edge AI hardware (e.g., NVIDIA Jetson Orin Nano)

![System Model](figures/fig1_system_model.png)

---

## Project Structure

```
ePASS/
├── src/
│   ├── gen_img_raw.py         # Generate Cartesian IQ images (224x224)
│   ├── gen_img_amc.py         # Generate Polar (AMC-style) images (256x256)
│   ├── train_cnn.py           # CNN training (ResNet-18) with PyTorch
│   ├── train_rnn.py           # RNN training (LSTM/GRU)
│   └── upscale_images.py      # EDSR/Bicubic x4 upscaling (CUDA accelerated)
├── models/
│   └── EDSR_x4.pb             # Pre-trained EDSR model (x4 scale)
├── figures/                   # Paper figures
├── results/                   # Experiment results
└── data/sample/               # Sample data
```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

**Python Environment (3.12+)**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.5.0 (with CUDA support for EDSR acceleration)
- opencv-contrib-python >= 4.5.0
- numpy, pandas, matplotlib
- scikit-learn, tqdm, Pillow, openpyxl

**MATLAB Environment (R2025a)**
- Deep Learning Toolbox
- Image Processing Toolbox

**Hardware (Recommended)**
- NVIDIA GPU with CUDA support
- For deployment: NVIDIA Jetson Orin Nano or equivalent edge AI hardware

---

## Usage

### 1. Generate Images from IQ Data

```bash
# Cartesian coordinates (224x224) - Standard ePASS input
python src/gen_img_raw.py

# Polar coordinates (256x256) - AMC-style comparison
python src/gen_img_amc.py
```

**Configuration (in scripts):**
- Sample sizes: 500, 1000, 2000, 5000
- Normalization: Per-satellite min-max scaling to [-1, 1]

### 2. Apply Super-Resolution

```bash
python src/upscale_images.py
```

**Output:**
- `images_896_edsr/`: EDSR x4 upscaled images (896x896)
- `images_896_bicubic/`: Bicubic x4 upscaled images (896x896)

**Note:** CUDA acceleration is enabled by default. Requires OpenCV built with CUDA support.

### 3. Train Classification Models

```bash
# CNN (ResNet-18) - Multiple modes and sample sizes
python src/train_cnn.py --mode edsr bicubic --samples 500 1000 5000 --epochs 10

# RNN (LSTM/GRU)
python src/train_rnn.py --model lstm gru --mode edsr --samples 5000 --epochs 10
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Data mode: base, amc, edsr, bicubic | edsr, bicubic |
| `--samples` | Sample sizes per image | 500, 1000 |
| `--epochs` | Training epochs | 1 |
| `--limit` | Images per class | 200 |
| `--batch_size` | Batch size | 32 |
| `--split_strategy` | Data split: random, sequential | random |

### Training Parameters (ResNet-18)

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD with Momentum (0.9) |
| Initial Learning Rate | 1e-3 |
| Learning Rate Drop Factor | 0.1 (every 5 epochs) |
| Mini-batch Size | 32 |
| Max Epochs | 10 |
| Normalization | ImageNet mean/std |

---

## Results

### Intra-Satellite Authentication (Hit Rate)

| Method | 0.5K | 1K | 2K | 5K | 10K |
|--------|------|------|------|------|------|
| Raw + ResNet-18 | 73.83% | 82.27% | 85.30% | 96.74% | 97.31% |
| Bicubic + ResNet-18 | 86.82% | 96.74% | - | - | - |
| AMC + ResNet-18 | 46.44% | 60.11% | 67.12% | 91.33% | 94.73% |
| EDSR + LSTM | 1.52% | 1.52% | - | - | - |
| EDSR + GRU | 1.52% | 1.52% | - | - | - |
| **ePASS (EDSR + ResNet-18)** | **90.30%** | **94.77%** | - | - | - |

### Spoofing Detection Performance

| Sample Size | Scenario | Authentication | Spoofing Detection | Accuracy | F1 Score |
|-------------|----------|----------------|-------------------|----------|----------|
| 1K | Best (weak attacker) | 0.312 | 0.912 | 0.615 | 0.415 |
| 1K | Worst (strong attacker) | 0.347 | 0.972 | 0.685 | 0.495 |
| 5K | Best | 0.957 | 0.706 | 0.761 | 0.825 |
| 5K | Worst | 0.987 | 0.756 | 0.861 | 0.866 |
| 10K | Best | 0.938 | 0.937 | 0.887 | 0.891 |
| 10K | Worst | 0.978 | 1.000 | 0.989 | 0.993 |

### Computational Complexity

| Metric | ePASS | PAST-AI (Baseline) |
|--------|-------|-------------------|
| Computational Cost | ~8.21 TFLOPs | ~1.80 GFLOPs |
| Input Data Size | 500 IQ samples (~5 packets) | 10,000 IQ samples (~87 packets) |
| Data Acquisition Latency | Low (instantaneous) | High (depends on inter-packet delay) |
| Feasible Hardware | NVIDIA Jetson Orin Nano | Generic IoT CPU/MCU |

---

## Dataset

This work uses the publicly available **Iridium satellite IQ sample dataset**:

> G. Oligeri, S. Sciancalepore, and R. Di Pietro, "Physical-layer data of IRIDIUM satellites broadcast messages," *Data in Brief*, vol. 46, p. 108905, 2023.

**Dataset Characteristics:**
- 66 Iridium LEO satellites
- Over 102 million IQ samples total
- ~1.5 million samples per satellite
- IRA (Iridium Ring Alert) channel messages

---

## Citation

*Citation information will be added upon publication.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **EDSR**: [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) - Super-resolution model architecture
- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - Classification backbone
- **Iridium Dataset**: [PAST-AI](https://github.com/spritz-group/PAST-AI) - Public satellite IQ sample dataset
- **AMC Framework**: T. K. Oikonomou et al., "CNN-Based Automatic Modulation Classification Under Phase Imperfections," IEEE WCL, 2024 - Polar coordinate transformation baseline

---

## Contact

For questions or issues, please open an issue or contact:
- Taehoon Kim: thkim@hanbat.ac.kr
- Inkyu Bang: ikbang@hanbat.ac.kr
