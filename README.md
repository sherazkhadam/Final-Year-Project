
# 📍 Few-Shot Landmark Recognition using Deep Learning and Prototypical Networks

This repository presents a comprehensive implementation and comparative analysis of three deep learning models—**ResNet-50**, **EfficientNet-B0**, and **Prototypical Networks (ProtoNet)**—for landmark classification using the **Google Landmark v2 dataset**. This project emphasizes the role of **bounding box preprocessing**, **transfer learning**, and **few-shot learning (FSL)** in addressing cross-domain generalization and data scarcity challenges in real-world landmark recognition.

## 🔍 Overview

Real-world landmark recognition systems must operate under diverse environmental conditions, limited training data, and high intra-class variability. To address these challenges, this study explores three different learning paradigms:

- **Transfer Learning** using pretrained **ResNet-50** and **EfficientNet-B0**
- **Few-Shot Learning** using **Prototypical Networks**
- **Bounding Box Localization** to isolate architectural features
- **Model Comparison** in terms of accuracy, computational cost, and inference speed

All models are trained and evaluated using GPU acceleration on the **Kaggle platform**.


## 🧠 Models and Methodology

### 1. Transfer Learning

- **ResNet-50** and **EfficientNet-B0** are pretrained on ImageNet.
- Fine-tuning is performed on the full image dataset.
- Evaluation includes accuracy, confusion matrices, and inference time.

### 2. Few-Shot Learning (ProtoNet)

- Implemented using an episodic training strategy.
- EfficientNet-B0 used as a feature encoder.
- Training and evaluation are done on cropped images localized via bounding boxes.
- COCO-style annotations are generated for bounding box supervision.

### 3. Bounding Box Preprocessing

- Automatically generated using object detection (e.g., pretrained R-CNN).
- Removes background noise and focuses model attention on key architectural features.
- Enables COCO-style training for enhanced generalization.

## 📦 Installation & Dependencies

```bash
# Clone the repository
git clone https://github.com/sherazkhadam/Final-Year-Project.git
```

*Note: This project is optimized for execution on Kaggle kernels with GPU enabled.*

## 📊 Dataset

- https://www.kaggle.com/datasets/google/google-landmarks-dataset: A curated subset of 50 landmark classes was used.
- Augmentation techniques such as horizontal flips, rotations, and brightness adjustments were applied.
- Bounding boxes were used to crop key landmark regions for ProtoNet training.

## 🚀 Training & Evaluation

### Transfer Learning (ResNet-50 / EfficientNet)

```bash
# Run the notebook to train and evaluate
open codes/resnet-efficient.ipynb
```

### Few-Shot Learning (ProtoNet)

```bash
# Open and run the notebook
open codes/bounding-boxes code.ipynb
```

- Generates bounding boxes
- Converts to COCO annotations
- Trains ProtoNet using episodic learning





## 👨‍💻 Author

Developed by SHERAZ KHADAM as part of a graduate-level dissertation on cross-domain few-shot landmark recognition.
