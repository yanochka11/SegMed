# SegMed: Implementation of MedSAM Segmentation and Enhancement with FeatUp and GAFL


Welcome to our SegMed project repository! This project focuses on automating brain tumor segmentation using advanced deep learning techniques, aiming to improve clinical decisions and treatment strategies.

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Project File Structure](#project-file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Code Snippets](#code-snippets)

## Project Overview

Brain tumors critically affect patient health, survival, and quality of life. Manual segmentation of tumors from MRI images, essential for treatment planning, is both slow and prone to errors. Our project aims to automate this process with advanced deep learning techniques, focusing on the development of the MedSAM model, a specialized adaptation of the Segment Anything Model (SAM) for medical imaging. By enhancing the accuracy and efficiency of tumor segmentation, we seek to support more informed clinical decisions, optimize treatment strategies, and ultimately improve patient outcomes.

## Team Members

| Name              | Role                | Contact Information |
|-------------------|---------------------|---------------------|
| Hasaan Maqsood    | Project Lead        | [Email](mailto:Hasaan.Maqsood@skoltech.ru) |
| Iana Kulichenko   | Data Scientist      | [Email](mailto:Iana.Kulichenko@skoltech.ru) |
| Daniil Volkov     | ML Engineer         | [Email](mailto:Daniil.Volkov@skoltech.ru) |

## Project File Structure

This repository supports the development and evaluation of segmentation models, alongside providing necessary documentation. Below is a detailed overview of the repository's structure.

```plaintext
SegMed/
├── weights/
│   └── instructions.txt
├── dataset/
│   └── download_link.txt
├── notebooks/
│   ├── GAFL.py
│   ├── SAM_Milestone.ipynb
│   ├── Segmentation_Baseline_Models.ipynb
│   └── featup_example.ipynb
├── README.md
└── requirements.txt
```
## Installation
To install the required libraries, use the following commands:
```
git clone https://github.com/Hasaanmaqsood/SegMed.git
cd SegMed
pip install -r requirements.txt
```

## Usage
Training Segmentation Models
To train segmentation models, run the notebooks in the notebooks directory:

Segmentation_Baseline_Models.ipynb: Train baseline segmentation models.
SAM_Milestone.ipynb: Train the MedSAM model.
Downloading and Preparing Datasets
To download and prepare the datasets, follow the instructions in dataset/download_link.txt.

Using Pre-trained Weights
Download the best saved model weights using the instructions in weights/instructions.txt and place them in the appropriate directory.
```
## Code Snippets
Download Segmentation Dataset

# Downloading dataset from Kaggle
from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p /content
!unzip /content/lgg-mri-segmentation.zip -d /content/dataset
```
## Load Models
## Make Segmentation Prediction
## Conclusion
In our quest to automate brain tumor segmentation through deep learning, we've made significant strides by deploying and assessing a range of foundational models such as DeepLabV3+, U-Net, U-Net++, DeepLabV3, and Pyramid Attention Network (PAN). Building on this foundation, we've taken a significant step forward by integrating the SAM model into our framework. After training it for 5 epochs, we've observed encouraging initial results, achieving a mean loss of 0.0054 and a Mean Intersection over Union (IoU) of 0.644. Moving forward, we plan to enhance our MedSAM model and incorporate advanced technologies like FeatUp and GAFL. This integration aims to refine feature resolution and optimize frequency content, respectively, further boosting our model's performance.
