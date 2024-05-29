(![Unknown](https://github.com/yanochka11/SegMed/assets/49607311/8963742d-adb0-4221-a28f-1405c98f0d60)

# SegMed: Implementation of MedSAM Segmentation and Enhancement with FeatUp and GAFL


Welcome to our SegMed project repository! This project focuses on automating brain tumor segmentation using advanced deep learning techniques, aiming to improve clinical decisions and treatment strategies.

## Table of Contents
- [Project Overview](#Project-Overview)
- [Team Members](#team-members)
- [Project File Structure](#Project-File-Structure)
- [Installation](#installation)
- [Code Snippets](#code-snippets)
- [Statistics Model Comparison](#statistics-model-comparison)
- [Reference Repository](#reference-repository)
- [Conclusion](#conclusion)

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
│   ├── download_link.txt
│   └── load_dataset_instructions
├── notebooks/
│   ├── SAM_and_SAM_plus_GAFL.ipynb
│   ├── SAM_Milestone.ipynb
│   ├── Segmentation_Baseline_Models.ipynb
│   └── SAM_and_SAM_plus_Featup.ipynb
├── results/
│   ├── SAM_Milestone_results
│   ├── SAM_General_results
│   ├── FeatUp_SAM_results
│   └── GAFL_SAM_results
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

### Dataset loading 
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
### Load Models
#### SAM Model 
```

model = SamModel.from_pretrained("facebook/sam-vit-base")
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
```
#### Feautup Model 
```
# Clone and Install Necessary Repositories

!git clone https://github.com/mhamilton723/FeatUp

import os
os.chdir("FeatUp/")

!pip install -e .
!pip install git+https://github.com/mhamilton723/CLIP.git

```
#### GAFL Model 
```
!git clone https://github.com/cviaai/GAFL.git
%cd GAFL
!pip install -r requirements.txt
!pip install -e .

```

## Statistics Model Comparison
## Reference Repository
- Segment anything: https://www.nature.com/articles/s41467-024-44824-z
- FeatUp: https://github.com/mhamilton723/FeatUp
- GAFL: https://github.com/cviaai/GAFL

## Conclusion
In our quest to automate brain tumor segmentation through deep learning, we've made significant strides by deploying and assessing a range of foundational models such as DeepLabV3+, U-Net, U-Net++, DeepLabV3, and Pyramid Attention Network (PAN). Building on this foundation, we've taken a significant step forward by integrating the SAM model into our framework. After training it for 5 epochs, we've observed encouraging initial results, achieving a mean loss of 0.0054 and a Mean Intersection over Union (IoU) of 0.644. Moving forward, we plan to enhance our MedSAM model and incorporate advanced technologies like FeatUp and GAFL. This integration aims to refine feature resolution and optimize frequency content, respectively, further boosting our model's performance.
