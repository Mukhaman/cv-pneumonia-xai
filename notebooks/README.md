# Computer Vision Final Project

This repository contains the notebook workflow for my computer vision final project. The project uses a modified version of the CXR-CLIP repository to study pneumonia classification and explainability on chest X-ray images.

The project focuses on two datasets:

- RSNA Pneumonia Detection Challenge
- NIH ChestX-ray14, converted into a binary Pneumonia vs Normal task

The main objectives were to compare pretrained and fine-tuned CXR-CLIP models, generate XAI maps using Grad-CAM and Integrated Gradients, and evaluate whether the explanation maps align with ground-truth pneumonia bounding boxes.

## Repository Structure

The notebooks are numbered in the order they are meant to be used:

| Notebook | Purpose |
|---|---|
| 0_env_setup.ipynb | Sets up the Colab environment and downloads the prepared files |
| 1_dataset_making_(optional).ipynb | Recreates the extracted RSNA and NIH datasets from the original sources |
| 2_finetuning_(optional).ipynb | Fine-tunes CXR-CLIP on RSNA and NIH |
| 3_testing.ipynb | Evaluates the pretrained and fine-tuned models |
| 4_xai_save_(optional).ipynb | Saves Grad-CAM and Integrated Gradients maps |
| 5_xai_visualisation.ipynb | Visualizes and evaluates the XAI results |

The optional notebooks do not need to be rerun if the prepared datasets, checkpoints, and XAI results are already downloaded.

## Related Code Repository

This notebook repository is meant to be used together with a modified version of the CXR-CLIP codebase. The original CXR-CLIP repository was extended with additional dataset configs, evaluation scripts, and XAI scripts for this project.

The original CXR-CLIP repository does not directly include the custom NIH pneumonia setup or the XAI-saving scripts used here.

## Environment Setup

The environment setup is handled in 0_env_setup.ipynb.

The notebook installs the required packages, clones the project repository, downloads prepared data and model files, and puts them into the expected folder structure.

The setup includes:

- Python 3.10 setup in Colab
- installation of the project requirements
- installation of additional packages such as pydicom, opencv-python-headless, captum, matplotlib, pillow, transformers, and gdown
- downloading prepared datasets
- downloading model checkpoints
- downloading saved XAI results when needed

The main workflow can be run directly after completing the environment setup.

## Datasets

Two datasets were used in this project.

### RSNA Pneumonia Dataset

The RSNA Pneumonia Detection Challenge dataset originally contains DICOM images and bounding-box annotations.

In the dataset preparation notebook, the DICOM images were converted to PNG. The conversion included reading the DICOM files, applying modality and VOI LUTs, handling MONOCHROME1 images, resizing, normalizing, and saving the result as PNG.

For classification, the RSNA labels were converted into image-level binary labels. If an image had at least one pneumonia/lung opacity bounding box, it was labeled as Pneumonia. Otherwise, it was labeled as Normal.

The final balanced RSNA split was:

| Split | Pneumonia | Normal | Total |
|---|---:|---:|---:|
| Train | 4208 | 4208 | 8416 |
| Validation | 902 | 902 | 1804 |
| Test | 902 | 902 | 1804 |

The extracted RSNA folder contains the images, train/validation/test CSV files, and the original annotation files needed for XAI evaluation.

### NIH ChestX-ray14 Dataset

The NIH dataset was originally a multi-label chest X-ray dataset. For this project, it was converted into a binary Pneumonia vs Normal task.

The positive class was defined as images whose labels contain Pneumonia. The negative class was defined as images whose label is exactly No Finding.

The NIH test set was constructed differently from the training set because XAI evaluation required bounding boxes. The final test set contains 120 pneumonia images with bounding boxes and 120 normal images without bounding boxes. Normal images do not have bounding boxes because they do not contain pneumonia findings.

The final NIH split was:

| Split | Pneumonia | Normal | Total |
|---|---:|---:|---:|
| Train | 1049 | 1048 | 2097 |
| Validation | 262 | 263 | 525 |
| Test | 120 | 120 | 240 |

This NIH setup is a custom binary version of the original dataset and should not be confused with the full NIH multi-label benchmark.

## Models

Three CXR-CLIP ResNet50-based models were used.

| Checkpoint | Description |
|---|---|
| r50_m_pretrained.tar | Original pretrained CXR-CLIP ResNet50 model |
| r50_m.tar | Model fine-tuned on the RSNA pneumonia dataset |
| nih_finetuned.tar | Model fine-tuned on the NIH Pneumonia vs Normal dataset |

The pretrained model was used as the starting point for fine-tuning.

## Fine-Tuning

Fine-tuning was done as linear probing. The pretrained CXR-CLIP image encoder was frozen, and only a linear classification head was trained for binary pneumonia classification.

The main fine-tuning settings were:

| Setting | Value |
|---|---|
| Backbone | CXR-CLIP ResNet50 image encoder |
| Backbone frozen | Yes |
| Classifier | Linear head |
| Output classes | 1 |
| Epochs | 10 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Scheduler | Cosine |
| Batch size | 64 |
| Image size | 224 |
| Normalization | ImageNet |

Training used image augmentations such as random resized crop, CLAHE, color jitter, and center crop. Validation and testing used resize and center crop.

Fine-tuning was performed separately for RSNA and NIH. The resulting models were saved as r50_m.tar and nih_finetuned.tar.

## Testing

Model evaluation is done in 3_testing.ipynb.

The three models were evaluated on both the NIH and RSNA test sets using AUROC, F1 score, and accuracy.

### NIH Results

| Model | AUROC | F1 | Accuracy |
|---|---:|---:|---:|
| Pretrained CXR-CLIP | 0.898 | 0.840 | 0.838 |
| RSNA fine-tuned | 0.894 | 0.744 | 0.779 |
| NIH fine-tuned | 0.897 | 0.817 | 0.817 |

The pretrained CXR-CLIP model achieved the highest NIH test accuracy.

### RSNA Results

| Model | AUROC | F1 | Accuracy |
|---|---:|---:|---:|
| Pretrained CXR-CLIP | 0.834 | 0.782 | 0.758 |
| RSNA fine-tuned | 0.883 | 0.802 | 0.798 |
| NIH fine-tuned | 0.856 | 0.788 | 0.751 |

The RSNA fine-tuned model achieved the highest RSNA test accuracy.

For the XAI part, the best-performing model on each dataset was used:

| Dataset | Model used for XAI |
|---|---|
| NIH | r50_m_pretrained.tar |
| RSNA | r50_m.tar |

## XAI Generation

XAI maps were saved in 4_xai_save_(optional).ipynb.

Two methods were used:

- Grad-CAM
- Integrated Gradients

The saved XAI results are organized as:

text results/   nih/     gradcam/       metadata.csv       cams/     integrated_gradients/       metadata.csv       ig_maps/   rsna/     gradcam/       metadata.csv       cams/     integrated_gradients/       metadata.csv       ig_maps/ 

For Grad-CAM, the target layer was the final ResNet block, model.image_encoder.resnet.layer4.

For fine-tuned classifier models, the target score was the pneumonia logit. For the pretrained zero-shot model, the target score was the difference between pneumonia and normal text similarities. The prompts used for the zero-shot setting were:

- No evidence of pneumonia
- Findings suggesting pneumonia.

Integrated Gradients was computed with Captum using 32 steps and a zero image baseline.

## XAI Visualization

XAI visualization is handled in 5_xai_visualisation.ipynb.

For each pneumonia-positive image, four panels are shown:

1. Original chest X-ray
2. Ground-truth bounding box
3. XAI heatmap
4. XAI overlay with bounding box

The compressed visualizations are saved as JPG files under:

Only pneumonia-positive images are used for bounding-box visualization, because normal images do not have pneumonia bounding boxes.

## XAI Evaluation

The XAI evaluation uses a bounding-box-centered metric.

For each pneumonia-positive image:

1. The XAI heatmap is normalized.
2. The top 15% most activated pixels are selected.
3. These pixels are converted into a binary XAI mask.
4. The mask is compared with the ground-truth pneumonia bounding boxes.
5. If an image has multiple boxes, the best-covered box is used.
6. Localization is counted as successful if the XAI mask covers at least 50% of at least one ground-truth bounding box.

In this project, good bbox localization means that the top 15% XAI region covers at least 50% of at least one pneumonia bounding box.

### XAI Localization Results

| Dataset | Method | Good bbox localization | Mean bbox covered by XAI | Mean energy in bbox | Mean IoU |
|---|---|---:|---:|---:|---:|
| NIH | Grad-CAM | 0.517 | 0.534 | 0.222 | 0.211 |
| NIH | Integrated Gradients | 0.050 | 0.283 | 0.137 | 0.097 |
| RSNA | Grad-CAM | 0.835 | 0.727 | 0.215 | 0.247 |
| RSNA | Integrated Gradients | 0.297 | 0.417 | 0.148 | 0.132 |

Grad-CAM performed better than Integrated Gradients on both datasets. The difference was especially large on NIH, where Integrated Gradients produced much weaker bbox localization.

## Main Findings

The main findings were:

- Fine-tuning improved performance on RSNA.
- The pretrained CXR-CLIP model performed best on the modified NIH test set.
- Grad-CAM produced more spatially meaningful explanations than Integrated Gradients.
- RSNA Grad-CAM had the strongest bbox localization result.
- NIH was more challenging because it was converted from a multi-label dataset and had only 120 pneumonia-positive test images with bounding boxes.

## Suggested Workflow

To run the project using the prepared files, use:

1. 0_env_setup.ipynb
2. 3_testing.ipynb
3. 5_xai_visualisation.ipynb

To reproduce everything from scratch, use:

1. 0_env_setup.ipynb
2. 1_dataset_making_(optional).ipynb
3. 2_finetuning_(optional).ipynb
4. 3_testing.ipynb
5. 4_xai_save_(optional).ipynb
6. 5_xai_visualisation.ipynb

## Notes and Limitations

The NIH dataset used here is a custom binary version of NIH ChestX-ray14, not the full original multi-label benchmark.

The NIH pneumonia bounding-box test set is small, with only 120 positive images. Normal images do not have bounding boxes, so localization evaluation is only done on pneumonia-positive images.

Bounding boxes are coarse annotations and do not represent exact segmentation masks. Therefore, the XAI localization results should be interpreted as approximate.

The model used for XAI was selected based on the highest test accuracy for each dataset. For a stricter experimental setup, model selection should ideally be done using validation performance only.

Finally, fine-tuning was performed with the image encoder frozen. Full fine-tuning could lead to different results but was not used in this project.

## Acknowledgment

This project is based on the official implementation of CXR-CLIP:

CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training

The original CXR-CLIP repository and pretrained checkpoints belong to the original authors. This repository contains the notebook workflow used for the final project experiments built on top of the modified CXR-CLIP codebase.