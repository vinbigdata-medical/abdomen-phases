# Phase Recognition in Contrast-Enhanced CT Scans basedon Deep Learning and Random Sampling

This repository contains the training code for our paper entitled "Phase Recognition in Contrast-Enhanced CT Scans basedon Deep Learning and Random Sampling", which was submitted and under review by [Medical Physics](https://www.medphys.org/).


## Abstract 

### Purpos
A fully automated system for interpreting abdominal computed tomography (CT) scans with multiple phases of contrast enhancement requires an accurate classification of the phases. Current approaches to classify the CT phases are commonly based on 3D convolutional neural network (CNN) approaches with high computational complexity and high latency. This work aims at developing and validating a precise, fast multi-phase classifier to recognize three main types of contrast phases in abdominal CT scans.

### Methods
We propose in this study a novel method that uses a random sampling mechanism on top of deep CNNs for the phase recognition of abdominal CT scans of four different phases: non-contrast, arterial, venous, and others. The CNNs work as a slice-wise phase prediction, while the random sampling selects input slices for the CNN models. Afterward, majority voting synthesizes the slice-wise results of the CNNs, to provide the final prediction at scan level.

### Results
Our classifier was trained on 271,426 slices from 830 phase-annotated CT scans, and when combined with majority voting on 30% of slices randomly chosen from each scan, achieved a mean F1-score of 92.09% on our internal test set of 358 scans. The proposed method was also evaluated on 2 external test sets: CTPAC-CCRCC (N = 242) and LiTS (N = 131), which were annotated by our experts. Although a drop in performance has been observed, the model performance remained at a high level of accuracy with a mean F1-score of 76.79% and 86.94% on CTPAC-CCRCC and LiTS datasets, respectively. Our experimental results also showed that the proposed method significantly outperformed the state-of-the-art 3D approaches while requiring less computation time for inference.


## Preprocess DICOM image

 Image read from raw .dcm file needs to be processed as followed:
 
 - Convert pixel values to HU standards using formula: 
 newValue = RescaleSlope * pixelValue + RescaleIntercept 
 > **RescaleSlope** and **RescaleIntercept** can be extracted from the metadata  .dcm file
 - Apply HU window to the image with window_width=400, window_center=50 


## Architecture
![](images/Pipeline_1.png)


## Model training

#### 1.  Data Preparation:

- We preprocess data as mentioned in **Preprocess DICOM image** section
- Training, validating and testing **.csv** file of **2D dataset** should follow this format:

| Study_ID  | Image       			  |  SeriesNumber  			   | Label  |
|-------------|--------------------|--------------------------- |-----|
| Study id    | Path to image    | Extract from metadata | slice label |


#### 2.  Training Configuration:

Configuration used in the paper are in folder `core/config`
It is  recommended that you change training configuration in .yaml files

Command to train the 2D model:
```
python main.py --config "PATH_TO_CONFIG_FILE"
```
Command to evaluate the 2D model:
```
python main.py --config "PATH_TO_CONFIG_FILE" --load "PATH_TO_MODEL_CHECKPOINT" --mode "VALID_OR_TEST"
```
