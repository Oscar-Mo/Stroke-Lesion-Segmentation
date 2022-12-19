# CSCI 3343: Computer Vision - Final Project

## Running the code
The notebooks can be executed locally or online (e.g. colab). However, the data is not available to the public and cannot be distributed by us according to terms we agreed to. This makes it difficult to run the code - please contact any of us if this causes issues for marking.

## Contribution
Oscar Moses - wrote the base code for training and evaluating UNet models that we all built upon for individual experiments, sourced the data from MICCAI Society and split the data into test/training sets, ran the (2+1)D experiment.

Joey Dronkers - 

Jian Huang - 

## Background
Ischemic lesions in the brain arise after a blockage to the cerebral blood supply, otherwise known as a stroke [1]. We are interested in the automatic segmentation of ischemic lesions using convolutional neural networks. The goal is to train a model using multimodal MRI data (DWI, ADC and FLAIR images) from the MICCAI Society. 


## References/credits
ATLAS paper - https://www.nature.com/articles/s41597-022-01401-7 

ATLAS sample code - https://github.com/npnl/isles_2022/ 

UNet 3D - https://github.com/jphdotam/Unet3D/blob/main/unet3d.py 

BCELoss - https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook 

2+1 D - https://paperswithcode.com/method/2-1-d-convolution 
