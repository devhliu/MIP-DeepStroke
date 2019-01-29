# MIP-DeepStroke
Deep learning network for predicting lesions in stroke.

This project aims to predict stroke lesions from T2-weighted, CT-Perfusion and Diffusion-weighted images.
This research was supported by the Medical Image Processing Lab (MIPLAB) in Campus Biotech Geneva.

## Requirements
- Tensorflow(-GPU)
- Keras

## How to I use this code?
This code is relatively generic and might be used for any binary segmentation task.

### Dataset creation
First, the different input and output images should be organised in the following manner.
The images should stored by patient in .nii format.

`~/training_data/patient_id/[input_modalities1.nii, input_modalities2.nii, output_mask.nii]`

Now that the folder is well organised, you can run the following script to create a dataset of patches from the images.
- For 2D patches: `python ~/MIP-Deepstroke/Unet2D/create_dataset.py -d ~/training_data -p 512x512` for 512x512 patches.
- For 3D patches: `python ~/MIP-Deepstroke/UnetCT/create_dataset.py -d ~/training_data -p 32x32x32` for 32x32x32 (3D) patches.

By default, the data is saved under `~/Data/{ddmmyy-hhmm}/{patch_size}`.

### Training
To train a model on the data just created, use the following command:
- For 2D : `python ~/MIP-Deepstroke/Unet2D/train_2D.py -d {DATAPATH} -i {INPUT_MODALITY1} -i {INPUT_MODALITY2} -o {OUTPUT_MODALITY}`
- For 3D : `python ~/MIP-Deepstroke/Unet2D/train.py -d {DATAPATH} -i {INPUT_MODALITY1} -i {INPUT_MODALITY2} -o {OUTPUT_MODALITY}`

Example :  `python ~/MIP-Deepstroke/Unet2D/train.py -d ~/Data/010119-1000/32x32x32 -i T2 -i TRACE -o LESION`

By default, models are saved under `~/Models/{date}/checkpoints`.

### Note
This README is still under construction.

## External code
- 2D-Unet : https://github.com/nabsabraham/focal-tversky-unet
- 3D-Unet : https://github.com/ellisdg/3DUnetCNN
