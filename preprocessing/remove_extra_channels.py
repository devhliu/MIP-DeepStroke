import numpy as np
import nibabel as nb
import os
from tqdm import tqdm

input_dir = "C:\\Users\simon\Documents\EPFL\Master\Semestre5\Projet MIPLAB\Data_2016_T2_TRACE_LE"

patients = [os.path.join(input_dir,x) for x in os.listdir(input_dir)]
# through each patients
for p in tqdm(patients):
    #through each folder of the patient
    folders = [os.path.join(p,x) for x in os.listdir(p) if os.path.isdir(os.path.join(p,x))]
    for f in folders:
        #through each files of the folder
        files = [os.path.join(f,x) for x in os.listdir(f) if x.endswith(".nii")]
        for file in files:
            if "coreg" in file:
                os.remove(file)

        files = [os.path.join(f, x) for x in os.listdir(f) if x.endswith(".nii")]
        for file in files:
            copy_filename = file.replace(".nii", "-cleaned.nii")

            img = nb.load(file)
            if len(img.get_data().shape)==4:
                print(file)
                farray_img = nb.Nifti1Image(img.get_data()[:, :, :, 1], img.affine)
            else:
                farray_img = nb.Nifti1Image(img.get_data(), img.affine)

            nb.save(farray_img, copy_filename)
            img = None
            farray_img = None
            file = str(file)
            os.remove(file)
            os.rename(copy_filename, file)

