import os
import nibabel as nb

import numpy as np
import nibabel as nb
import os
from tqdm import tqdm

input_dir = "C:\\Users\simon\Documents\EPFL\Master\Semestre5\Projet MIPLAB\DataTest"

patients = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if "excluded" not in x]
# through each patients
for p in tqdm(patients):
    #through each folder of the patient
    folders = [os.path.join(p, x) for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))]
    for f in folders:
        #through each files of the folder
        files = [os.path.join(f, x) for x in os.listdir(f) if x.endswith(".nii") and x.startswith("rcoreg")]
        if len(files) < 3:
            print("MISSING FILES IN {}".format(p))

        shape = []
        file_shape = []
        for file in files:
            img = nb.load(file).get_data()
            shape_actual = img.shape
            if len(shape)==0:
                shape = shape_actual
            if shape is not None and shape_actual != shape:
                print("{} - SHAPE {}!={}".format(os.path.basename(p), shape_actual, shape))
            else:
                shape = shape_actual

        for f in files:
            print("{}-{}".format(os.path.basename(f), nb.load(f).get_data().shape))

