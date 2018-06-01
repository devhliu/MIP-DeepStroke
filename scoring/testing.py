import numpy as np
import nibabel as nb
from UnetCT.predict import predict

def predict(image, model, patch_size):
    y_pred = predict(images=[image], model=model, patch_size=patch_size)