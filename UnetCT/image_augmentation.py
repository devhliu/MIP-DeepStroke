import scipy.ndimage as sn
import random
import numpy as np


def normalize(v, new_max=1.0, new_min=0.0):
    min_v = np.min(v)
    max_v = np.max(v)
    if max_v==min_v:
        return v

    v_new = (v-min_v)*(new_max-new_min)/(max_v-min_v) + new_min
    return v_new


def rotate3D(imgsx, imgsy, pitch, yaw, roll, reshape=False):
    def _rotateArray(x, pitch, yaw, roll, reshape=False):
        # pitch
        pitched = sn.rotate(x, angle=pitch, axes=(0, 2), reshape=reshape)
        # yaw
        yawn = sn.rotate(pitched, angle=yaw, axes=(0, 1), reshape=reshape)
        # roll
        rolled = sn.rotate(yawn, angle=roll, axes=(1, 2), reshape=reshape)
        return rolled

    rotatedXs = []
    rotatedYs = []
    for x in imgsx:
        rotatedXs.append(_rotateArray(x, pitch, yaw, roll, reshape))
    for y in imgsy:
        rotatedYs.append(_rotateArray(y, pitch, yaw, roll, reshape))
    return rotatedXs, rotatedYs


def adjust_contrast(imgsx, imgsy, contrast=1.0, brightness=1.0):
    def _normalize(v, new_max=1.0, new_min=0.0):
        new_max = float(new_max)
        new_min = float(new_min)
        min_v = float(v.min())
        max_v = float(v.max())

        arr = np.array((v - min_v) * (new_max - new_min))
        arr = np.true_divide(arr, (max_v - min_v), out=None)
        v_new = arr + new_min
        return v_new

    def _adjust_contrast(img, contrast=1.0, brightness=1.0):
        max_v = img.max()
        min_v = img.min()
        if (min_v < 0 or max_v > 1):
            raise Exception("Image should be between 0.0 and 1.0")
        img_base = img.copy()
        img = _normalize(img_base, new_max=1.0, new_min=0.0)
        imgc = contrast * (img - 0.5) + 0.5 + (brightness - 1.0)
        imgc[imgc > 1] = 1
        imgc[imgc < 0] = 0
        # img_norm = _normalize(imgc, new_max=max_v, new_min=min_v)
        return imgc

    imgs_c = imgsx.copy()
    contrasted = []
    for x in imgs_c:
        b = _adjust_contrast(x, contrast, brightness)
        contrasted.append(b)

    return contrasted, imgsy


def salt_and_pepper(imgsx, imgsy, salt_vs_pepper=0.2, amount=0.04):
    def _salt_and_pepper(x, salt_vs_pepper=0.2, amount=0.04):
        num_salt = np.ceil(amount * x.shape[0] * x.shape[1] * x.shape[2] * salt_vs_pepper)
        num_pepper = np.ceil(amount * x.shape[0] * x.shape[1] * x.shape[2] * (1.0 - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in x.shape]
        x[coords[0], coords[1], coords[2]] = np.max(x)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in x.shape]
        x[coords[0], coords[1], coords[2]] = np.min(x)
        return x

    imgs_c = imgsx.copy()
    salted = []
    for x in imgs_c:
        salted.append(_salt_and_pepper(x, salt_vs_pepper, amount))

    return salted, imgsy


def flip(imgsx, imgsy, axis=0):
    def _flip(img, axis):
        img = np.flip(img, axis)
        return img

    flippedXs = []
    flippedYs = []
    for x in imgsx:
        flippedXs.append(_flip(x, axis))
    for y in imgsy:
        flippedYs.append(_flip(y, axis))
    return flippedXs, flippedYs


def randomly_augment(imgsx, imgsy, prob=0.15):

    # Normalize between 0 and 1
    imgsx = [normalize(x) for x in imgsx]
    imgsy = [normalize(y) for y in imgsy]

    # randomly rotate
    r = random.random()
    if r < prob:
        rotation = 1 if random.randint(-1, 1) > -1 else -1
        degx = rotation*int(random.random() * 10.0)  # in degree
        rotation = 1 if random.randint(-1, 1) > -1 else -1
        degy = rotation*int(random.random() * 10.0)  # in degree
        rotation = 1 if random.randint(-1, 1) > -1 else -1
        degz = rotation*int(random.random() * 10.0)  # in degree
        imgsx, imgsy = rotate3D(imgsx, imgsy, degx, degy, degz)

    # randomly add salt and pepper
    r = random.random()
    if r < prob:
        imgsx, imgsy = salt_and_pepper(imgsx, imgsy, salt_vs_pepper=0.4, amount=0.05)

    # randomly flip
    r = random.random()
    if r < prob:
        ax = random.randint(0, 2)
        imgsx, imgsy = flip(imgsx, imgsy, axis=ax)
        ax = random.randint(0, 2)
        imgsx, imgsy = flip(imgsx, imgsy, axis=ax)
        ax = random.randint(0, 2)
        imgsx, imgsy = flip(imgsx, imgsy, axis=ax)

    # randomly change contrast and brightness
    r = random.random()
    if r < prob:
        contrast = np.random.normal(loc=1.0, scale=0.3)
        brightness = np.random.normal(loc=1.0, scale=0.3)
        imgsx, imgsy = adjust_contrast(imgsx, imgsy, contrast, brightness)

    imgsx = [normalize(x) for x in imgsx]
    imgsy = [normalize(y) for y in imgsy]
    return imgsx, imgsy
