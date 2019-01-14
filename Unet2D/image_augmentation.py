import scipy.ndimage as sn
import random
import numpy as np
import cv2
from PIL import Image

def normalize(v, new_max=1.0, new_min=0.0):
    v = np.nan_to_num(v)
    min_v = np.min(v)
    max_v = np.max(v)
    if max_v==min_v:
        return v

    v_new = (v-min_v)*(new_max-new_min)/(max_v-min_v) + new_min
    return v_new


def rotate2D(imgsx, imgsy, angle, reshape=False):
    def _rotateArray(x, angle, reshape=False):
        x = np.nan_to_num(x)
        img = Image.fromarray(x)
        # pitch
        rotated = img.rotate(angle)
        return np.asarray(rotated)

    rotatedXs = []
    rotatedYs = []
    for x in imgsx:
        rotatedXs.append(_rotateArray(x, angle, reshape))
    for y in imgsy:
        rotatedYs.append(_rotateArray(y, angle, reshape))
    return rotatedXs, rotatedYs


def adjust_contrast(imgsx, imgsy, contrast=1.0, brightness=1.0):
    def _normalize(v, new_max=1.0, new_min=0.0):
        v_new = normalize(v, new_max, new_min)
        return v_new

    def _adjust_contrast(img, contrast=1.0, brightness=1.0):
        img = np.nan_to_num(img)
        max_v = img.max()
        min_v = img.min()
        if (min_v < 0 or max_v > 1):
            raise Exception("Image should be between 0.0 and 1.0, found {} and {}".format(min_v, max_v))
        img_base = img.copy()
        img = _normalize(img_base, new_max=1.0, new_min=0.0)
        imgc = contrast * (img - 0.5) + 0.5 + (brightness - 1.0)
        imgc = np.nan_to_num(imgc)
        imgc[imgc >= 1] = 1
        imgc[imgc <= 0] = 0
        # img_norm = _normalize(imgc, new_max=max_v, new_min=min_v)
        return imgc

    imgs_c = imgsx.copy()
    contrasted = []
    for x in imgs_c:
        x = normalize(x)
        b = _adjust_contrast(x, contrast, brightness)
        contrasted.append(b)

    return contrasted, imgsy


def salt_and_pepper(imgsx, imgsy, salt_vs_pepper=0.2, amount=0.04):
    def _salt_and_pepper(x, salt_vs_pepper=0.2, amount=0.04):
        x = np.nan_to_num(x)
        num_salt = np.ceil(amount * x.shape[0] * x.shape[1] * salt_vs_pepper)
        num_pepper = np.ceil(amount * x.shape[0] * x.shape[1] * (1.0 - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in x.shape]
        x[coords[0], coords[1]] = np.max(x)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in x.shape]
        x[coords[0], coords[1]] = np.min(x)
        return x

    imgs_c = imgsx.copy()
    salted = []
    for x in imgs_c:
        salted.append(_salt_and_pepper(x, salt_vs_pepper, amount))

    return salted, imgsy


def flip(imgsx, imgsy, axis=0):
    def _flip(img, axis):
        img = np.nan_to_num(img)
        img = np.flip(img, axis)
        return img

    flippedXs = []
    flippedYs = []
    for x in imgsx:
        flippedXs.append(_flip(x, axis))
    for y in imgsy:
        flippedYs.append(_flip(y, axis))
    return flippedXs, flippedYs

def zoom(imgsx, imgsy, zoom=0):

    def _zoom(image, zoom_pix):
        IMAGE_HEIGHT = image.shape[0]
        IMAGE_WIDTH = image.shape[1]
        zoom_factor = 1 + (2*zoom_pix)/IMAGE_HEIGHT
        image = cv2.resize(image, None, fx=zoom_factor,
                           fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        top_crop = (image.shape[0] - IMAGE_HEIGHT)//2
        left_crop = (image.shape[1] - IMAGE_WIDTH)//2
        image = image[top_crop: top_crop+IMAGE_HEIGHT,
                      left_crop: left_crop+IMAGE_WIDTH]
        return image

    zoomedXs = []
    zoomedYs = []
    for x in imgsx:
        zoomedXs.append(_zoom(x, zoom))
    for y in imgsy:
        zoomedYs.append(_zoom(y, zoom))
    return zoomedXs, zoomedYs

def randomly_augment(imgsx, imgsy, prob={"rotation": 0.15,
                                         "rotxmax": 90.0,
                                         "rotation_step": 1.0,
                                         "salt_and_pepper": 0.15,
                                         "flip": 0.15,
                                         "contrast_and_brightness": 0.15}):

    def get_random_angle(rotation_max, rotation_step):
        if rotation_max == 0:
            return 0
        # rotate forward or backward
        rotation = 1 if random.randint(-1, 1) > -1 else -1
        deg_rotation = rotation*random.randrange(0, rotation_max, rotation_step)
        return deg_rotation

    # randomly rotate
    r = random.random()
    if r < prob["rotation"]:
        degx = get_random_angle(prob["rotxmax"], prob["rotation_step"])
        imgsx, imgsy = rotate2D(imgsx, imgsy, degx)

    # randomly add salt and pepper
    r = random.random()
    if r < prob["salt_and_pepper"]:
        imgsx, imgsy = salt_and_pepper(imgsx, imgsy, salt_vs_pepper=0.4, amount=0.05)

    # randomly flip
    r = random.random()
    if r < prob["flip"]:
        ax = random.randint(0, 1)
        imgsx, imgsy = flip(imgsx, imgsy, axis=ax)
        ax = random.randint(0, 1)
        imgsx, imgsy = flip(imgsx, imgsy, axis=ax)

    # randomly zoom
    r = random.random()
    if r < prob["zoom"]:
        zoom_value = random.randint(0, 10)
        imgsx, imgsy = zoom(imgsx, imgsy, zoom=zoom_value)
    # Normalize between 0 and 1 to be sure that everything is correct
    #imgsx = [normalize(x) for x in imgsx]
    #imgsy = [normalize(y) for y in imgsy]

    # randomly change contrast and brightness
    r = random.random()
    if r < prob["contrast_and_brightness"]:
        contrast = np.random.normal(loc=1.0, scale=0.3)
        brightness = np.random.normal(loc=1.0, scale=0.3)
        imgsx, imgsy = adjust_contrast(imgsx, imgsy, contrast, brightness)

    imgsx = [normalize(x) for x in imgsx]
    imgsy = [normalize(y) for y in imgsy]
    return imgsx, imgsy
