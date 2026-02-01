#augumentations
import cv2
import numpy as np
import random

def random_rotate(image, angle_range=(-25,25)):
    angle = random.uniform(*angle_range)
    h,w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
    return cv2.warpAffine(image, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)

def random_flip(image):
    if random.random()<0.5:
        return cv2.flip(image, 1)
    return(image)

def random_brightness(image, factor_range=(0.6,1.4)):
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:,:,2] = hsv[:,:,2]*factor
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0 , 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_zoom(image, zoom_range=(0.8, 1.2)):
    zoom_factor = random.uniform(*zoom_range)
    h,w = image.shape[:2]

    if zoom_factor>1:
        new_h, new_w = int(h/zoom_factor), int(w/zoom_factor)
        y1 = (h - new_h)//2
        x1 = (w - new_w)//2
        cropped = image[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w,h))
    else:
        new_h, new_w = int(h*zoom_factor), int(w*zoom_factor)
        resized = cv2.resize(image, (new_w, new_h))
        pad_y = (h-new_h)//2
        pad_x = (w-new_w)//2
        padded = cv2.copyMakeBorder(resized, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT_101)
        return padded


def random_shift(image, shift_range=0.2):
    h,w = image.shape[:2]
    max_dx = w*shift_range
    max_dy = h*shift_range

    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)
    M = np.float32([[1,0, dx], [0,1,dy]])
    return cv2.warpAffine(image, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)

def random_blur(image, ksize=(3,5)):
    if random.random()<0.3:
        k = random.choice(ksize)
        return cv2.GaussianBlur(image, (k,k), 0)
    return image

def random_noise(image):
    if random.random()<0.3:
        noise = np.random.normal(0,10, image.shape).astype(np.int16)
        noisy = image.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    return image


def augment_image(image):
    image = random_rotate(image)
    image = random_flip(image)
    image = random_brightness(image)
    image = random_zoom(image)
    image = random_shift(image)
    image = random_blur(image)
    image = random_noise(image)
    return image