# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 22:44:07 2025

@author: tanja
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def load_TIF(path):
    image_storage = []
    for p in path:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        img = cv2.convertScaleAbs(img, alpha=255.0/65535.0)

        img = cv2.bitwise_not(img)

        img = cv2.equalizeHist(img)
        h, w = img.shape
        # resize to your working size
        img_small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        image_storage.append(img_small)

    return image_storage

trf_folder = Path(r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 1\Collection 2\TRF")

paths = list(trf_folder.glob("*.TIF"))

trf_imgs = load_TIF(paths)

for i, img in enumerate(trf_imgs, start=1):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Pair {i}")  
    plt.axis("off")
    plt.show()