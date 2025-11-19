# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 19:44:29 2025

@author: tanja
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors, plot_matched_features, SIFT

# Function for loading images in grey scale and converting to smaller size for faster runtimes
def load_images(path):
    image_storage = []
    for p in path:
        # greyscale
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]

        # resize
        img_small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        image_storage.append(img_small)       

    return image_storage

def load_TIF(path):
    image_storage = []
    for p in path:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        img = cv2.convertScaleAbs(img, alpha=255.0/65535.0)

        img = cv2.bitwise_not(img)

        img = cv2.equalizeHist(img)
        
        h, w = img.shape
        
        # resize to working size
        img_small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        image_storage.append(img_small)

    return image_storage

def sift_extractor(img, extractor):
    extractor.detect_and_extract(img)
    keypoints = extractor.keypoints
    descriptors = extractor.descriptors

    return keypoints, descriptors

def match_features(img1, img2, extractor, num, plot):
    keypoints1, descriptors1 = sift_extractor(img1, extractor)
    keypoints2, descriptors2 = sift_extractor(img2, extractor)

    match = match_descriptors(
        descriptors1, descriptors2, max_ratio=0.77, cross_check=True)

    if (plot == True):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_matched_features(img1, img2, keypoints0=keypoints1,
                              keypoints1=keypoints2, matches=match, ax=ax)
        ax.set_title(f"Pair number {num + 1}")
        plt.tight_layout()
        plt.show()
    
    return match, keypoints1, keypoints2

def estimate_s_R_t(P, Q):           # different
    p_mean = np.mean(P, axis=0)
    q_mean = np.mean(Q, axis=0)
    
    Pc = P - p_mean
    Qc = Q - q_mean
    
    S = Pc.T @ Qc   
    
    U, Sigma, Vt = np.linalg.svd(S)
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    s = Sigma.sum() / np.sum(Pc**2)

    t = q_mean - s * (R @ p_mean)

    return s, R, t

def error_similarity(P, Q, s, R, t): # changed
    # predicts sR P + t, compares to Q
    return np.linalg.norm((P @ (s*R).T) + t - Q, axis=1)
    
def ransac_similarity(kp1, kp2, matches, thres=3.0): #changed
    ran = np.random.default_rng(0)

    # skimage keypoints are (row, col)
    kp1_xy = kp1[:, ::-1]
    kp2_xy = kp2[:, ::-1]

    matches = np.asarray(matches)
    M = len(matches)

    best_inliers = np.zeros(M, dtype=bool)

    # Pre-compute all matched coordinates once
    P_all = kp1_xy[matches[:, 0], :]
    Q_all = kp2_xy[matches[:, 1], :]

    for _ in range(1000):
        # Pick two random matches
        gr = ran.choice(M, size=2, replace=False)

        # Indices into kp1 and kp2
        i = matches[gr, 0]
        j = matches[gr, 1]

        # Extract the two matched points
        P = kp1_xy[i, :]       # shape (2,2)
        Q = kp2_xy[j, :]       # shape (2,2)

        # Hypothesis s,R,t
        s, R, t = estimate_s_R_t(P, Q)

        # Calculate reprojection errors on all matches
        errors = error_similarity(P_all, Q_all, s, R, t)
        inliers = errors < thres

        # Keep the best inlier set
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # Refit R,t on all inliers
    P_in = P_all[best_inliers, :]
    Q_in = Q_all[best_inliers, :]

    s_best, R_best, t_best = estimate_s_R_t(P_in, Q_in)

    return s_best, R_best, t_best, best_inliers

   # Magnitude of the translation vector in pixels
   # Rotation angle in degrees

if __name__ == "__main__":
    map_HE = Path(
        r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 1\Collection 2\HE")
    map_TRF = Path(
        r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 1\Collection 2\TRF")

    path_HE = sorted(map_HE.glob("*.jpg"))
    path_TRF = sorted(map_TRF.glob("*.TIF"))

    extractor_SIFT = SIFT()

    HE = load_images(path_HE)
    TRF = load_TIF(path_TRF)

    img1 = HE[0]
    img2 = TRF[0]

    matches = []
    P = []
    Q = []

    for i in range(len(HE)):
        match, kp1, kp2 = match_features(HE[i], TRF[i], extractor_SIFT, i, plot=False)
        s_best, R_best, t_best, inliers = ransac_similarity(kp1, kp2, match, thres=3.0)
    
        # Normal overlay with RANSAC result
        M = np.hstack([(s_best * R_best), t_best.reshape(2,1)]).astype(np.float32)
        h2, w2 = TRF[i].shape[:2]
        warped = cv2.warpAffine(HE[i], M, (w2, h2), flags=cv2.INTER_LINEAR)
    
        plt.figure(figsize=(6,6))
        plt.imshow(0.5*TRF[i] + 0.5*warped, cmap="gray")
        plt.title(f"Similarity transformation, overlay after RANSAC, pair {i+1}")
        plt.axis("off")
        plt.show()
        
        theta_deg = np.degrees(np.arctan2(R_best[1,0], R_best[0,0]))
        t = float(np.linalg.norm(t_best))
        s = float(s_best)
        print(f"Pair {i+1}: inliers {inliers.sum()}/{len(match)}, Rotation = {theta_deg:.2f}°, Translation = {t:.2f}, Scale = {s:.4f}")
