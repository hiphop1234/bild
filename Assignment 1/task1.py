# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:20:36 2025

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

        # plt.figure()
        # plt.imshow(img_small, cmap="gray")
        # plt.title(p.name)
        # plt.show()

    return image_storage

def sift_extractor(img, extractor):
    extractor.detect_and_extract(img)
    keypoints = extractor.keypoints
    descriptors = extractor.descriptors

    return keypoints, descriptors

def match_features(img1, img2, extractor, num, plot):
    keypoints1, descriptors1 = sift_extractor(img1, extractor)
    keypoints2, descriptors2 = sift_extractor(img2, extractor)
    
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        match = np.empty((0, 2), dtype=int)

    match = match_descriptors(
        descriptors1, descriptors2, max_ratio=0.77, cross_check=True)

    if (plot == True):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_matched_features(img1, img2, keypoints0=keypoints1,
                              keypoints1=keypoints2, matches=match, ax=ax)
        ax.set_title(f"Pair number {num + 1}")
        plt.tight_layout()
        plt.show()

    #print(f"Matches for pair number {num+1}, are: {match}, there are {len(match)} number of matches")
    
    return match, keypoints1, keypoints2

def estimate_R_t(P, Q):
    # Compute centroids
    p_mean = np.mean(P, axis=0)
    q_mean = np.mean(Q, axis=0)

    # Center the points
    Pc = P - p_mean
    Qc = Q - q_mean

    # Covariance matrix
    S = Pc.T @ Qc

    # SVD
    U, Sigma, Vt = np.linalg.svd(S)

    # Rotation no stretching
    R = Vt.T @ U.T

    # If determinant is negative we fix it by changing the signs of V
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate translation
    t = q_mean - R @ p_mean

    return R, t

# Error calculation, the Orthogonal Procrustes problem
def error(P, Q, R, t):
    return np.linalg.norm((P @ R.T) + t - Q, axis=1)
    
def ransac_rigid(kp1, kp2, matches, thres=3.0):
    ran = np.random.default_rng(0)

    # skimage keypoints are (row, col). Convert to (x, y).
    kp1_xy = kp1[:, ::-1]
    kp2_xy = kp2[:, ::-1]

    matches = np.asarray(matches)
    M = len(matches)

    best_inliers = np.zeros(M, dtype=bool)

    # All matched coordinates
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

        # Hypothesis R,t
        R, t = estimate_R_t(P, Q)

        # Calculate reprojection errors on all matches
        errors = error(P_all, Q_all, R, t)
        inliers = errors < thres

        # Keep the best inlier set
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # Refit R,t on all inliers
    P_in = P_all[best_inliers, :]
    Q_in = Q_all[best_inliers, :]

    R_best, t_best = estimate_R_t(P_in, Q_in)

    return R_best, t_best, best_inliers

   # Magnitude of the translation vector in pixels
   # Rotation angle in degrees

if __name__ == "__main__":
    map_HE = Path(
        r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 1\Collection 1\HE")
    map_AMACR = Path(
        r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 1\Collection 1\p63AMACR")

    path_HE = sorted(map_HE.glob("*.bmp"))
    path_AMACR = sorted(map_AMACR.glob("*.bmp"))

    extractor_SIFT = SIFT()

    HE = load_images(path_HE)
    AMACR = load_images(path_AMACR)

    img1 = HE[0]
    img2 = AMACR[0]

    # match_features(img1,img2,extractor_SIFT,1)
    matches = []
    P = []
    Q = []

   # for i in range(len(HE)):
   #     match, kp1, kp2 = match_features(HE[i], AMACR[i], extractor_SIFT, i, False)
   #     matches.append(match)
   #     P.append(kp1)
   #     Q.append(kp2)
        
   # ransac_rigid(P, Q, matches, thres=0.5)

    for i in range(len(HE)):
        match, kp1, kp2 = match_features(HE[i], AMACR[i], extractor_SIFT, i, plot=False)
    #    
    #    if match is None or len(match) < 2:
    #        print(f"Pair {i+1}: too few matches ({0 if match is None else len(match)}), skipping.")
    #        continue
    #    
        R_best, t_best, inliers = ransac_rigid(kp1, kp2, match, thres=3.0)
    
        # Normal overlay with RANSAC result
        M = np.hstack([R_best, t_best.reshape(2,1)]).astype(np.float32)
        h2, w2 = AMACR[i].shape[:2]
        warped = cv2.warpAffine(HE[i], M, (w2, h2), flags=cv2.INTER_LINEAR)
    
        plt.figure(figsize=(6,6))
        plt.imshow(0.5*AMACR[i] + 0.5*warped, cmap="gray")
        plt.title(f"Rigid transformation, overlay after RANSAC, pair {i+1}")
        plt.axis("off")
        plt.show()
    
        theta_deg = np.degrees(np.arctan2(R_best[1,0], R_best[0,0]))
        t_mag = float(np.linalg.norm(t_best))
        print(f"Pair {i+1}: inliers {inliers.sum()}/{len(match)}, Rotation = {theta_deg:.2f}, Translation = {t_mag:.2f}")
        
    # image_for_size = cv2.imread(str(path_HE[1]), cv2.IMREAD_GRAYSCALE)

    # plt.figure()
    # plt.imshow(image_for_size, cmap="gray")
    # plt.show()

    # h,w = image_for_size.shape[:2]
    # print(h)
    # print(w)

    # According to the print-out, the size of an image is originally 1066x1147

    # Load all the images, in grayscale and correct size

# Instructions
# convert images to grayscale - OK
# downscale images to half the size- a quarter of the number of pixels to reduce runtimes - OK
# with the TRF images- beneficial to make an inversion + histogram equalisation
