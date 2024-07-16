# -*- coding: utf-8 -*

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from numpy import linalg as la
from sklearn.decomposition import TruncatedSVD

import time


def build_matrix(people_count):
    pixelsNumber = 10304
    A = np.zeros((pixelsNumber, 10 * people_count))

    folderPath = 'att_faces'
    folders = os.listdir(folderPath)
    j = 0
    k = 0
    for folder in folders:
        for i in range(10):
            if(folder != '.DS_Store'):
                photo = cv2.imread(f'{folderPath}/{folder}/{(i + 1)}.pgm', 0)
                photo = np.array(photo)
                photo = photo.reshape(-1, )
                A[:, j] = photo
                j = j + 1

                if j == 10 * people_count:
                    return A

#principal component analysis
def PCA(A, k):
    mean = np.mean(A, axis=1)
    A = A - mean.reshape(A.shape[0], 1)

    start_time = time.time()
    L = A.T @ A  # optimizata
    d, v = la.eigh(L)  # v: 320x320 sort
    v = A @ v  # 10304x320 * 320x320 = 10304x320

    d_idx = np.argsort(d)[::-1]
    d_idx_k = d_idx[:k]

    HQPB = v[:, d_idx_k]  # 10304xk
    # projections = A.T@HQPB # 320x10304 * 10304xk = 320xk

    return HQPB, time.time() - start_time

#singular value decomposition
def SVD(A, k):
    mean = np.mean(A, axis=1)
    A = A - mean.reshape(A.shape[0], 1)

    start_time = time.time()

    svd = TruncatedSVD(n_components=k, random_state=0)
    svd.fit_transform(A.T)
    HQPB = svd.components_.T

    return HQPB, time.time() - start_time


def statistics():
    k_elements = [20, 40, 60, 80, 100]

    pca_times = []
    svd_times = []

    A = build_matrix(20)

    for k in k_elements:
        _, pca_time = PCA(A, k)
        _, svd_time = SVD(A, k)

        pca_times.append(pca_time)
        svd_times.append(svd_time)

    return pca_times, svd_times


def plot_pca():
    A = build_matrix(20)

    pca, _ = PCA(A, 20)

    for i in range(pca.shape[1]):
        plt.imshow(pca[:, i].reshape(112, 92), cmap='gray')
        plt.title(f"PCA - Persoana {i + 1}")
        plt.show()


def plot_svd():
    A = build_matrix(20)

    svd, _ = SVD(A, 20)

    for i in range(svd.shape[1]):
        plt.imshow(svd[:, i].reshape(112, 92), cmap='gray')
        plt.title(f"SVD - Persoana {i + 1}")
        plt.show()

A = build_matrix(20)
svd, _ = SVD(A, 20)

print(statistics())
plot_pca()
plot_svd()
