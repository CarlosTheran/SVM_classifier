#!/usr/bin/env python
from scipy.io import loadmat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import numpy.matlib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# pdb.set_trace() It is used to debug your code, it works as break point


def main():
    #pdb.set_trace()
    # Start coding
    classes_data = loadmat('/home/carlos/Documents/data_houston/classes.mat')
    data_hs = loadmat(
            '/home/carlos/Documents/data_houston/houston_hsi_hiat.mat')
    data_lidar = loadmat(
            '/home/carlos/Documents/data_houston/houston_lidar_hiat.mat')
    lidar_img = data_lidar.get("pixels")
    lidar_cube_img = lidar_img.reshape(data_lidar.get("number_of_rows")[
                                           0][0], data_lidar.get("number_of_columns")[0][0])
    hs_img = data_hs.get("pixels")
    hs_cube_img = hs_img.reshape(data_hs.get("number_of_bands")[0][0], data_hs.get(
            "number_of_rows")[0][0], data_hs.get("number_of_columns")[0][0])

        # show lidar image
    #fig = plt.figure()
    #ax = fig.add_subplot(111, label="lidar")
    #im = ax.imshow(lidar_cube_img)
    #plt.show()

    # show hs image
    #fig1 = plt.figure()
        #ax1 = fig1.add_subplot(211)
        # ax1.imshow(hs_cube_img[1,:,:])
        #plt.title('band number 1')
        #ax2 = fig1.add_subplot(212)
        # ax2.imshow(hs_cube_img[5,:,:])
        #plt.title('band number 5')
        # plt.show()

    # Loading classes structure
    classes_pixels = classes_data.get("classes")

    # Set parameters
    set_pca = 1
    number_of_bands = 10
    kernel = 'linear'

    #PCA
    bandpixels = pca_hiat(hs_img, number_of_bands)


    #Running classifiers
    svm_multiclassHSI(classes_pixels, kernel, bandpixels)

def svm_multiclassHSI(classes_pixels, kernel, bandpixels):

    n_classes = len(classes_pixels.T)
    label = []
    for i in range(n_classes):
        label.append(int(classes_pixels[0][i][0][0]))

    train_column = []
    test_column = []
    for i in range(n_classes):
        train_column.append(len(classes_pixels[0][i][1]))
        test_column.append(len(classes_pixels[0][i][2]))

    size_train = 0
    size_test = 0
    for i in range(n_classes):
        size_train = size_train + train_column[i]
        size_test = size_test + test_column[i]


# craeting label for classes  TRAINING & TESTING
    train_classes_label = np.matlib.repmat(label[0], train_column[0], 1)
    train_classes_label = train_classes_label.flatten()
    test_classes_label = np.matlib.repmat(label[0], test_column[0], 1)
    test_classes_label = test_classes_label.flatten()
    for i in range(1, n_classes):
        train_classes_label = np.append(train_classes_label, np.matlib.repmat(
            label[i], train_column[i], 1).flatten())
        test_classes_label = np.append(test_classes_label, np.matlib.repmat(
            label[i], test_column[i], 1).flatten())

# Building data for training and testing
    x_train = []
    for i in range(n_classes):
        for j in range(train_column[i]):
            x_train.append(bandpixels[:, classes_pixels[0][i][1][j][0]])
    x_train = np.array(x_train)

    x_test = []
    for i in range(n_classes):
        for j in range(test_column[i]):
            x_test.append(bandpixels[:, classes_pixels[0][i][2][j][0]])
    x_test = np.array(x_test)


# SVM classifier: x_train is row x bands
    svm = SVC()
    svm.fit(x_train, train_classes_label)
    print(svm.predict(x_test))
    img_predicted = svm.predict(bandpixels.T)
    pdb.set_trace()
def mat2gray(pixels):
    max_pixel = np.amax(pixels)
    min_pixel = np.amin(pixels)

    pixels_mat2gray = 1/(max_pixel-min_pixel)*(pixels-min_pixel)
    return pixels_mat2gray


def pca_hiat(pixels, number_of_bands):
    cov_matrix_bands = np.cov(pixels)
    U, s, V = np.linalg.svd(cov_matrix_bands)
    bandpixels = np.matmul(V[0:number_of_bands, :], pixels)
    bandpixels_norm = mat2gray(bandpixels)
    return bandpixels_norm


if __name__ == "__main__":
    main()
