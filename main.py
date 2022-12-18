import random

import numpy as np
from PyQt5.QtGui import QImage, QPixmap

import process
from process import Ui_MainWindow

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtCore
import sys
import cv2
from matplotlib import pyplot as plt

import cmath
from math import log, ceil, sqrt, exp

# histogram function
def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1

    # return final result
    return histogram

# cumulative sum function
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def omega(p, q):
    """
    The omega term in DFT and IDFT formulas
    """
    return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def pad(lst):
    """
    padding the list to next nearest power of 2 as FFT implemented is radix 2
    """
    k = 0
    while 2**k < len(lst):
        k += 1
    return np.concatenate((lst, ([0] * (2 ** k - len(lst)))))

def pad2(x):
    m, n = np.shape(x)
    M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
    F = np.zeros((M,N), dtype = x.dtype)
    F[0:m, 0:n] = x
    return F, m, n

## FFT - 1D
def fft(x):
    """
    FFT of 1-d signals
    usage : X = fft(x)
    where input x = list containing sequences of a discrete time signals
    and output X = dft of x
    """

    n = len(x)
    if n == 1:
        return x
    Feven, Fodd = fft(x[0::2]), fft(x[1::2])
    combined = [0] * n
    for m in range(n/2):
        combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
        combined[m + n/2] = Feven[m] - omega(n, -m) * Fodd[m]
    return combined

## FFT - 2D
def fft2(f):
    """
    FFT of 2-d signals/images with padding
    usage X, m, n = fft2(x), where m and n are dimensions of original signal
    """

    f, m, n = pad2(f)
    return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
    """
    IFFT of 2-d signals
    usage x = ifft2(X, m, n) with unpaded,
    where m and n are odimensions of original signal before padding
    """

    f, M, N = fft2(np.conj(F))
    f = np.matrix(np.real(np.conj(f)))/(M*N)
    return f[0:m, 0:n]

## ifft - 1D
def ifft(X):
    """
    IFFT of 1-d signals
    usage x = ifft(X)
    unpadding must be done implicitly
    """

    x = fft([x.conjugate() for x in X])
    return [x.conjugate()/len(X) for x in x]

## ifft - 2D
def ifft2(F, m, n):
    """
    IFFT of 2-d signals
    usage x = ifft2(X, m, n) with unpaded,
    where m and n are odimensions of original signal before padding
    """

    f, M, N = fft2(np.conj(F))
    f = np.matrix(np.real(np.conj(f)))/(M*N)
    return f[0:m, 0:n]

def fftshift(F):
    """
    this shifts the centre of FFT of images/2-d signals
    """
    M, N = F.shape
    R1, R2 = F[0: M//2, 0: N//2], F[M//2: M, 0: N//2]
    R3, R4 = F[0: M//2, N//2: N], F[M//2: M, N//2: N]
    sF = np.zeros(F.shape,dtype = F.dtype)
    sF[M//2: M, N//2: N], sF[0: M//2, 0: N//2] = R1, R4
    sF[M//2: M, 0: N//2], sF[0: M//2, N//2: N]= R3, R2
    return sF

def DFT_1D(fx):
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    fu = fx.copy()

    for i in range(M):
        u = i
        sum = 0
        for j in range(M):
            x = j
            tmp = fx[x]*np.exp(-2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        # print(sum)
        fu[u] = sum
    # print(fu)

    return fu

def inverseDFT_1D(fu):
    fu = np.asarray(fu, dtype=complex)
    M = fu.shape[0]
    fx = np.zeros(M, dtype=complex)

    for i in range(M):
        x = i
        sum = 0
        for j in range(M):
            u = j
            tmp = fu[u]*np.exp(2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        fx[x] = np.divide(sum, M, dtype=complex)

    return fx

def FFT_1D(fx):
    """
    use recursive method to speed up
    """
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    minDivideSize = 4

    if M % 2 != 0:
        raise ValueError("the input size must be 2^n")

    if M <= minDivideSize:
        return DFT_1D(fx)
    else:
        fx_even = FFT_1D(fx[::2])  # compute the even part
        fx_odd = FFT_1D(fx[1::2])  # compute the odd part
        W_ux_2k = np.exp(-2j * np.pi * np.arange(M) / M)

        f_u = fx_even + fx_odd * W_ux_2k[:M//2]

        f_u_plus_k = fx_even + fx_odd * W_ux_2k[M//2:]

        fu = np.concatenate([f_u, f_u_plus_k])

    return fu

def inverseFFT_1D(fu):
    """
    use recursive method to speed up
    """
    fu = np.asarray(fu, dtype=complex)
    fu_conjugate = np.conjugate(fu)

    fx = FFT_1D(fu_conjugate)

    fx = np.conjugate(fx)
    fx = fx / fu.shape[0]

    return fx

def FFT_2D(fx):
    h, w = fx.shape[0], fx.shape[1]

    fu = np.zeros(fx.shape, dtype=complex)

    if len(fx.shape) == 2:
        for i in range(h):
            fu[i, :] = FFT_1D(fx[i, :])

        for i in range(w):
            fu[:, i] = FFT_1D(fu[:, i])
    elif len(fx.shape) == 3:
        for ch in range(3):
            fu[:, :, ch] = FFT_2D(fx[:, :, ch])

    return fu

def inverseDFT_2D(fu):
    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(h):
            fx[i, :] = inverseDFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = inverseDFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = inverseDFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx

def inverseFFT_2D(fu):
    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(h):
            fx[i, :] = inverseFFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = inverseFFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = inverseFFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

class main_WIN(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.open_pushButton.clicked.connect(self.openImage)

        self.save_pushButton.clicked.connect(self.saveImage)

        self.grayScale_checkBox.stateChanged.connect(self.convertToGray)

        self.histEqual_pushButton.clicked.connect(self.histogram_equalisation)
        self.plot_checkBox.stateChanged.connect(self.histogram_equalisation)

        self.lowPass_pushButton.clicked.connect(self.lowPass)
        self.highPass_pushButton.clicked.connect(self.highPass)

        self.gaussianLP_pushButton.clicked.connect(self.gaussianLowPass)
        self.gaussianHP_pushButton.clicked.connect(self.gaussianHighPass)

        self.butterworth_LP_pushButton.clicked.connect(self.butterworthLowPass)
        self.butterworthHP_pushButton.clicked.connect(self.butterworthHighPass)

    # Load an Image
    def loadImage(self, fname):
        self.image = cv2.imread(fname, cv2.IMREAD_COLOR)
        self.tmp = self.image
        self.displayImage(1)

    # Display image in window
    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if (len(self.image.shape) == 3):
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image,
                     self.image.shape[1],
                     self.image.shape[0],
                     self.image.strides[0],
                     qformat)

        img = img.rgbSwapped()
        if window == 1:
            self.ori_img.setPixmap(QPixmap.fromImage(img))
            self.ori_img.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 2:
            self.res_img.setPixmap(QPixmap.fromImage(img))
            self.res_img.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    # open Image from file
    def openImage(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open image', 'C:\\Users\\User\\PycharmProjects\\imageProcessing\\img', "Image File (*)")
        if fname:
            self.loadImage(fname)
            self.displayImage(1)
        else:
            print("Invalid image")

    # Reset image to original version
    def resetImage(self):
        self.image = self.tmp
        self.displayImage(2)

    # Save image
    def saveImage(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save image as ', 'E:\\ImageProcessing\imgs', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image)
            print("ERROR")

    # convert image to grayscale
    def convertToGray(self):
        if self.grayScale_checkBox.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.displayImage(1)
            self.displayImage(2)

    def histogram_equalisation(self):
        self.image = self.tmp
        self.convertToGray()
        self.image = np.asarray(self.image)

        flat_img = self.image.flatten()

        # show the histogram
        #plt.hist(flat_img, bins=50)
        #plt.show()

        hist = get_histogram(flat_img, 256)

        # execute the fn
        cs = cumsum(hist)
        # display the result
        #plt.plot(cs)
        #plt.show()

        # numerator & denomenator
        nj = (cs - cs.min()) * 255
        N = cs.max() - cs.min()

        # re-normalize the cumsum
        cs = nj / N

        # cast it back to uint8 since we can't use floating point values in images
        cs = cs.astype('uint8')
        #plt.plot(cs)
        #plt.show()

        # get the value from cumulative sum for every index in flat, and set that as img_new
        img_new = cs[flat_img]

        # put array back into original shape since we flattened it
        img_new = np.reshape(img_new, self.image.shape)

        if self.plot_checkBox.isChecked():
            print('plot')
            ax = plt.hist(self.image.ravel(), bins=256)

            ax1 = plt.hist(img_new.ravel(), bins=256)
            plt.legend(['Original Image', 'Equalized Image'])
            plt.show()

        self.image = img_new
        self.displayImage(2)

    def lowPass(self):
        self.image = self.tmp
        self.convertToGray()

        original = FFT_2D(self.image)
        # plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

        center = fftshift(original)
        # plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

        LowPassCenter = center * idealFilterLP(50, self.image.shape)
        # plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")

        LowPass = fftshift(LowPassCenter)
        # plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

        inverse_LowPass = inverseFFT_2D(LowPass)
        #plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

        filtered_img = np.abs(inverse_LowPass)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        #plt.imshow(filtered_img, "gray"), plt.title("Processed Image1")

        #plt.show()

        self.image = filtered_img
        print('done')
        self.displayImage(2)

    def highPass(self):
        self.image = self.tmp
        self.convertToGray()

        original = FFT_2D(self.image)
        # plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

        center = fftshift(original)
        # plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

        HighPassCenter = center * idealFilterHP(50, self.image.shape)
        # plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply High Pass Filter")

        HighPass = fftshift(HighPassCenter)
        # plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

        inverse_HighPass = inverseFFT_2D(HighPass)
        #plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

        # expand the result such that all values are between 0 and 255
        filtered_img = np.abs(inverse_HighPass)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        #plt.imshow(filtered_img, "gray"), plt.title("Processed Image1")

        #plt.show()

        self.image = filtered_img
        print('done')
        self.displayImage(2)

    def gaussianLowPass(self):
        self.image = self.tmp
        self.convertToGray()

        original = FFT_2D(self.image)
        # plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

        center = fftshift(original)
        # plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

        LowPassCenter = center * gaussianLP(50, self.image.shape)
        # plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")

        LowPass = fftshift(LowPassCenter)
        # plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

        inverse_LowPass = inverseFFT_2D(LowPass)
        #plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

        filtered_img = np.abs(inverse_LowPass)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        #plt.imshow(filtered_img, "gray"), plt.title("Processed Image1")

        #plt.show()

        self.image = filtered_img
        print('done')
        self.displayImage(2)

    def gaussianHighPass(self):
        self.image = self.tmp
        self.convertToGray()

        original = FFT_2D(self.image)
        # plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

        center = fftshift(original)
        # plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

        HighPassCenter = center * gaussianHP(50, self.image.shape)
        # plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply High Pass Filter")

        HighPass = fftshift(HighPassCenter)
        # plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

        inverse_HighPass = inverseFFT_2D(HighPass)
        #plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

        # expand the result such that all values are between 0 and 255
        filtered_img = np.abs(inverse_HighPass)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        #plt.imshow(filtered_img, "gray"), plt.title("Processed Image1")

        #plt.show()

        self.image = filtered_img
        print('done')
        self.displayImage(2)

    def butterworthLowPass(self):
        self.image = self.tmp
        self.convertToGray()

        original = FFT_2D(self.image)
        # plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

        center = fftshift(original)
        # plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

        LowPassCenter = center * butterworthLP(50, self.image.shape, 10)
        # plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")

        LowPass = fftshift(LowPassCenter)
        # plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

        inverse_LowPass = inverseFFT_2D(LowPass)
        # plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

        filtered_img = np.abs(inverse_LowPass)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        # plt.imshow(filtered_img, "gray"), plt.title("Processed Image1")

        # plt.show()

        self.image = filtered_img
        print('done')
        self.displayImage(2)

    def butterworthHighPass(self):
        self.image = self.tmp
        self.convertToGray()

        original = FFT_2D(self.image)
        # plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")

        center = fftshift(original)
        # plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")

        HighPassCenter = center * butterworthHP(50, self.image.shape, 10)
        # plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply High Pass Filter")

        HighPass = fftshift(HighPassCenter)
        # plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

        inverse_HighPass = inverseFFT_2D(HighPass)
        # plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

        # expand the result such that all values are between 0 and 255
        filtered_img = np.abs(inverse_HighPass)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        # plt.imshow(filtered_img, "gray"), plt.title("Processed Image1")

        # plt.show()

        self.image = filtered_img
        print('done')
        self.displayImage(2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = main_WIN()
    win.setWindowTitle('3820221076 王艺瑾')
    win.show()
    sys.exit(app.exec())