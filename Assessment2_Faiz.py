# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:47:56 2021

@author: zhirobyte
"""


#import library yang dibutuhkan  
import cv2
import numpy as np
from matplotlib import pyplot as plt

#import gambar dan resource dari citra data
im1 = cv2.imread('covid_image.png')
im2 = cv2.imread('mask.png')


#masukkan method bitwise untuk mendapatkan hasil dari gabungan kedua gambar
bitwiseAnd = cv2.bitwise_and(im1,im2)/255.0


#memasukkan method untuk mengurangi jumlah noise pada gambar sebelumnya yaiut gambar bitwise
rmNoise = cv2.pow(bitwiseAnd,2.2)

#memperbaiki kontras
rnimg = cv2.imread("res_noise_removal.jpeg", 0)#mengambil data gambar noise removal
equ = cv2.equalizeHist(rnimg)

#----------------------------------------
img = cv2.imread('res_contrast.jpeg',0)
histr1 = cv2.calcHist([img],[0],None,[256],[0,256])
#memunculkan data plot histogram awal
#----------------------------------------
img2 = cv2.imread('res_final.jpeg',0)
histr2 = cv2.calcHist([img2],[0],None,[256],[0,256])



#THRESHOLDING
#----------------------------------------
final = cv2.imread('res_contrast.jpeg',0)
ret,thresh1 = cv2.threshold(final,127,255,cv2.THRESH_BINARY)


#tampilkan gambar output 
#----------------------------------------
cv2.imshow("hasil bitwise", bitwiseAnd)
cv2.imshow("hasil blur noise",rmNoise)
cv2.imshow("hasil contrast",equ)
cv2.imshow("hasil final",thresh1)
plt.plot(histr1)
plt.plot(histr2)
plt.show()

#----------------------------------------

#menyimpan hasil gambar pada program 
#----------------------------------------
cv2.imwrite("res_bitwise.jpeg", bitwiseAnd)
cv2.imwrite("res_noise_removal.jpeg", 255*rmNoise)
cv2.imwrite("res_contrast.jpeg",equ)
cv2.imwrite("res_final.jpeg",thresh1)



cv2.waitKey(0)