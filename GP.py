
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
Folder_name="image"
Extension=".png"
Folder_name="image"
def scale_image(image,fx,fy,i):
    image = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(Folder_name+"/Scale-"+str(fx)+str(fy)+ str(i)+Extension, image)

def translation_image(image,x,y,i):
    rows, cols ,c= image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Translation-" + str(x) + str(y)+ str(i) + Extension, image)

def rotate_image(image,deg,i):
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Rotate-" + str(deg) +str(i)+Extension, image)

def resize_image(image,w,h,i):
    image=cv2.resize(image,(w,h))
    cv2.imwrite(Folder_name+"/Resize-"+str(w)+"*"+str(h)+str(i)+Extension, image)

#crop
def crop_image(image,y1,y2,x1,x2,i):
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Folder_name+"/Crop-"+str(x1)+str(x2)+"*"+str(y1)+str(y2)+ str(i)+Extension, image)

def padding_image(image,topBorder,bottomBorder,leftBorder,rightBorder,i,color_of_border=[0,0,0]):
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    cv2.imwrite(Folder_name + "/padd-" + str(topBorder) + str(bottomBorder) + "*" + str(leftBorder) + str(rightBorder)+ str(i) + Extension, image)

def flip_image(image,dir,i):
    image = cv2.flip(image, dir)
    cv2.imwrite(Folder_name + "/flip-" + str(dir)+ str(i)+Extension, image)

def invert_image(image,channel,i):
    # image=cv2.bitwise_not(image)
    image=(channel-image)
    cv2.imwrite(Folder_name + "/invert-"+str(channel)+ str(i)+ Extension, image)

def add_light(image,i, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light-"+str(gamma)+ str(i)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark-" + str(gamma)+ str(i) + Extension, image)

def add_light_color(image, color,i, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light_color-"+str(gamma)+ str(i)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark_color" + str(gamma)+ str(i) + Extension, image)

def saturation_image(image,saturation,i):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/saturation-" + str(saturation)+ str(i) + Extension, image)

def hue_image(image,saturation,i):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/hue-" + str(saturation)+ str(i) + Extension, image)
def multiply_image(image,R,G,B,i):
    image=image*[R,G,B]
    cv2.imwrite(Folder_name+"/Multiply-"+str(R)+"*"+str(G)+"*"+str(B)+ str(i)+Extension, image)

def gausian_blur(image,blur,i):
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(Folder_name+"/GausianBLur-"+str(blur)+ str(i)+Extension, image)

def averageing_blur(image,shift,i):
    image=cv2.blur(image,(shift,shift))
    cv2.imwrite(Folder_name + "/AverageingBLur-" + str(shift)+ str(i) + Extension, image)

def median_blur(image,shift,i):
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(Folder_name + "/MedianBLur-" + str(shift)+ str(i) + Extension, image)

def bileteralBlur(image,d,color,space,i):
    image = cv2.bilateralFilter(image, d,color,space)
    cv2.imwrite(Folder_name + "/BileteralBlur-"+str(d)+"*"+str(color)+"*"+str(space)+ str(i)+ Extension, image)

def erosion_image(image,shift,i):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Erosion-"+"*"+str(shift)+ str(i) + Extension, image)

def dilation_image(image,shift,i):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Dilation-" + "*" + str(shift)+ str(i)+ Extension, image)

def opening_image(image,shift,i):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Folder_name + "/Opening-" + "*" + str(shift)+ str(i)+ Extension, image)

def closing_image(image, shift,i):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(Folder_name + "/Closing-" + "*" + str(shift)+ str(i) + Extension, image)

def morphological_gradient_image(image, shift,i):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(Folder_name + "/Morphological_Gradient-" + "*" + str(shift)+ str(i) + Extension, image)

def top_hat_image(image, shift,i):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Folder_name + "/Top_Hat-" + "*" + str(shift)+ str(i) + Extension, image)

def black_hat_image(image, shift,i):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(Folder_name + "/Black_Hat-" + "*" + str(shift)+ str(i) + Extension, image)
def sharpen_image(image,i):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name+"/Sharpen-"+ str(i)+Extension, image)

def emboss_image(image,i):
    kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1)+128
    cv2.imwrite(Folder_name + "/Emboss-"+ str(i) + Extension, image)

def edge_image(image,ksize,i):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(Folder_name + "/Edge-"+str(ksize) + str(i) + Extension, image)

def addeptive_gaussian_noise(image,i):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(Folder_name + "/Addeptive_gaussian_noise-"+ str(i) + Extension, image)

def contrast_image(image,contrast,i):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/Contrast-" + str(contrast) + str(i)+ Extension, image)

def edge_detect_canny_image(image,th1,th2 , i):
    image = cv2.Canny(image,th1,th2)
    cv2.imwrite(Folder_name + "/Edge Canny-" + str(th1) + "*" + str(th2)+ str(i) + Extension, image)

def grayscale_image(image,i):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name + "/Grayscale-" + str(i)+ Extension, image)


def transformation_image(image,i):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(1)+ str(i) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [0, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(2)+ str(i) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [30, 175]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(3)+ str(i) + Extension, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [70, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(4) + str(i)+ Extension, image)


path = "C:\\Users\\Owner\\Desktop\\EUS\\*.png"
inn=0
for file in glob.glob(path):
    inn = inn+1
    print(file)
    a=cv2.imread(file)
    print(a)

    cv2.imshow('Images1', a)
    k = cv2.waitKey(1000)


    scale_image(a,0.3,0.3,inn)
    scale_image(a,0.7,0.7,inn)
    scale_image(a,2,2,inn)
    scale_image(a,3,3,inn)

    translation_image(a,50,50,inn)
    translation_image(a,-50,50,inn)
    translation_image(a,40,-40,inn)
    translation_image(a,-40,-40,inn)

    rotate_image(a,90,inn)
    rotate_image(a,180,inn)
    rotate_image(a,270,inn)



    resize_image(a,450,400,inn)

    crop_image(a,100,400,0,350,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    crop_image(a,100,400,100,450,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    crop_image(a,0,300,0,350,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    crop_image(a,0,300,100,450,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    crop_image(a,100,300,100,350,inn)#(y1,y2,x1,x2)(bottom,top,left,right)

    padding_image(a,100,0,0,0,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    padding_image(a,0,100,0,0,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    padding_image(a,0,0,100,0,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    padding_image(a,0,0,0,100,inn)#(y1,y2,x1,x2)(bottom,top,left,right)
    padding_image(a,100,100,100,100,inn)#(y1,y2,x1,x2)(bottom,top,left,right)

    flip_image(a,0,inn)#horizontal
    flip_image(a,1,inn)#vertical
    flip_image(a,-1,inn)#both



    invert_image(a,255,inn)
    invert_image(a,200,inn)
    invert_image(a,150,inn)
    invert_image(a,100,inn)
    invert_image(a,50,inn)

    add_light(a,1.5,inn)
    add_light(a,2.0,inn)
    add_light(a,2.5,inn)
    add_light(a,3.0,inn)
    add_light(a,4.0,inn)
    add_light(a,5.0,inn)
    add_light(a,0.7,inn)
    add_light(a,0.4,inn)
    add_light(a,0.3,inn)
    add_light(a,0.1,inn)

    add_light_color(a,255,1.5,inn)
    add_light_color(a,200,2.0,inn)
    add_light_color(a,150,2.5,inn)
    add_light_color(a,100,3.0,inn)
    add_light_color(a,50,4.0,inn)
    add_light_color(a,255,0.7,inn)
    add_light_color(a,150,0.3,inn)
    add_light_color(a,100,0.1,inn)

    saturation_image(a,50,inn)
    saturation_image(a,100,inn)
    saturation_image(a,150,inn)
    saturation_image(a,200,inn)

    hue_image(a,50,inn)
    hue_image(a,100,inn)
    hue_image(a,150,inn)
    hue_image(a,200,inn)
    multiply_image(a,0.5,1,1,inn)
    multiply_image(a,1,0.5,1,inn)
    multiply_image(a,1,1,0.5,inn)
    multiply_image(a,0.5,0.5,0.5,inn)

    multiply_image(a,0.25,1,1,inn)
    multiply_image(a,1,0.25,1,inn)
    multiply_image(a,1,1,0.25,inn)
    multiply_image(a,0.25,0.25,0.25,inn)

    multiply_image(a,1.25,1,1,inn)
    multiply_image(a,1,1.25,1,inn)
    multiply_image(a,1,1,1.25,inn)
    multiply_image(a,1.25,1.25,1.25,inn)

    multiply_image(a,1.5,1,1,inn)
    multiply_image(a,1,1.5,1,inn)
    multiply_image(a,1,1,1.5,inn)
    multiply_image(a,1.5,1.5,1.5,inn)


    gausian_blur(a,0.25,inn)
    gausian_blur(a,0.50,inn)
    gausian_blur(a,1,inn)
    gausian_blur(a,2,inn)
    gausian_blur(a,4,inn)

    averageing_blur(a,5,inn)
    averageing_blur(a,4,inn)
    averageing_blur(a,6,inn)

    median_blur(a,3,inn)
    median_blur(a,5,inn)
    median_blur(a,7,inn)

    bileteralBlur(a,9,75,75,inn)
    bileteralBlur(a,12,100,100,inn)
    bileteralBlur(a,25,100,100,inn)
    bileteralBlur(a,40,75,75,inn)

    erosion_image(a,1,inn)
    erosion_image(a,3,inn)
    erosion_image(a,6,inn)

    dilation_image(a,1,inn)
    dilation_image(a,3,inn)
    dilation_image(a,5,inn)


    opening_image(a,1,inn)
    opening_image(a,3,inn)
    opening_image(a,5,inn)

    closing_image(a,1,inn)
    closing_image(a,3,inn)
    closing_image(a,5,inn)

    morphological_gradient_image(a,5,inn)
    morphological_gradient_image(a,10,inn)
    morphological_gradient_image(a,15,inn)

    top_hat_image(a,200,inn)
    top_hat_image(a,300,inn)
    top_hat_image(a,500,inn)

    black_hat_image(a,200,inn)
    black_hat_image(a,300,inn)
    black_hat_image(a,500,inn)


    sharpen_image(a,inn)
    emboss_image(a,inn)

    edge_image(a,3,inn)
    edge_image(a,5,inn)
    edge_image(a,9,inn)

    addeptive_gaussian_noise(a,inn)



    contrast_image(a,25,inn)
    contrast_image(a,50,inn)
    contrast_image(a,100,inn)

    edge_detect_canny_image(a,100,200,inn)
    edge_detect_canny_image(a,200,400,inn)

    grayscale_image(a,inn)
    transformation_image(a,inn)

















