import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_dat(dat, is_first=False):  
    lbls = ["Area", "Standard Deviation", "Mean"]
    colors = ["r", "g", "b"]
    
    mask = np.logical_and(dat[:,0] > 0.01, dat[:,1] > 0.46)
    mask = mask * (1 - 0.2*2) + 0.2
    
    for i in range(len(lbls)):
        plt.plot(dat[:,i], label=lbls[i] if is_first else "", c=colors[i], alpha=1.0 if is_first else 0.5)
    plt.plot(mask, label="Decision" if is_first else "", c="k", linestyle="dashed", alpha=1.0 if is_first else 0.5)

def xy(img, testing=False):
    blur = cv2.blur(img, ksize=(11, 11))
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, (9,9), iterations=3)
    
    opening[:,:70] = 255
    opening[-50:,:] = 255
    
    area = len(np.where(opening.flatten() == 0)[0])
    std = np.std(blur)
    mean = np.mean(blur)
    r = [img, blur, thresh1, opening], [area, std, mean]
    
    if testing:
        row = 430
        
        plt.plot(img[row,:], c="r", alpha=0.3, label="raw data")
        plt.plot(blur[row,:], c="b", alpha=1.0, label="smoothed")
        plt.plot(thresh1[row,:], c="k", alpha=0.5, label="rounded", linestyle="dashed")
        
        plt.legend()
        plt.show()
    
    return r