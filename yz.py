import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences

def scale(d, max_val=255, convert_int=True):
    a = ((d - np.min(d)) / (np.max(d) - np.min(d)) * max_val)
    if convert_int:
        return a.astype("uint8")
    else:
        return a

def scale_axis(x):
    return  ((x - x.min(0)) / x.ptp(0) * 255).astype("uint8")

def plot_dat(dat, is_first=False):
    return

def yz(img, testing=False):
    m0 = np.mean(img, axis=1)
    m1 = np.repeat(m0, repeats=img.shape[1], axis=0).reshape(*img.shape)
    
    d = scale(img - m1)
    
    blur = cv2.bilateralFilter(d,5,50,20)
    blur2 = cv2.medianBlur(blur, ksize=11)
    blur3 = cv2.GaussianBlur(blur2, (1,1), 0)
    
    sobely = scale((cv2.Sobel(blur3,cv2.CV_64F,0,1,ksize=31)))
    
    sobley_crop = sobely[:160,:]
    otsu_threshold, otsu = cv2.threshold(sobley_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    nz = np.nonzero(otsu)
    ymin,ymax = min(nz[0]), max(nz[0])
    
    top_profile = otsu[ymin:ymax, :]
    vals = []
    
    # Used to remove artifacts from analysis
    artifacts = np.copy(sobely)
    artifacts[:150,:] = 0
    artifacts = scale_axis(artifacts)
    artifacts[:,:140] = 0
    
    arts = scale(-np.mean(artifacts, axis=0))
    arts_peaks, _ = find_peaks(arts, prominence=40) # used to be 50
    widths, width_heights, ips1, ips2 = peak_widths(arts, arts_peaks)
    
    sobely_copy = np.copy(sobely)
    for (p, w) in zip(arts_peaks, widths):
        w *= 2
        sobely_copy[:,int(p-w/2):int(p+w/2)] = 0
    # Idea: if the width is too large it means that there
    # are a lot of artifacts, hence the image is unprocessable
    # ####
    
    for i in range(sobely.shape[0] - top_profile.shape[0] + 1): # subtract top profile
        selection = sobely_copy[i:i+top_profile.shape[0],:]
        v = np.multiply(top_profile, selection)
        
        vals.append(np.sum(v))
    vals = np.array(vals)
    
    x = scale(vals, max_val=1, convert_int=False)
    peaks, _ = find_peaks(x, prominence=0.06)
    peaks = peaks[peaks > 150]
    prominences = peak_prominences(x, peaks)[0]

    for p in peaks:
        sobely_copy[p-3:p+3,:] = 0

    if testing:        
        plt.plot(arts_peaks, arts[arts_peaks], "x", ms=20)
        plt.plot(arts)
        plt.xlabel("Horizontal Position (pixels)")
        plt.ylabel("Average Column Value")
        plt.show()
        
        plt.plot(peaks, x[peaks], "x", ms=20)
        plt.plot(x)
        plt.xlabel("Vertical Position (pixels)")
        plt.ylabel("Normalized Convolution Value")
        plt.ylim(0, 1)
        plt.show()
        
        plt.subplot(3, 1, 1)
        plt.imshow(sobley_crop, cmap="gray")
        plt.subplot(3, 1, 2)
        plt.imshow(otsu, cmap="gray")
        plt.subplot(3, 1, 3)
        plt.imshow(top_profile, cmap="gray")
        plt.show()
    
    return [img, m1, d, blur, blur2, blur3, sobely, artifacts, sobely_copy], [prominences.sum()*100, widths.sum(), np.mean(img)]














