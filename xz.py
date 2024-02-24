import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences

def scale(d, max_val=255, convert_int=True):
    ptp = np.ptp(d)
    if ptp > 0.1:
        a = ((d - np.min(d)) / ptp) * max_val
    else:
        a = np.zeros(d.shape)
    if convert_int:
        return a.astype("uint8")
    else:
        return a

def scale_axis(x, axis=0):
    ptp = x.ptp(axis=axis, keepdims=True)
    return  ((x - x.min(axis=axis, keepdims=True)) / ptp * 255).astype("uint8")

def plot_dat(dat, is_first=False):
    return

def xz(img, testing=False):
    m = np.mean(img)
    
    m0 = np.median(img, axis=0)
    m1 = np.repeat(m0, repeats=img.shape[0], axis=0).reshape(*img.shape, order="F")
    
    d = scale(img - m1)
    
    blur = cv2.bilateralFilter(d,5,50,20)
    blur2 = cv2.medianBlur(blur, ksize=11)
    blur3 = cv2.GaussianBlur(blur2, (1,1), 0)
    
    sobelx = scale((cv2.Sobel(blur3,cv2.CV_64F,1,0,ksize=31)))
    
    soblex_crop = sobelx[:,:160]
    otsu_threshold, otsu = cv2.threshold(soblex_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.median(otsu) > 120:
        otsu = 255 - otsu
    
    nz = np.nonzero(otsu)
    
    try:
        xmin,xmax = min(nz[1]), max(nz[1])
    except ValueError:
        return [img, m1, d, blur, blur2, blur3, sobelx, otsu], [0.0, 0.0, m]
    
    top_profile = otsu[:, xmin:xmax]
    vals = []
    
    # Used to remove artifacts from analysis
    artifacts = np.copy(sobelx)
    artifacts[:,:160] = 0
    artifacts = scale_axis(artifacts, axis=1)
    artifacts[-20:,:] = 0
    
    arts = scale(-np.mean(artifacts, axis=1))
    arts_peaks, _ = find_peaks(arts, prominence=75) # used to be 50
    widths, width_heights, ips1, ips2 = peak_widths(arts, arts_peaks)
    
    sobelx_copy = np.copy(sobelx)
    for (p, w) in zip(arts_peaks, widths):
        w *= 2
        sobelx_copy[int(p-w/2):int(p+w/2),:] = 0
    sobelx_copy[:20,:] = 0
    # Idea: if the width is too large it means that there
    # are a lot of artifacts, hence the image is unprocessable
    # ####
    
    for i in range(sobelx.shape[1] - top_profile.shape[1] + 1): # subtract top profile
        selection = sobelx_copy[:,i:i+top_profile.shape[1]]
        v = np.multiply(top_profile, selection)
        
        vals.append(np.sum(v))
    vals = np.array(vals)
    
    x = scale(vals, max_val=1, convert_int=False)
    peaks, _ = find_peaks(x, prominence=0.04)
    peaks = peaks[peaks > 150]
    prominences = peak_prominences(x, peaks)[0]

    for p in peaks:
        sobelx_copy[:,p-3:p+3] = 0

    if testing:        
        plt.plot(arts_peaks, arts[arts_peaks], "x")
        plt.plot(arts)
        plt.show()
        
        plt.plot(peaks, x[peaks], "x")
        plt.plot(x)
        plt.ylim(0, 1)
        plt.show()
    
    # better color 83,255,115
    return [img, m1, d, blur, blur3, sobelx, artifacts, sobelx_copy, top_profile], [prominences.sum()*100, widths.sum(), m]














