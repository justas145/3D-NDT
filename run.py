import xy, xz, yz

from os import walk
import numpy as np
import cv2
from functools import cmp_to_key
from timeit import default_timer as timer
import random

import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300

# What plane to look at
PLANE = ["xy", "xz", "yz"][2]

# MODES
# 0 = select specific images
# 1 = all images
# 2 = plot in order
# 3 = random 10 images
# 4 = write to file
# 5 = write to csv file
mode = 0

# Location of the OCT scan processed images
image_folder = "../Processed_Images/Processed_Images/"
filenames = next(walk(image_folder), (None, None, []))[2]

# Where to save figs if on mode 1 - especially useful for general inspection
out_folder = "out/"
should_save = False

plot_more = True # Should the analysis plot more info?

def get_num(f):
    n = f[13:-4]
    n1 = f[4:][0]
    return int(n) + 500 * int(n1)

def compare(item1, item2):
    return get_num(item1) - get_num(item2)

filenames = sorted(filenames, key=cmp_to_key(compare))

def plot(imgs, title=None, titles=None):
    f, axs = plt.subplots((len(imgs) - 1) // 3 + 1, 3)
    axs = axs.ravel()
    #for ax in axs:
    #    ax.get_xaxis().set_visible(False)
    #    ax.get_yaxis().set_visible(False)
    for (idx, img) in enumerate(imgs):
        ax = axs[idx]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        
        if titles is not None:
            ax.set_title(titles[idx])
    if title is not None:
        plt.suptitle(title)
    plt.plot()

def normalize_rows(d):
    res = np.ones(d.shape)
    for (idx, row) in enumerate(d):
        res[idx,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
    return res

def process(img, testing=False):
    if PLANE == "xy":
        return xy.xy(img, testing=testing)
    elif PLANE == "xz":
        return xz.xz(img, testing=testing)
    elif PLANE == "yz":
        return yz.yz(img, testing=testing)
    raise NotImplementedError

def plot_dat(dat, is_first=False):
    if PLANE == "xy":
        return xy.plot_dat(dat, is_first=is_first)
    elif PLANE == "xz":
        return xz.plot_dat(dat, is_first=is_first)
    elif PLANE == "yz":
        return yz.plot_dat(dat, is_first=is_first)
    raise NotImplementedError

imgs = {}
for f in filenames:
    if PLANE.upper() in f:
        imgs[f] = cv2.imread(image_folder + f, cv2.IMREAD_GRAYSCALE)
    

if mode == 0:
    cube = "1"
    #n = ["470", "471", "472", "473", "474", "475", "476", "477", "478", "479", "480", "481"]
    #n = ["49", "50", "51", "52", "53"]
    #n = ["470", "471", "472", "473", "474", "475", "476", "477", "478", "479", "480", "481", "390", "392"]
    n = ["472"]
    
    times = []
    for k in imgs.keys():
        if cube == k[4] and k[13:-4] in n:
            start = timer()
            imgs_lst, score = process(imgs[k], testing=plot_more)
            times.append((timer() - start))
            
            s = f"{np.round(score, 1)}"
            plot(imgs_lst, title=f"{k} | score = {s}")
            plt.show()
            plt.close()
    
    times = np.array(times)*1000
    print(f"Took {round(times.sum(),1)}ms, mean of {round(times.mean(),1)}ms")
elif mode == 1:    
    if should_save:
        import shutil
        shutil.rmtree(out_folder)
        import os
        os.makedirs(out_folder)
    
    times = []
    for k in imgs.keys():
        start = timer()
        imgs_lst, score = process(imgs[k])
        times.append((timer() - start))
        
        s = f"{np.round(score, 1)}"
        plot(imgs_lst, title=f"{k} | score = {s}")
        
        if should_save:
            plt.savefig(out_folder + k)
        
        plt.show()
        plt.close()
    times = np.array(times)*1000
    t = int(round(sum(times), 0))
    m = round(times.mean(), 1)
    print(f"Took {t}ms. Mean: {m}ms (n={len(times)})")
elif mode == 2:
    scores1 = []
    scores2 = []
    times = []
    for k in imgs.keys():  
        start = timer()
        imgs_lst, score = process(imgs[k])
        times.append((timer() - start))
        
        if k[4] == "1":
            scores1.append(score)
        elif k[4] == "2":
            scores2.append(score)
        else:
            raise NotImplementedError
    times = np.array(times)*1000
    t = int(round(sum(times), 0))
    m = round(times.mean(), 3)
    print(f"Took {t}ms. Mean: {m}ms (n={len(times)}).")
    scores1 = normalize_rows(np.array(scores1).T).T
    scores2 = normalize_rows(np.array(scores2).T).T
    
    plot_dat(scores1, is_first=True)
    plot_dat(scores2)
    
    plt.ylabel("Normalized Score")
    plt.xlabel("Image position (sorted)")
    plt.title(f"{PLANE.upper()} Plane Scores Overview")
    
    plt.ylim(0, 1)
    
    plt.legend()
    plt.show()
elif mode == 3:
    keys = list(imgs.keys())
    random.shuffle(keys)
    times = []
    
    n = 10
    
    for (idx, k) in enumerate(keys):
        start = timer()
        imgs_lst, score = process(imgs[k], testing=plot_more)
        times.append((timer() - start))
        
        s = f"{np.round(score, 1)}"
        plot(imgs_lst, title=f"{k} | score = {s}")
        plt.show()
        plt.close()
        
        if idx >= n:
            break
    times = np.array(times)*1000
    t = int(round(sum(times), 0))
    m = round(times.mean(), 3)
    print(f"Took {t}ms. Mean: {m}ms (n={len(times)}).")
elif mode == 4:
    keys = list(imgs.keys())
    
    with open(f"{PLANE}_labels.txt", "w") as f:
        for k in keys:
            f.write(f"{k}: 0\n")
elif mode == 5:
    known_errors = ["2_49", "2_50", "2_51", "2_52", "2_53"]
    times = []
    
    with open(f"{PLANE}_scores.csv", "w") as f:
        for k in imgs:
            start = timer()
            imgs_lst, score = process(imgs[k], testing=False)
            times.append((timer() - start))
            
            ke = f"{k[4]}_{k[13:-4]}"
            val = 2 if ke in known_errors else 0
            
            if len(times) % 100 == 0:
                print(len(times), len(imgs))
            
            s = ",".join([str(i) for i in score])
            f.write(f"{k},{val},{s}\n")
    times = np.array(times)*1000
    t = int(round(sum(times), 0))
    m = round(times.mean(), 3)
    print(f"Took {t}ms. Mean: {m}ms (n={len(times)}).")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        