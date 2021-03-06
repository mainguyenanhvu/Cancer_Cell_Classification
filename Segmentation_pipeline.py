#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.patches as mpatches
from Segmentation_pipeline_helper import find, watershed_lab, watershed_lab2, resize_pad, find_border, pixel_norm, shift_center_mass, regionprops

##### EXECUTION PIPELINE FOR CELL SEGMENTATION

# Define path to image input directory
imageinput ='training_upload'

# Define desired path to save the segmented images
imageoutput = "training_seg_3"

if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)
    
# Make a list of path to the tif.gz files
nuclei = find(imageinput,prefix=None, suffix="blue.tif",recursive=False) #blue chanel =nu
nuclei = nuclei[0:2]
nucleoli = []
microtubule = []
for f in nuclei:
    f=f.replace('blue','yellow')
    nucleoli.append(f)
    f=f.replace('yellow','red')
    microtubule.append(f)

# For each image, import 3 chanels
# Use nuclei as seed, microtubule as edges to segment the image
# Cut the bounding box of each cell (3channels) in the respective image, slack and save
for index, imgpath in enumerate(nuclei):

    print("stacking image {0}/{1}".format(index, len(nucleoli)))
    # Unzip .gz file and read content image to img
    try:
        nu = plt.imread(imgpath)
        if len(nu.shape) > 2:
            nu = nu[:, :, 2]
    except:
        print("%s does not have valid nucleus channel" % (nuclei[index].split("_blue.tif"))[0])
        continue

    try:
        org = plt.imread(nucleoli[index])
        if len(org.shape) > 2:
            org = org[:, :, 1]
    except:
        print("%s does not have valid nucleoli channel" % (nuclei[index].split("_blue.tif"))[0])
        continue

    try:
        mi = plt.imread(microtubule[index])
        if len(mi.shape) > 2:
            mi = mi[:, :, 0]
    except:
        print("%s does not have valid microtubule channel" % (nuclei[index].split("_blue.tif"))[0])
        continue

    # obtain nuclei seed for watershed segmentation
    seed, num = watershed_lab(nu, rm_border=False)
    
    # segment microtubule image
    marker = np.full_like(seed, 0)
    marker[mi == 0] = 1 #background
#    marker = skimage.morphology.binary_erosion(marker,skimage.morphology.square(3)).astype(int)
    marker[seed > 0] = seed[seed > 0] + 1 #foreground
    labels = watershed_lab2(mi, marker = marker)
    
    #remove all cells where nucleus is touching the border
    labels = labels - 1
    border_indice = find_border(seed)
    mask = np.in1d(labels,border_indice).reshape(labels.shape)
    labels[mask] = 0

    """
    #Plotthing the segmentation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mi)
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area >= 20000:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    """

    # Cut boundingbox
    i=0
    for region in skimage.measure.regionprops(labels):
        i=i+1
        
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region.bbox
                
        # get mask
        mask = labels[minr:maxr,minc:maxc].astype(np.uint8)
        mask[mask != region.label] = 0
        mask[mask == region.label] = 1

        cell_nuclei = pixel_norm(nu[minr:maxr,minc:maxc]*mask)
        cell_nucleoli = pixel_norm(org[minr:maxr,minc:maxc]*mask)
        cell_microtubule = pixel_norm(mi[minr:maxr,minc:maxc]*mask)

        # stack channels
        cell = np.dstack((cell_microtubule,cell_nucleoli,cell_nuclei))
        cell = (cell*255).astype(np.uint8) #the input file was uint16         

        # align cell to the 1st major axis  
        theta=region.orientation*180/np.pi #radiant to degree conversion
        cell = ndi.rotate(cell, 90-theta)

        # resize images
        fig2 = resize_pad(cell) # default size is 256x256
        # center to the center of mass of the nucleus
        fig = shift_center_mass(fig)
        fig = Image.fromarray(fig)
        name = "%s_cell%s.%s" % ((nuclei[index].split("_blue.tif"))[0], str(i), "png")
        name = name.replace("training_upload/", "")
        
        savepath= os.path.join(imageoutput, name)
        #plt.savefig(savepath)
        fig.save(savepath)

