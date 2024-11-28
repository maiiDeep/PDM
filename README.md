# PDM
Polynomial Distance between two lines
import os
import cv2
import time
import numpy as np
import math
import pandas as pd
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage import measure
def PDM_Calculate (BorderOutput1, BorderOutput2, ErrorDir, old, new):
   
    filenames = os.listdir(BorderOutput1)
    for filename in filenames:
        print (filename)
        Segment_1 = np.genfromtxt(os.path.join (BorderOutput1,filename), dtype=None).astype(int)
        Segment_2 = np.genfromtxt(os.path.join (BorderOutput2,filename.replace(old, new)), dtype=None).astype(int)
        #print (Segment_1)
        #print (Segment_2)
        mylist = []
        item =  poly_dist_method(Segment_1, Segment_2)
        mylist.append(item)
        #print (item)
    
        mat = np.array(mylist)
        #print (mat)
        np.savetxt(os.path.join(ErrorDir, filename.replace(old,new)), mat, fmt='%f')
        #break
def poly_dist_method(B1, B2):
    B1 = np.transpose(B1)
    B2 = np.transpose(B2)
    
    S1 = B1.shape
    S2 = B2.shape
    
    DvB1SB2 = np.zeros(S1[1])
    DvB2SB1 = np.zeros(S1[1])
    
    # Compute distance of B1 vertices from B2 segments (B2 is GT)
    for j in range(S1[1]):
        Dvs = np.zeros(S2[1] - 1)
        #Dvs = []
        for k in range(S2[1] - 1):
            
            denominator = ((B2[0, k+1] - B2[0, k])**2 + (B2[1, k+1] - B2[1, k])**2)
            if denominator==0:
                Dvs[k] = np.nanmax(Dvs)
                #continue
            else:
                numerator = abs(((B2[1, k+1] - B2[1, k]) * (B1[1, j] - B2[1, k]) + (B2[0, k+1] - B2[0, k]) * (B1[0, j] - B2[0, k])))
                Lambda = numerator / denominator if denominator != 0 else 0
            
                if 0 <= Lambda <= 1: 
                    Dvs[k] = (abs(((B2[1, k+1] - B2[1, k]) * (-B1[0, j] + B2[0, k]) + (B2[0, k+1] - B2[0, k]) * (B1[1, j] - B2[1, k])) / np.sqrt(denominator)))
                else:
                    d1 = np.sqrt((B1[0, j] - B2[0, k])**2 + (B1[1, j] - B2[1, k])**2)
                    d2 = np.sqrt((B1[0, j] - B2[0, k+1])**2 + (B1[1, j] - B2[1, k+1])**2)
                    Dvs[k] = min(d1, d2)
        
        DvB1SB2[j] = min(Dvs)
    
    # Swap the profiles and compute the distance between B2 vertices and B1 segments (B2 is GT)
    B1, B2 = B2, B1
    S1 = B1.shape
    S2 = B2.shape
    
    for j in range(S1[1]):
        #Dvs = []
        Dvs = np.zeros(S2[1] - 1)
        for k in range(S2[1] - 1):
            
            denominator = ((B2[0, k+1] - B2[0, k])**2 + (B2[1, k+1] - B2[1, k])**2)
            if denominator==0:
                Dvs[k] = np.nanmax(Dvs)
                #continue
            else:
                numerator = abs(((B2[1, k+1] - B2[1, k]) * (B1[1, j] - B2[1, k]) + (B2[0, k+1] - B2[0, k]) * (B1[0, j] - B2[0, k])))
                Lambda = numerator / denominator if denominator != 0 else 0
           
                if 0 <= Lambda <= 1: 
                    Dvs[k] = (abs(((B2[1, k+1] - B2[1, k]) * (-B1[0, j] + B2[0, k]) + (B2[0, k+1] - B2[0, k]) * (B1[1, j] - B2[1, k])) / np.sqrt(denominator)))
                else:
                    d1 = np.sqrt((B1[0, j] - B2[0, k])**2 + (B1[1, j] - B2[1, k])**2)
                    d2 = np.sqrt((B1[0, j] - B2[0, k+1])**2 + (B1[1, j] - B2[1, k+1])**2)
                    Dvs[k] = min(d1, d2) 
        
        DvB2SB1[j] = min(Dvs)
    
    PDM = (np.sum(np.abs(DvB1SB2)) + np.sum(np.abs(DvB2SB1))) / (S1[1] + S2[1])
    
    return PDM
