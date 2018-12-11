import fixmat as ft
import sys
import cv2
import numpy as np
import os

'''
Creating the ground-truth mapping for fixation locations using a Gaussian
filter:

    1) add values to pixel of fixation location and surrounding pixels to
       account for angle of error in fixation tracking

    2) Gaussian filter applied to fixation locations
'''

def main(path_to_data_file, path_to_imgs, path_to_output):
    #load data from eye-tracking file including labelme images
    eeg,meta_eeg = ft.load(path_to_data_file, "EEG")
    img_idx = range(1, len(os.listdir(path_to_imgs))+1)
    for i in img_idx:
	img = cv2.imread(path_to_imgs + str(i) + ".jpg")
	img_filter = np.zeros((img.shape[0],img.shape[1]))
	test_img_data = eeg.groupby("filenumber").get_group(i)
	test_img_data = test_img_data[test_img_data.y < img_filter.shape[0]]
	test_img_data = test_img_data[test_img_data.x < img_filter.shape[1]]
	test_img_data = test_img_data[test_img_data.x >= 0]
	test_img_data = test_img_data[test_img_data.y >= 0]
	
	m = range(img.shape[0])
	n = range(img.shape[1])
	for idx,row in test_img_data.iterrows():
	    
	    # add values to surrounding pixels according to gaussian
	    y = int(row["x"])
	    x = int(row["y"])
	    img_filter[x,y] += 4
	    top = x - 1
	    bottom = x + 1
	    left = y - 1
	    right = y + 1
	    if top in m:
		img_filter[top,y] += 2
		if left in n:
		    img_filter[top,left] += 1
		if right in n:
		    img_filter[top,right] += 1
	    if bottom in m:
		img_filter[bottom,y] += 2
		if left in n:
		    img_filter[bottom,left] += 1
		if right in n:
		    img_filter[bottom,right] += 1
	    if left in n:
		img_filter[x,left] += 2
	    if right in n:
		img_filter[x,right] += 2
	    
	    
	blurred = gaussian_filter(img_filter, sigma=5)
	
	# normalize
	blurred *= (255/np.amax(blurred))
	cv2.imwrite(path_to_output + str(i) + ".jpg", blurred)

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
