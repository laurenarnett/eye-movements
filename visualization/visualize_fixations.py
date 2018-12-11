import fixmat as ft
import sys
import os
import cv2


def main(path_to_data_file, path_to_imgs, path_to_output):
    #load data from eye-tracking file including labelme images
    eeg,meta_eeg = ft.load(path_to_data_file, "EEG")

    img_idx = range(1, len(os.listdir(path_to_imgs))+1)

    # visualizing fixation locations by drawing circles on the images
    for i in img_idx:
        test_img_data = eeg.groupby("filenumber").get_group(i)
        img = cv2.imread(path_to_imgs + str(i) + ".jpg")
        for idx,row in test_img_data.iterrows():
            cv2.circle(img, (int(row["x"]),int(row["y"])),7,color=(0,255,0))

        cv2.imwrite(path_to_output + str(i) + ".jpg", img)

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
