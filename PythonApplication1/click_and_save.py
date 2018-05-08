# import the necessary packages
import argparse
import cv2
import os
import json
from glob import glob
 

def click_and_crop(event, x, y, flags, param):
# grab references to the global variables
    global refPt, cropping
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    point_list=[]
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])
        print "("+str(x)+","+str(y)+"), ",
        #cropping = True
 
    # check to see if the left mouse button was released
    #elif event == cv2.EVENT_LBUTTONUP:
    #    # record the ending (x, y) coordinates and indicate that
    #    # the cropping operation is finished
    #    refPt.append((x, y))
    #    cropping = False
 
    #    # draw a rectangle around the region of interest
    #    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
    #    cv2.imshow("image", image)

#make list of files to process

files = []
start_dir=raw_input("Image Directory: ")

pattern = "*.jpg"
        
for dir,_,_ in os.walk(start_dir):
            
    files.extend(glob(os.path.join(dir,pattern))) 

test_name=raw_input("Input test name: ")
coordinate_map={}

print "You will be analyzing "+str(len(files))+" images"
print "To save the candidate points, press 's'"
print "To reset the candidate points, press 'r'"



for index, image_path in enumerate(files):
    print ""
    print "The test is "+str(float(index)/float(len(files)))+"% Complete."

    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
    refPt = []
    cropping = False
 
   
    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required=True, help="Path to the image")
    #args = vars(ap.parse_args())
 
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(image_path)
    clone = image.copy()
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)

    print "The following points will be saved: ",

    cv2.setMouseCallback("image", click_and_crop)

 
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            print ""
            print "Candidate Points have been reset!"
            print "The following points will be saved: ",
            refPt=[]
 
        # if the 's' key is pressed, break from the loop
        elif key == ord("s"):
            break
    
 
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt)>=1:
        #roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        coordinate_map[image_path]=refPt
        print ""
        print "Process complete."
        print "File: "+image_path
        print "Points: ",
        print refPt
        refPt=[]
        #cv2.imshow("ROI", roi)
        #cv2.waitKey(0)
 
    # close all open windows
    cv2.destroyAllWindows()
print coordinate_map
with open(test_name+".json", "w") as write_file:
    json.dump(coordinate_map, write_file)