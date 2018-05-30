#This function runs through all the images in the image list and determines the 36 element vector for each image
#It saves everythign to a 'otsu_0.txt' file.
#The 'otsu_0.txt' file contains the filename, photoset, and 36 element information
import os, sys, math, random, json

f = open("log.txt")

num_done = -1
lf = open("otsu_0.txt", "wb")
for line in f:
    #load in the filename and photoset
  fn, pset = line.strip().split(",")
  fn = fn
  sfn = "images/" + fn
  num_done += 1
  print "Working", sfn, pset
  #run the 'cotsu.exe' program on each image, and save the result in 'junk_0.txt'
  #the result is a 36 element vector, which describes the background and forground contribution of each bin
  os.system("./cotsu %s > junk_0.txt" % (sfn))
  s = open("junk_0.txt").read()
  #save the output of the cotsu analysis to a dictionary 'q'
  qq = {}
  qq["file"] = fn
  qq["set"] = pset
  qq["otsu"] = s
  #Save the information in dictionary 'q' to the 'otsu_0.txt. file
  lf.write(json.dumps(qq) + "\n")
os.system("rm junk_0.txt")
