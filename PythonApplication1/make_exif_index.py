#Function makes a 'log.txt' file that contains the image name and photoset.
#It doesn't include pictures that don't meet various requirements

import os, sys, time, math, random, json

#images = [x.strip() for x in open("images/image_list.txt")]
images = [x.strip() for x in open(sys.argv[1])]

lf = open("log.txt","wb")
num_done = -1
t0 = -1
#Loop through all imges in the input text file
for img_fn in images:
  num_done += 1 
  #Skip the image if it meets some conditions
  if img_fn.find("(") >= 0: continue
  if img_fn.find(" ") >= 0: continue
  if img_fn.find("egacy") > -1: continue

  #Save the EXIF info to the 'junk.txt' file
  bad = os.system("exiftool images/%s > junk.txt" % img_fn)

  #M will contain the EXIF info for each file.
  M = {}
  M["file"] = img_fn
  #Iterate through the lines of 'junk.txt' and save each element as a key:value pair in M
  for line in open("junk.txt"):
    pos = line.find(":")
    L,R = line[:pos], line[pos+1:]
    #strip removes whitespace
    M[L.strip()] = R.strip()
  #If you don't find UserComment in the EXIF input, skip the file
  if "User Comment" not in M: continue

  # parse the user comment, saving to dictionary 'd'
  uc = M["User Comment"]
  pieces = uc.split(",")
  d  = {}
  for p in pieces:
    pp = p.split(":")
    if len(pp) != 2: continue
    L,R = p.split(":")
    d[L] = R.strip()

  # must be flash.  If it isn't flash, skip the file
  if d["Count"] != "30": continue

  #Save PhotoBunch and PhotoSet and filename info into their own dictionary
  q = {}
  pb = d["PhotoBunch"]
  ps = d["PhotoSet"]
  q["set"] = ps
  q["file"] = M["file"]
 
  #Write 'q' info to the log.txt file
  lf.write("%s,%s\n" % (q["file"],ps))
  lf.flush()

  #From time to time, report the progress
  if time.time() > t0 + 10.0:
    print "%8.2f%% " % (100*num_done/float(len(images)))
    t0 = time.time()
#Delete the 'junk.txt' file
os.system("rm junk.txt")

