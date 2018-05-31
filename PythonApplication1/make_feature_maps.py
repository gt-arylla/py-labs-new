#This function takes in photo information and returns a 'transcript' file, which describest the bin, index, and ratio information for each file
#The transcript is used to determine the best


import math, os, sys, random, json

fns = []
for i in [0]: #,1,2]
    #combine together all the otsu_#.txt filenames into a list
  fns.append ( "otsu_%i.txt" % i )

# ==== cofigurations ==========================================================
#This matches photo information and photoset information to an index

v1_logo = ("Mi.50_v1app/", ["103"], ["101"])
v1_text = ("Mi.50_v1app/", ["104"], ["102"])

v2_Mi50_logo = ("Mi.50/", ["103"], ["101"])
v2_Mi50_text = ("Mi.50/", ["104"], ["102"])

v2_Nii40_logo = ("Nii.40/", ["109"], ["101"])
v2_Nii40_text = ("Nii.40/", ["110"], ["102"])

gt_test5=("v2app/Nii.40/",["110"],["102"])
vt_test=("C",["1"],["0"])

# ==== choose a configuration =================================================
#This determines what is actually used int he analysis.  The config file defines:
#prefix (part of the filename)
#posclass (photoset that identifies as a positive result)
#negclass (photoset that identifies as a negative result)

#config = v1_logo # 69, 70, 87
#config = v1_text # 57, 52, 73

#config = v2_Nii40_logo # 67, 73, 81, 86
#config = v2_Nii40_text # 65, 84, 92, 96 

config = vt_test # 83, 84, 89
#config = v2_Mi50_text # 75, 82, 82, 89 

# ============================================================================

#split the config tuple into three discrete variables
prefix, posclass, negclass = config

counts  = {}

def ovec (o):
    #This function pulls out the bin data and returns a 36 element vector, where the first 18 elements refer to the background, and the second 18 elements refer to the foreground

    #remove whitespace and split up input by newline
  o = o.strip().split("\n")
  buff = []
  buff2 = []
  for x in o:
      #split up element by space
      #each element has the format [channel name] [background value] [foreground value]
      #for example: AZ 0.0021 0.0003
    xs = x.split()
    #yL is the background value, yR is the foreground value
    yL = float(xs[1])
    yR = float(xs[2])
    #save the numbers you pulled out to temporary buffer lists
    buff.append(yL)
    buff2.append(yR)
#concatenate together the two lists, so the output is [back0, back1, back2, ... , fore0, fore1, fore2, ...]
  return buff + buff2

def vectorize(J):
    #This function pulls out the bin data from the otsu file
  oyflash = ovec(J["otsu"])
  return oyflash 

# == static seed for reproducible test/train split ===========================
random.seed(200)
pos = []
neg = []
buff = []
num_filtered = 0
min_contrast = 0

#iterate through all the 'otsu_#.txt' files in the list you defined int the beginning
for fn in fns:
    #iterate through each line in the file
  for line in open(fn):
      #load the information as a json object
    try:
      J = json.loads(line)
    except:
      continue
    #print J
    #define the filename.  If it does not contain the necessary prefix, defined in the config file, skip that file
    f = J["file"]
    if not f.startswith(prefix): continue
    #define the photoset.  
    k = J["set"]
    #keep track of how many of each photoset have been loaded in
    if k not in counts: counts[k] = 0
    counts[k] += 1
    #pull out the bin data from the otsu file
    J["vec"] = vectorize(J)

    #the final element in the vector is the contrast between the foreground and the background luminance
    #it is defined in the 'analyze' code as `contrast = (best_mR - best_mL) / 1000.0;`
    #best_mR is the average luminance of the foreground (light color) and best_mL is the average luminance of the background (dark color)
    #if the contrast is low, it means that the otsu threshold did a bad job separating out the foreground and background, or that the image is underexposed
    if J["vec"][-1] < min_contrast: continue
    #Save the J dictionary to a buffer list
    buff.append ( J )
    #Save the J dictionary to either the negativeclass or the positive class list
    if k in negclass: neg.append(J)
    elif k in posclass: pos.append(J)
    #print negclass
    #print posclass

def split (x):
    #assign a random number between 0 and 1 to each element in the input list
    #it'll look like [(random #, element0),(random #, element1)...]
  rx = [(random.uniform(0,1), xv) for xv in x]
  #sort by the random number
  rx.sort()
  #set the split point so that 2/3rds will be train data, 1/3rd will be test data
  K = (len(rx)*2)/3
  return [xv for r,xv in rx[:K]], [xv for r,xv in rx[K:]]

#split up each class list into train and test data using the split function
pos_train, pos_test = split(pos)
neg_train, neg_test = split(neg)

#each of these are lists of dictionaries

# == feature statistics ======================================================

def mstd(vns, p_mu = 0.0, p_std = 1.0, p_n = 0):
    #calculate the standard deviation in the input data
    #can accept either a list of numbers or a list of tuples
    #if it is a list of tuples, it is formatted as (value, # of occurances of said value)
    #Therefore the list of tuples represents a histogram
  totv = 0 + p_mu*p_n
  totv2 = 0 + (p_std*p_std + p_mu*p_mu)*p_n
  totn = 0 + p_n
  if isinstance(vns[0], tuple):
    for v,n in vns:
      totv += v*n
      totv2 += v*v*n
      totn += n
  else:
    for v in vns:
      totv += v
      totv2 += v*v
      totn += 1.0
    #calculate the average and variance of the input dataset
  mu = totv/float(totn)
  va = totv2/float(totn) - mu*mu
  #standard deviate is returned instead of variance
  return mu, va**0.5

def make_buckets (vals):
    #make buckets that have width that's 1/4er the size of the standard deviation
    #starting bin is 30*1/4=7.5 std deviations less than the mean, end bin is 7.5 std deviations greater than the mean

    #calculate the average and std dev of the input dataset
  m,s = mstd(vals)
  #calculate the bin width
  D = s*0.25
  #find all the bins.  You're making a list of tuples
  buff = [ (m-D/2.0 +(i-30)*D,m+D/2.0+(i-30)*D) for i in range(61) ]
  return buff

def bucketize (buck, v):
    #returns the bucket index that v is in
    #buck - list of buckets as a list of tuples
    #v - number

  i = len(buck)/2
  #L is the left side of the middle bucket, R is the right side
  L,R = buck[i]
  #we are looking for the bucket that the value sits in
  if L <= v <= R: return i
  #if the current candidate bucket is too big, D is negative.  Otherwise, D is positive
  D = -1 if v < L else 1
  #iterate through all the buckets, starting at the middle bucket, until you find the correct budket
  while 0 < i < len(buck)-1:
    i += D
    L,R = buck[i]
    if L <= v <= R: return i
  return i

# == map all the relative densities ==========================================

scobuff = []
maps = {}
out_map = []
for K0 in range(36):
    #pull out a list of numbers from the specified index K0
    #The input to the make_buckets funtion will look like (assuming index = z) [pos0, pos1, pos2, ..., pos_n, neg1, neg2, neg3,...,neg_n]
    #the output will be a list of tuples, which represent bins
  buck0 = make_buckets( [x["vec"][K0] for x in pos_train+neg_train] )
  for K1 in range(K0+1,36):
      #same as buck0, but you're using a different index
    buck1 = make_buckets( [x["vec"][K1] for x in pos_train+neg_train] )
    #set up matrix holders that will keep track of the number of counts for each bin
    #the matrix holder is 61 by 61
    #the first index indicates the bin for K0
    #the second index indicates the bin for K1
    NB0 = len(buck0)
    pos_counts = [[0 for i in range(NB0)] for j in range(NB0)]
    neg_counts = [[0 for i in range(NB0)] for j in range(NB0)]
 
    #co: count list
    #x: list of positive data or negative data
    #this loop only iterates over two things: (pos_counts, pos_train) and (neg_counts, neg_train)
    for co,x in [ (pos_counts, pos_train), (neg_counts, neg_train) ]:
        #iterate through image in list of positive or negative data
      for v in x:
        vec = v["vec"]
        #find the bucket index that v is in for the K0 and K1 values
        b1 = bucketize(buck0,vec[K0])
        b2 = bucketize(buck1,vec[K1])
        #add a count to the approriate location in either the positive_counts or negative_counts list
        co[b1][b2] += 1
    #create a text file that will store the counts information
    #the first line is the number of rows and columns in the counts matrix
    f = open("map.txt","w")
    f.write("%i %i\n" % (len(pos_counts[0]), len(pos_counts)))
    for i in range(len(pos_counts)):
      for j in range(len(pos_counts[0])):
        n1,n2 = (pos_counts[i][j], neg_counts[i][j])
        if max(n1,n2) == 0: continue
        #save cases where there is information for both the positive and the negative case
        #i is K0 information
        #j is K1 information
        #n1 is the number of positive elements that appeared in that bin
        #n2 is the number of negative elements that appeared in that bin
        f.write("%i %i %i %i\n" % (i,j,n1,n2))
    f.close() 
    #map_feature calculates the smoothed ln ratio of each bin.  If the ratio is positive it more describes a positive case.  If it is negative it more describes a negative case.
    #It finishes by outputting 'mapdata.txt', which is a 61 x 61 matrix of normalized ratio values
    os.system("python map_feature.py map.txt")
    #open the output of map_feature
    J = json.loads(open("mapdata.txt").read())
    #save the last row of the matrix with its accompanying K0 and K1 indices
    scobuff.append( (J[-1], K0, K1) )
    scobuff.sort()
    #save the indices, the buckets, and the ratio grid to a dictionary
    maps[(K0,K1)] = (buck0, buck1, J)
    #save the indices, the buckets, and the ratio grid to a list
    out_map.append ( [K0,K1,buck0,buck1,J]  )
    print '\r ... %4i %4i        '  % (K0,K1),
  print '\r = ', K0, '                     ',

# == feature transcript ======================================================

f = open("transcript.txt","wb")
#the zip makes a list of tupes.  In this case, the list will be: [(pos_train,0),(neg_train,1),(pos_test,2),(neg_test,3)]
for x,s in zip( [pos_train, neg_train, pos_test, neg_test], [0,1,2,3]):
    #pc is the element index.  p is the element
    #x is the data
    #s is the data index
  for pc,p in enumerate(x):
      #v is the 36 element vector.  fn is the filename
    v = p["vec"]
    fn = p["file"]
    #keyn is the number of times data has been writted for this image
    keyn = 0
    buff = []
    #iterate through all combinations of the 36 indices
    for K0 in range(36):
      for K1 in range(K0+1,36):
          #grab the bucket info and ratio grid info for the particular index pair
        buck0, buck1, J = maps[(K0,K1)]
        #find what bucket the input data is in
        b0 = bucketize(buck0,v[K0])
        b1 = bucketize(buck1,v[K1])
        #find the score for that input data
        sco = "%1.4f" % J[b0][b1]
        key = "%s:%s:%s:%s:%s:" % (K0,b0,K1,b1,keyn)
        buff.append(key + str(sco))
        keyn += 1
    #the data is saved to the transcript file.  It is in the format:
    #[filename] [DATA]
    #The DATA is in the format [index0]:[bin0]:[index1]:[bin1]:[number of times data has been writtedn for this image]:[score for this particular data bin]
    f.write(str(s) + ";" + fn + ";")
    f.write(";".join(buff) + "\n")
f.close()

#save the supermap with all 36 choose 2 combinations, each of which has a 61x61 grid in the format:
#[K0,K1,buck0,buck1,J] 
open("master_map.json","wb").write(json.dumps(out_map))

