#test the performance of the your coeffs against your transcript file

import os, math, sys, random, json, fun


classes = {}
for i in range(4):
  classes[i] = []

#prep empty coeffs vector
coeffs = [0.0 for i in range(630)]

#populate coeffs vector with data.  For now we don't include the intercept, so we skip a=0 and everything else gets put in an index back
for line in open("coeffs.txt"):
  a,v = line.strip().split()
  if int(a) == 0: continue
  coeffs[int(a)-1] = float(v)

test = []
for line in open("transcript.txt"):
  pieces = line.strip().split(";") 
  #class
  cl = int(pieces[0])
  #filename
  fn = pieces[1]
  #index data.  v is only the score value for the index data
  v = [float(x.strip().split(":")[-1]) for x in pieces[2:]]
  #throw an error if the size of v does not match the size of the coeffs
  assert len(v) == len(coeffs)
  #calculate the dumbscore by multiplying the coeff value by the v value and taking the sum of everything
  dumbscore = sum([cc*vv for cc,vv in zip(coeffs,v)])
  # TODO clean up this debugging hack
  #if fn.find("9327") > -1:
  #  for cc,vv,ss in zip(coeffs,v,pieces[2:]):
  #    print "%10.6f %10.6f %10s" % (cc,cc*vv,ss)
  #  print "==>", dumbscore
  
  #save the scores
  classes[cl].append(v)
  #if it's test data, add it to the test list
  #if cl in [2,3]: test.append( (dumbscore, random.uniform(0,1), cl, fn) )
  if cl in [0,1,2,3]: test.append( (dumbscore, random.uniform(0,1), cl, fn) )

#sort is by the first element, so ideally it'll be a block of negative samples then a block of positive samples
test.sort()
test.reverse()

#print test

npos = len(classes[2])
nneg = len(classes[3])

poskept = 0
negreject = nneg

targ = 0.85
#determine what the score threshold is.  the test list is sorted by score
for sco, ro, cl, fn in test:
 # print sco, fn  # debugging hack
  if poskept >= targ*npos: break
  if cl == 2 or cl == 0: poskept += 1
  else: negreject -= 1
  lastscore = sco

#print "        J score: %8.4f" % ((targ + 3*negreject/float(nneg))/4.0)
odds = math.exp(lastscore)
#print "    Threshold %%: %8.4f" % (odds/(1+odds))
#print "Threshold score: %8.4f" % lastscore

perf_dict={}
perf_dict["DM_J"]=(targ + 3*negreject/float(nneg))/4.0
perf_dict["DM_ThreshPerc"]=odds/(1+odds)
perf_dict["DM_ThreshScore"]=lastscore


for cl_filter,name in zip([[0,1],[2,3],[0,1,2,3]],["train","test","tot"]): 
    diff_holder=[]
    mark_holder=[]
    for sco, ro, cl, fn in test:
      if not cl in cl_filter: continue
      diff_holder.append(sco)
      if cl==0 or cl==2: mark_holder.append(1)
      elif cl==1 or cl==3: mark_holder.append(0)
    thresh,J_abs,J,sen,spec=fun.threshold_finder(diff_holder,mark_holder)


    odds = math.exp(thresh)
    perf_dict[name+"_"+"J"]=J
    perf_dict[name+"_"+"Sen"]=sen
    perf_dict[name+"_"+"Spec"]=spec
    perf_dict[name+"_"+"n_B"]=len(mark_holder)-sum(mark_holder)
    perf_dict[name+"_"+"n_P"]=sum(mark_holder)
    perf_dict[name+"_"+"ThreshPerc"]=odds/(1+odds)
    perf_dict[name+"_"+"ThreshScore"]=thresh


    print "        J score: %8.4f" % (J)

    print "    Threshold %%: %8.4f" % (odds/(1+odds))
    print "Threshold score: %8.4f" % thresh


with open("testperf.txt","wb") as file:
    file.write(json.dumps(perf_dict,sort_keys=True,indent=4,separators=(',', ': ')))