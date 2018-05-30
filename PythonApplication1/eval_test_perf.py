#test the performance of the your coeffs against your transcript file

import os, math, sys, random


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
  pieces = line.strip().split() 
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
  if cl in [2,3]: test.append( (dumbscore, random.uniform(0,1), cl, fn) )

#sort is by the first element, so ideally it'll be a block of negative samples then a block of positive samples
test.sort()
test.reverse()

npos = len(classes[2])
nneg = len(classes[3])

poskept = 0
negreject = nneg

targ = 0.85
#determine what the score threshold is.  the test list is sorted by score
for sco, ro, cl, fn in test:
  # print sco, fn  # debugging hack
  if poskept >= targ*npos: break
  if cl == 2: poskept += 1
  else: negreject -= 1
  lastscore = sco

print "        J score: %8.4f" % ((targ + 3*negreject/float(nneg))/4.0)
odds = math.exp(lastscore)
print "    Threshold %%: %8.4f" % (odds/(1+odds))
print "Threshold score: %8.4f" % lastscore
