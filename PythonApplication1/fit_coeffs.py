import os, math, sys, random

#prep a holding dictionary for the four classes
classes = {}
# class 0 = pos train, 1 = neg train, 2 = pos test, 3 = neg test
for i in range(4):
  classes[i] = []

def transcript_iter ():
    #this reads the transcript data and returns the class and the scores
  for line in open("transcript.txt"):
      #each line is for an individual image.  It is setup as [class] [filename] [DATA]
      #The DATA is set up as [a:b:c:d:e:f a:b:c:d:e:f a:b:c:d:e:f a:b:c:d:e:f ...]


    #split up the data by whitespace.
    pieces = line.strip().split(";") 
    #print line
    #print pieces
    #this is the class
    cl = int(pieces[0])
    #return list of the scores
    v = [float(x.strip().split(":")[-1]) for x in pieces[2:]]
    #the yeild return means we are returning a generator rather than a value
    #a generator is an object that can only be iterated over once
    #it's helpful because you don't need to use as much memory in the loop since the generator will forget the old data when it's calculating the new data
    yield cl,v

test = []
for cl,v in transcript_iter():
    #the length of v is always 630, since that is 36 choose 2
  NV = len(v)
  #I assume the dumbscore will be improved later as the machine learning progresses
  dumbscore = sum(v)
  classes[cl].append(v)

  #prep the regression with number of positive, negative, and base log ratio of the two
npos = len(classes[0])
nneg = len(classes[1])
base_logodds = math.log(npos/float(nneg))

#base coefficient is the inverse of the count of v
base_coeff = 1.0/NV
#concatenate odds and base_coeff
coeffs = [base_logodds] + [base_coeff for i in range(NV)]

pavLL = -99999999999999
numdone = 0
ssize = 0.1
#prep stuff.  Another concatenation
prev_good_coeffs = [1.0] + [base_coeff for i in range(NV)]
#prep empty vector the same size as prev_good_coeffs
prev_grad = [0.0 for x in prev_good_coeffs]
print "%8s %8s %8s %7s %8s" % ("LogLik", "Prob", "Iter", "Conv.", "Speed")
#machine learning!!!!  Iterate a bunch of times and make things better and better
while True and numdone < 1000000:
  numdone += 1 
  # calculate the log likelihood and build up the gradient
  avLL = 0.0
  grad = [0.0 for i in range(NV+1)]
  for cl,v in transcript_iter():
      #if it's test data skip it
    if cl not in [0,1]: continue
    #concatenate the intercept to the start of the scores
    v = [1.0] + v 
    tot = 0.0
    #prep empty array for all 630 scores plus the intercept
    curr_grad = [0.0 for x in grad]
    for i in range(len(v)):
        #calculate the t value for the current coeffs
      x = v[i]
      tot += coeffs[i]*x
      curr_grad[i] = x
    #this is the logistic regression equation.  (e^t)/(e^t+1)
    odds = math.exp(tot)
    prob = odds/(1.0 + odds)
    #say what you expect the probability to be.  expect 1 in the case of pos train.  Else, expect 0
    prob_targ = 1.0 if cl == 0 else 0.0
    #calculate your current error
    prob_err = prob_targ - prob
    #calculate the log likelihood - which we want to maximize
    LL = math.log(prob) if cl == 0 else math.log(1.0-prob)
    #find average log likelihood
    avLL += LL/float(npos+nneg)
    for j in range(len(coeffs)):
      # I've hacked the negative sign; where did I get it wrong above? (that's not a good sign)
      grad[j] -= prob_err*curr_grad[j] / float(npos+nneg)
    #difference between current average log likelihood and new log likelihood
  delta = avLL - pavLL
  #print current results
  print "\r%8.4f %8.4f %8i %8.4f %8.4f         " % (avLL, math.exp(avLL), numdone, (1e-6)/delta, ssize),
  #we want avLL to get bigger and bigger as the log likelihood increases.  This first case is if the new best guess is not better than the old guess
  if avLL < pavLL + 1e-6:
      #ssize ensures that we don't keep trying over and over if things are bad.
      #delta being negative means that our latest attempt was worse than our last attempt
    if ssize > 1e-6 and delta < 0:
      ssize *= 0.95
      coeffs = [x for x in prev_good_coeffs]
      for j in range(len(coeffs)):
          #we modify the coeffs to make them better
          #the coeffs are scaled down by whatever percent ssize currently is
          #we don't however allow the coeffs to go beyond the bounds of [0,1]
          #we ensure that the intercept is always base_logodds
        c = coeffs[j] - ssize*prev_grad[j]
        if c > 1.0: c = 1.0
        if c < 0.0: c = 0.0
        if j == 0: c = base_logodds
        coeffs[j] = c
    else:
        #program is not converging, so break
      break
  else:
      #update log likelihood to the current best
    pavLL = avLL
    prev_good_coeffs = [x for x in coeffs]
    for j in range(len(coeffs)):
        #scale coeffs by whatever ssize currently is
        #make ssize bigger at the end
      c = coeffs[j] - ssize*grad[j]
      if c > 1.0: c = 1.0
      if c < 0.0: c = 0.0
      if j == 0: c = base_logodds
      coeffs[j] = c
    ssize *= 1.05
    prev_grad = [x for x in grad]

#final coeffs file has 630 numbers in it, which is the combination of 36 choose 2
f = open("coeffs.txt","wb")
for j in range(len(coeffs)):
  f.write("%s %s\n" % (j,coeffs[j]))
f.close()

