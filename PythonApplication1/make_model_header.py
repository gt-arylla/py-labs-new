import json, sys

#master map is generated in 'make_feature_maps.py'
#it is a list of information relating to the 630 combinations:
#[combo0, combo1, combo2...combo629]
#combo0=[K0, K1, buck0, buck1, J]
#the contents of the list are:
#K0 - The first comparison index
#K1 - The second comparison index
#buck0 - the buckets for K0
#buck1 - the buckets for K1
#J - the 61x61 matrix

J = json.loads(open("master_map.json").read())

K = {}
#make a dictionary of coeffs.  It's important to note that this is all coeffs, including the intercept
for line in open("coeffs.txt"):
  id, w = line.strip().split()
  id = int(id)
  w = float(w)
  K[id] = w

print "float model[] = {"
for q in range(len(J)):
    #jj is the info relating to a single index pair
  jj = J[q]
  #w is the coefficient value.  You have to add one to q since there is no row for the intercept
  w = K[q+1]
  K0,K1,buck0,buck1,k0k1map = jj
  # 4*61
  #make four rows that describe the buckets.
  #row0 - left side of bucket0
  #row1 - right side of bucket0
  #row3 - left side of bucket1
  #row4 - right side of bucket1
  thing = [ [x[0] for x in buck0], [x[1] for x in buck0] ]
  thing += [ [x[0] for x in buck1], [x[1] for x in buck1] ]
  #then add the remaining 61x61 grid of normalized raio data to the big matrix
  #the data is added multiplied by the coefficient
  for row in k0k1map:
    # 61*61
    for i in range(len(row)):
      row[i] *= w
    thing.append(row)
  print "  //", q
  #thing is the 65 element set of data
  #the first four elements are the bin buckets
  #the remaining 61 elements are coefficients

  #the 'thing' matrix is printed to the console window in a pretty way so it can be pasted into the c code.
  for t in thing:
    for i in range(8):
        #out is 8 elements.  0->8, then 8->16, etc
      out = t[8*i:8*(i+1)]
      o1 = out[:6]
      o2 = out[6:]
      f1 = ",".join(["%6.4e"]*len(o1))
      f2 = ",".join(["%6.4e"]*len(o2))
      if len(o1) > 0:
        print "  " + f1 % tuple(o1), ","
      if len(o2) > 0:
        print "  " + f2 % tuple(o2), ","
print "};"

