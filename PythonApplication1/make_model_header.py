import json, sys,os

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

load_file="master_map.json"
serial_input=False
serial_number=""
if len(sys.argv)>1:
    serial_input=True
    serial_number=str(int(sys.argv[1]))
if serial_input: load_file="master_map_"+serial_number+".json"
J = json.loads(open(load_file).read())

K = {}
#make a dictionary of coeffs.  It's important to note that this is all coeffs, including the intercept
for line in open("coeffs.txt"):
  id, w = line.strip().split()
  id = int(id)
  w = float(w)
  K[id] = w
lf = open("model.json", "wb")
model_name="model";
if serial_input: model_name="model_"+serial_number
superstring=''
#lf.write('{"model": [')
superstring+='{"model": ['

#print "float model[] = {"
counter=0
tuple_list=[]
consec_index=-1

#for q in range(len(J)):
#    #jj is the info relating to a single index pair
#  jj = J[q]
#  #w is the coefficient value.  You have to add one to q since there is no row for the intercept
#  w = K[q+1]
#  K0,K1,buck0,buck1,k0k1map = jj
#  # 4*61
#  #make four rows that describe the buckets.
#  #row0 - left side of bucket0
#  #row1 - right side of bucket0
#  #row3 - left side of bucket1
#  #row4 - right side of bucket1

#  step0=buck0[0][1]-buck0[0][0]
#  start0=buck0[0][0]

#  step1=buck1[0][1]-buck1[0][0]
#  start1=buck1[0][0]
#  thing=[]
#  for row in k0k1map:
## 61*61
#      for i in range(len(row)):
#          row[i] *= w
#      thing.append(row)
#  print len(thing)
#  print len(thing[0])
#  lf.write("%6.4e" % step0 + ",")
#  lf.write("%6.4e" % start0 + ",")
#  lf.write("%6.4e" % step1 + ",")
#  lf.write("%6.4e" % start1 + ",")

#  #the 'thing' matrix is printed to the console window in a pretty way so it can be pasted into the c code.
#  compression=True
#  if compression:
#      for t in thing:
#          for val in t:
#              if (1):
#                  if not val==0:
#                      tuple_list.append((counter,val))
#                      lf.write(str(int(counter))+",")
#                      lf.write("%6.4e" % val + ",")
#                      #print counter
#              counter+=1
#              #elif (1):
#              #    if not val==0:
#              #         lf.write("%6.4e" % val + ",")
#              #    else:
#              #         lf.write("0,")
#              #else:
#              #    if not val==0:
#              #        if counter==consec_index+1:
#              #            consec_index=counter
#              #        else:
#              #            lf.write(str(int(counter))+",")
#              #counter+=1
#      continue
#  #else:
#  #    for t in thing:
#  #      for i in range(8):
#  #          #out is 8 elements.  0->8, then 8->16, etc
#  #        out = t[8*i:8*(i+1)]
#  #        o1 = out[:6]
#  #        o2 = out[6:]
#  #        f1 = ",".join(["%6.4e"]*len(o1))
#  #        f2 = ",".join(["%6.4e"]*len(o2))
#  #        if len(o1) > 0:
#  #          #print "  " + f1 % tuple(o1), ","
#  #          lf.write( "  " + f1 % tuple(o1)+ ","+"\n")
#  #        if len(o2) > 0:
#  #          #print "  " + f2 % tuple(o2), ","
#  #          lf.write( "  " + f2 % tuple(o2)+ ","+"\n")


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
  #print "  //", q
  #lf.write("  //")
  #lf.write(str(q)+"\n")
  #thing is the 65 element set of data
  #the first four elements are the bin buckets
  #the remaining 61 elements are coefficients

  #the 'thing' matrix is printed to the console window in a pretty way so it can be pasted into the c code.
  compression=True
  if compression:
      for t in thing:
          for val in t:
              if (0):
                  if not val==0:
                      tuple_list.append((counter,val))
                      lf.write(str(int(counter))+",")
                      lf.write("%6.4e" % val + ",")
              elif (1):
                  if not val==0:
                      superstring+="%3.2e" % val + ","
                       #lf.write("%3.2e" % val + ",")
                  else:
                       superstring+="0,"
                       #lf.write("0,")
              else:
                  if not val==0:
                      if counter==consec_index+1:
                          consec_index=counter
                      else:
                          lf.write(str(int(counter))+",")
              counter+=1
      continue
  else:
      for t in thing:
        for i in range(8):
            #out is 8 elements.  0->8, then 8->16, etc
          out = t[8*i:8*(i+1)]
          o1 = out[:6]
          o2 = out[6:]
          f1 = ",".join(["%6.4e"]*len(o1))
          f2 = ",".join(["%6.4e"]*len(o2))
          if len(o1) > 0:
            #print "  " + f1 % tuple(o1), ","
            lf.write( "  " + f1 % tuple(o1)+ ","+"\n")
          if len(o2) > 0:
            #print "  " + f2 % tuple(o2), ","
            lf.write( "  " + f2 % tuple(o2)+ ","+"\n")


superstring=superstring[:-1]
superstring+="]}"
lf.write(superstring)

#lf.write("]}")
lf.close()

#print tuple_list
print len(tuple_list)

#new_model_name='model_data.c'
#if serial_input: new_model_name="model_data_"+serial_number+".c"
#os.rename('model_data.txt',new_model_name)

