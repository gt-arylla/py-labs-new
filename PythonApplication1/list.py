import itertools as it
import copy


if(0): #make a list
    numbers=['brightness','luxpred','luspredcorr','avg','blue','lux']
    start_number_of_digits=1
    print "{",
    for iter in range(start_number_of_digits,len(numbers)+1):
    #for iter in range(start_number_of_digits,4):
        for val in it.combinations(numbers,iter):
            x_cols=list(val)
           # x_cols.extend([15,35])
            print "{",
            for iter in range(len(x_cols)-1):
                print x_cols[iter],
                print ";",
            print x_cols[-1],
            print "},",
    print "}"

  
if(0): #make a text list for python
    numbers=['blue_avg','red_avg','green_avg','value_avg','grey_avg']
    start_number_of_digits=1
    print "[",
    for iter in range(start_number_of_digits,len(numbers)+1):
    #for iter in range(start_number_of_digits,4):
        for val in it.combinations(numbers,iter):
            x_cols=list(val)
            x_cols.extend(['brightness'])
            print "[",
            for iter in range(len(x_cols)-1):
                print "'",
                print x_cols[iter],
                print "',",
            print "'",
            print x_cols[-1],
            print "'],",
    print "]"

if(0): #make a simple list
    numbers=[3140,3160,3180,3200,3220,3240,3260,3280,3300]
    start_number_of_digits=1
    print "{",
    #for iter in range(start_number_of_digits,len(numbers)+1):
    for number in numbers:
        x_cols=[number]
        x_cols_temp=[15 ,4,-101,2040]
        x_cols_temp.extend(x_cols)
        x_cols=copy.copy(x_cols_temp)
       # x_cols.extend()
        print "{",
        for iter in range(len(x_cols)-1):
            print x_cols[iter],
            print ",",
        print x_cols[-1],
        print "},",
    print "}"
    #print result
    #for val in result[3]:
    #    for val2 in val:
    #        print str(val2)+";",

if(0): #make a slightly more complex list
    numbers1=[2005,2011,2015,2021,2025,2031,2035,2041,2045,2051,2061,2081]
    numbers2=[3100,3120,3140,3160,3180,3200,3220]
    start_number_of_digits=1
    print "{",
    #for iter in range(start_number_of_digits,len(numbers)+1):
    for number1 in numbers1:
        for number2 in numbers2:
            x_cols=[]
            x_cols.append(number1)
            x_cols.append(number2)
            x_cols_temp=[15 ,4,-101]
            x_cols_temp.extend(x_cols)
            x_cols=copy.copy(x_cols_temp)
           # x_cols.extend()
            print "{",
            for iter in range(len(x_cols)-1):
                print x_cols[iter],
                print ",",
            print x_cols[-1],
            print "},",
    print "}"

if(0): #make a slightly more complex list
    start=0
    end=0
    index=0
    step=0.05
    max=1.01
    while start<=max:
        while end<=max:
            if end>start:
                print index,
                print ",",
                print start,
                print ",",
                print end
                index +=1
            end+=step
        start+=step
        end=start

if(1): #make a nested for loop type list
    super_tuple=(
        [6750,6625,6500,6375,6250],
        [3005,3009,3011,3015,3025]
        )
    all_output_numbers=[-10,22,300]
    counter=0
    print "{",
    for p in it.product(*super_tuple):
        counter+=1
        x_cols=list(p)
        x_cols.extend(all_output_numbers)
        print "{",
        for iter in range(len(x_cols)-1):
            print x_cols[iter],
            print ",",
        print x_cols[-1],
        print "},",
    print "};"
    print counter