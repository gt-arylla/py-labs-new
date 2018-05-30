import math, sys, os, random, struct, json

#lines is a list of each line of the text file
#the first line is the number of bins for the K0 and K1 index
#the following lines are cases where both the neg data and the pos data had counts in a given bin
#[K0 information] [K1 information] [number of positive elements for that bin] [number of negative elements for that bin]
lines = [x for x in open(sys.argv[1])]

#N1 and N2 are simply the number of bins for the K0 and K1 index
#they are both ints
N1,N2 = map(int, lines[0].strip().split())
N = N1

#make 61 by 61 matrices for all the bins
gridp = [ [0 for i in range(N)] for j in range(N) ]
gridn = [ [0 for i in range(N)] for j in range(N) ]
ngridp = [ [0 for i in range(N)] for j in range(N) ]
ngridn = [ [0 for i in range(N)] for j in range(N) ]

#buff is just converting the information in the input map file to a list of tuples
buff = []

for line in lines[1:]:
  i,j, np, nn = map(int, line.strip().split())
  buff.append( (i,j, np, nn) )

#find the total number of positive and negative cases in the buffer list
totp = sum([np for i,j,np,nn in buff])
totn = sum([nn for i,j,np,nn in buff])

#ratio is defined as positive cases divided by negative cases  Ideally negative is balanced with positive so this is equal to one
base_ratio = float(totp)/totn

#populate the matrices defined earlier with data
#there are two types of matrices - percent matrices and count matrices
#the former gets filled with the percent of the total negative or positive data that is in that bin
#the latter gets filled with the number of negative of positive data in that bin
for i,j,np,nn in buff:
  gridp[i][j] = np/float(totp)
  gridn[i][j] = nn/float(totn)
  ngridp[i][j] = np
  ngridn[i][j] = nn

def gridabsmax (grid):
    #find the maximum of the absolute value of the input grid
  buff = -1
  for i in range(len(grid)):
    for j in range(len(grid)):
      buff = max(buff, abs(grid[i][j]))
  return buff

def smooth(grid):
    #make input grid less spikey.  Values will be reduced if surrounding data is all 0
    #grid - 2D input matrix.  For this purpose it'll be 61x61
    #For this case the grid is a matrix of floats, which are the percentage occurance of each bin

  N = len(grid)
  #gg is a zeroes matrix the same dimention as the input matrix
  gg = [ [0 for i in range(N)] for j in range(N) ]
  #iterate through every element of the input matrix
  for i in range(N):
    for j in range(N):
      totw = 0
      totn = 0
      #look at data behind your focus
      if i - 1 >= 0:
        totw += 1.0
        totn += grid[i-1][j] 
    #look at data in front of your focus
      if i + 1 < N:
        totw += 1.0
        totn += grid[i+1][j] 
        #deal with the edge case when you're looking at j=0
      if j - 1 >= 0:
        totw += 1.0
        totn += grid[i][j-1] 
        #this is the typical case, when you're dealing with data in the body of the matrix
      if j + 1 < N:
        totw += 1.0
        totn += grid[i][j+1] 
      gg[i][j] = (grid[i][j] + totn/totw*0.5)/1.5
  return gg

def smoothn (grid, n=1):
    #run the smoothing function a number of times
  while n > 0:
    grid = smooth(grid)
    n -= 1
  return grid

#smooth out the count grids
gridp = smoothn(gridp,50)
gridn = smoothn(gridn,50)

def gridrat (g1, g2):
    #make a grid of zeroes
  v = [[0 for i in range(len(g1))] for j in range(len(g1))]
  #iterate through each number in the grid
  for i in range(len(g1)):
    for j in range(len(g1)):
      v1 = g1[i][j]
      v2 = g2[i][j]      
      v[i][j] = math.log((v1 + 1e-6)/(v2 + 1e-6))
  return v

#find the grid ratio of input positive and negative data
gr = gridrat(gridp,gridn)

def drawrat (grid, fn):
  pointmap = []
  #find maximum absolute value
  M = gridabsmax(grid)
  totalvalue = 0
  for i in range(len(grid)):
      #add a new empty list to holder and make the row be equal to that new list
    pointmap.append([])
    row = pointmap[-1]
    for j in range(len(grid)):
        #v is the normalized value between -1 and 1
        #it will be negative if it describes the negative case more than the positive case
      v = grid[i][j]/max(float(M),1e-6)
      #add the v value to row, which is then added to the pointmap
      row.append(v)
      #we add to the total the total count of positive and negative counts for a given index multiplied by the absolute value of the ln ratio for that index
      totalvalue += abs(grid[i][j])*(ngridp[i][j] + ngridn[i][j])
      #fr is the abs value of the normalized ratio.  I don't know why you have to take the min of abs(v) and 1 since the maximum that abs(v) can be is 1
      fr = min(abs(v),1.0)
      #the RGB information below isn't tied at all to the export mapdata, so my guess is that it's just used to draw the spikey graph
      #this is the case where the raio is more describing a positive value
      if v > 0: pR,pG,pB = 1.0, 1.0-fr, 1.0-fr
      #this is the case where ratio is more describing a negative value
      else: pR,pG,pB = 1.0-fr,1.0-fr,1.0
      pR,pG,pB = [x**(1.0/2.2) for x in [pR,pG,pB]]
      pR,pG,pB = [int(round(255*x)) for x in [pR,pG,pB]]
  #pointmap.append( totalvalue/(totp+totn+1e-6)  )
  #save the pointmap information to json
  open("mapdata.txt","wb").write(json.dumps(pointmap))

drawrat(gr,"rat.png")

