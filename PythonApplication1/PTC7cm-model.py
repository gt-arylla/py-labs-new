# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:07:28 2018

@author: gagandeepphull"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# This function is used to convert the raw pixel data from the 'clicktosave' program into a list of values that
#   correspond to the difference in pixel values between each 1cm interval on the ruler.
# Note: This function can determine the orientation of the ruler in the image
def PullData(filename):

    # Pulling original data
    data = json.load(open(filename))
    KeyValues = data.keys()

    # xPixelDiff and yPixelDiff are used to determine if the Pixels are changing in the x or y direction
    # This is how the function determines the ruler's orientation
    xPixelDiff = abs(data[KeyValues[0]][1][0] - data[KeyValues[0]][0][0])
    yPixelDiff = abs(data[KeyValues[0]][1][1] - data[KeyValues[0]][0][1])

    num = range(0,len(data))
    PixelData = range(0,len(num))

    # These two loops convert the raw data into a list containing the difference in pixel values between each interval
    for x in num:
        # The original file outputs a dictionary file with all the raw data
        # This outer loop iterates through each key of the dictionary
        # A is the variable that is assigned the raw data
        A = data[KeyValues[x]]
        # Pixels is the variable that is used to store the values
        Pixels = range(0,len(A)-1)

        for y in range(1,len(A)):
            # These if statements are what differentiate the ruler's orientation
            if xPixelDiff > yPixelDiff:
                # The calculation of pixel difference
                Pixels[y-1] =abs(A[y][0] - A[y-1][0])
                # The final value is stored in PixelData below
                PixelData[x] = Pixels
            if yPixelDiff > xPixelDiff:
                Pixels[y - 1] = abs(A[y][1] - A[y - 1][1])
                PixelData[x] = Pixels

    return PixelData


# This function determines if the user's clicking is causing significant error or not
# Note: This analysis requires the same interval (ex. 0-1cm) for all 10 pictures to be analyzed, which
#   requires a reformat of the data structure that was pulled in the 'PullData' function
def HumanError(Data):

    #Setting up the structure that will have the data of interest
    LocationData = [0]*len(Data[0])
    for x in range(0,len(LocationData)):
         LocationData[x] = [x]*len(Data)

    # Assigning the data as needed to the variable 'LocationData'
    for y in range(0,len(Data[0])): # 1-9
        for x in range(0,len(Data)): #1-10
            LocationData[y][x] = Data[x][y]

    # Setting up a variable to analyze the standard deviation between 'clicks'
    Stdevlocation = [0]*len(LocationData)

    # Calculating the standard deviation between 'clicks'
    # Note: The standard deviation would normally be in units of 'pixels' however, I converted
    #   the value to mm
    # I assumed that the average of the data is the real pixel-to-cm ratio in order to convert the standard
    #   deviation to mm
    for x in range(0,len(Stdevlocation)):
        Stdevlocation[x] = (np.std(LocationData[x])/np.average(LocationData[x]))*10

    # Plotting the results
    plt.bar(range(1,len(Stdevlocation)+1),Stdevlocation)
    plt.title('Deviation in PTC Ratio due to Human Error')
    plt.ylabel('Standard Deviation [mm]')
    plt.xlabel('Interval Number')
    plt.show()
    return Stdevlocation


# For each ruler position, are 'X' pictures taken, and each of these pictures have different PTC ratios
# This function determines if there is a significant deviation between the PTC ratio determined
#   from each picture's data
def PictureDeviation(Data):
    # Setting up a variable for the deviation
    Stdevpicture= [0]*len(Data)

    # Calculating the standard deviation between pictures
    # As above:
        # Note: The standard deviation would normally be in units of 'pixels' however, I converted
        #   the value to mm
        # I assumed that the average of the data is the real pixel-to-cm ratio in order to convert the standard
        #   deviation to mm
    for x in range(0,len(Data)):
        Stdevpicture[x]= np.std(Data[x])/np.average(Data[x])*10

    # Plotting the results
    plt.bar(range(1,len(Stdevpicture)+1),Stdevpicture)
    plt.title('Deviation in PTC Ratio due to Error Between Pictures')
    plt.ylabel('Standard Deviation [mm]')
    plt.xlabel('Picture Taken')
    plt.show()

    return Stdevpicture


# Getting the all the file names that are of interest
list = sorted(os.listdir('Pixel-To-Cm-LargeDataset'))

# Setting up the variables of interest
Stdevhumanerror = [0]*len(list)
Stdevrulererror = [0]*len(list)
Speclocdata = [0]*(len(list))
Ratio = [1] * len(list)

# This for loop iterates through all the files of interest
for x in range(0,len(list)):

    # Gets filename
    filepath = 'Pixel-To-Cm-LargeDataset{}{}'.format('/',list[x])
    
    # Pulls data from the file of interest and put it into a useful format
    Data = PullData(filepath)

    # Temp variable used to calculate PTC ratio
    PTC = [1] * len(Data)

    # for loop used to average all the pixel difference values in each file
    # Note: this approach is taken when it is determined there is negligible sources of error
    for y in range(0,len(Data)):
        PTC[y] = np.average(Data[y])
    # Averaging out all the values in the PTC matrix
    Ratio[x] = np.average(PTC)

    # Analyzes human error
    X = HumanError(Data)
    # Calculates the average standard deviation of the human error
    Stdevhumanerror[x] = np.average(X)
    
    # Analyzes error between pictures
    Y = PictureDeviation(Data)
    # Calculates the average standard deviation between pictures
    Stdevrulererror[x] = np.average(Y)

# Getting the final PTC ratio by averaging 'Ratio'
PixelToCmRatio = np.average(Ratio)

# Returns statement of results

print 'Human Error Results:'
print 'The standard deviation of the pixel to cm ratio as a result of human error in all the horizontal pictures is {} {}'.format(np.average(Stdevhumanerror[0:5]),'mm.')
print 'The standard deviation of the pixel to cm ratio as a result of human error in all the vertical pictures is {} {}'.format(np.average(Stdevhumanerror[5:10]),'mm.')
print ''
print 'Picture To Picture Deviation:'
print 'The standard deviation of the pixel to cm ratio as a result of moving from left to right across a horizontal picture is {} {}'.format(np.average(Stdevrulererror[0:5]),'mm.')
print 'The standard deviation of the pixel to cm ratio as a result of moving up a vertical image is {} {}'.format(np.average(Stdevrulererror[5:10]),'mm.')
print ''
print 'The ratio is {} {}'.format(PixelToCmRatio, 'pixels per cm')