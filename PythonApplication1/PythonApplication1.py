#Make gaussian filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltfig
import os
import winsound
import sys
import time
import csv
import math
from scipy.interpolate import griddata
import fun

#Define Paths
directory="G://Developer//py-labs//npyImport//App7//"
npy_name="DATA_flash_ close_time 1.0_iso 0.5_exp 0.0_blank_lux0.2k_avg"

# #Import data
npy_file=directory+npy_name+".npy"
mat_reference="Flash_ref.jpg"
data_raw=np.load(npy_file)
data_raw=cv2.resize(data_raw,None,fx=.1,fy=.1,interpolation = cv2.INTER_CUBIC)

# for sigma_perc in np.linspace(.40,1,60):
	# for anchor_perc_rows in np.linspace(.45,.55,100):
		# for anchor_perc_cols in np.linspace(0.38,0.48,100):
##__CODE__##
k=501
sigma_perc=0.4
sigma=np.sqrt(data_raw.shape[0]**2+data_raw.shape[1]**2)*sigma_perc
fig_rows=data_raw.shape[0]
fig_cols=data_raw.shape[1]
anchor_perc_rows=0.477272727273
anchor_perc_cols=0.391111111111

#Calculate kernel
kernel_1d=cv2.getGaussianKernel(k,sigma)
kernel_1d=kernel_1d/max(kernel_1d)
# print 'filter done'
filler_fig=np.ones([fig_rows,fig_cols])
# print kernel_1d[2121]
anchor_rows=int(fig_rows*anchor_perc_rows)
anchor_cols=int(fig_cols*anchor_perc_cols)

for j in np.arange(fig_cols):
	for i in np.arange(fig_rows):
		anchor_diff_row=i-anchor_rows
		anchor_diff_col=j-anchor_cols
		kernel_loc_row=int(np.ceil(k/float(2))+anchor_diff_row)
		kernel_loc_col=int(np.ceil(k/float(2))+anchor_diff_col)
		# print kernel_loc
		filler_fig[i,j]=filler_fig[i,j]*kernel_1d[kernel_loc_row]*kernel_1d[kernel_loc_col]
# print 'stage 1'
# for j in np.arange(fig_rows):
	# for i in np.arange(fig_cols):
		# anchor_diff=i-anchor_cols
		# kernel_loc=int(np.ceil(k/float(2))+anchor_diff)
		# # print kernel_loc
		# filler_fig[j,i]=filler_fig[j,i]*kernel_1d[kernel_loc]
# print 'stage 2'
# final_fig=cv2.sepFilter2D(filler_fig,-1,kernel_1d,kernel_1d,delta=5)
# final_fig=cv2.sepFilter2D(filler_fig,-1,kernel_1d,kernel_1d,anchor=(anchor_rows,anchor_cols))

final_fig=-(filler_fig-1)

sub_fig=data_raw+15*final_fig-126
sub_fig_abs=np.abs(sub_fig)
sum_check=np.nansum(sub_fig_abs)
dev_check=np.nanstd(sub_fig)
print 'perc rows cols sum,'+str(sigma_perc)+','+str(anchor_perc_rows)+','+str(anchor_perc_cols)+','+str(sum_check)+','+str(dev_check)



min_range=np.nanpercentile(data_raw,1)
max_range=np.nanpercentile(data_raw,99)


plt.subplot(131)
plt.imshow(data_raw,cmap='jet',vmin=min_range,vmax=max_range)
plt.colorbar()
plt.subplot(132)
plt.imshow(final_fig,cmap='jet',vmin=0,vmax=1)
plt.colorbar()
plt.subplot(133)
min_range=np.nanpercentile(sub_fig,1)
max_range=np.nanpercentile(sub_fig,99)
plt.imshow(sub_fig,cmap='jet')
plt.colorbar()
# plt.imshow(sub_fig)
plt.show()