#Make gaussian filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fun
import copy
print 'running'

#Define Paths
directory="G://Developer//py-labs//npyImport//App7//"
npy_name="DATA_flash_ close_time 1.0_iso 0.5_exp 0.0_blank_lux0.2k_avg"

# #Import data
npy_file=directory+npy_name+".npy"
mat_reference="Flash_ref.jpg"
data_raw=np.load(npy_file)
#data_raw=cv2.resize(data_raw,None,fx=.1,fy=.1,interpolation = cv2.INTER_CUBIC)
data_save=copy.copy(data_raw)
for erode_val in np.array([99]):
    #Erode data
    #erode_val=9
    data_bin=fun.nanTObin(data_save)
    kernel = np.ones((erode_val,erode_val),np.uint8)
    data_erode = cv2.erode(data_bin,kernel,iterations = 1)
    data_erode_nan=fun.nanTObin(data_erode,True)
    data_raw=data_raw+data_erode_nan
    #fun.plotter([data_bin,data_erode])

    for sigma_perc in np.linspace(0.291666666667,1,1):
	            # for anchor_perc_rows in np.linspace(.45,.55,100):
		            # for anchor_perc_cols in np.linspace(0.38,0.48,100):
        ##__CODE__##
        k=501
        #sigma_perc=0.125
        fig_rows=data_raw.shape[0]
        fig_cols=data_raw.shape[1]
        anchor_perc_rows=0.477272727273
        anchor_perc_cols=0.391111111111

        final_fig=fun.gauss_maker(fig_rows,fig_cols,sigma_perc,anchor_perc_rows,anchor_perc_cols,flipper=True)

        sub_fig=data_raw+15*final_fig-126
        sub_fig_abs=np.abs(sub_fig)
        sum_check=np.nansum(sub_fig_abs)
        dev_check=np.nanstd(sub_fig)
        range1=np.nanpercentile(sub_fig,90)-np.nanpercentile(sub_fig,10)
        range2=np.nanpercentile(sub_fig,95)-np.nanpercentile(sub_fig,5)
        range3=np.nanpercentile(sub_fig,99)-np.nanpercentile(sub_fig,1)
        print 'erode perc rows cols sum,'+str(erode_val)+','+str(sigma_perc)+','+str(anchor_perc_rows)+','+str(anchor_perc_cols)+','+str(sum_check)+','+str(dev_check)+','+str(range1)+','+str(range2)+','+str(range3)



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