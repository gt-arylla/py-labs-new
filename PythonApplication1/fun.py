#functions
import numpy as np
import cv2
import matplotlib.pyplot as plt

#converts nan to binary where nan values are 0 and non-nan values are 1
def nanTObin(input_mat,flipper=False):
    output_mat=np.ones(input_mat.shape)
    if not flipper:
        for i in np.arange(input_mat.shape[0]):
            for j in np.arange(input_mat.shape[1]):
                if np.isnan(input_mat[i,j]):
                    output_mat[i,j]=0
    else:
        output_mat[:]=np.nan
        for i in np.arange(input_mat.shape[0]):
            for j in np.arange(input_mat.shape[1]):
                if not input_mat[i,j]==0:
                    output_mat[i,j]=0
    return output_mat

#makes a radially symmetrical gaussian filter based on input image dimentions
def gauss_maker(fig_rows,fig_cols,sigma_perc,anchor_perc_rows,anchor_perc_cols,max_val=1,flipper=False):
    #initialize sigma and k
    sigma=np.sqrt(fig_rows**2+fig_cols**2)*sigma_perc
    
    k=int(max([fig_rows,fig_cols]))*2
    sigma_fix=0.3*((k-1)*0.5-1)+0.8
    print str(sigma)+','+str(sigma_fix)
    #Calculate kernel
    kernel_1d=cv2.getGaussianKernel(k,sigma)
    kernel_1d=kernel_1d/max(kernel_1d)*max_val
    filler_fig=np.ones([fig_rows,fig_cols])

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
    if flipper:
        final_fig=-(filler_fig-max_val)
        filler_fig=final_fig
    return filler_fig

#automatically makes subplots from a list of mats
def plotter(input_list):
    numberOFplots=len(input_list)
    for i in np.arange(numberOFplots):
        plt.subplot(1,numberOFplots,i+1)
        plt.imshow(input_list[i])
    plt.show()
    return