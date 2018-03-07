

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
print 'running...'
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

maxAccuracy = 0.0
cb_best=0

test_number=81

cb_list=[]
cct_list=[]
if test_number==0:
    cb_list=range(0,54)
    cct_list=[0]
if test_number==3:
    cb_list=range(0,54)
    cct_list=[17]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-PictureSet-0'
elif test_number==301:
    cb_list=[7]
    cct_list=[17]
    val_keep_list_list=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    filter_col='0-PictureSet-0'
elif test_number==302:
    cb_list=[17]
    cct_list=range(0,54)
    val_keep_list_list=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    filter_col='0-PictureSet-0'
elif test_number==303:
    cb_list=[9]
    cct_list=[37]
    val_keep_list_list=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    filter_col='0-PictureSet-0'
elif test_number==304 or test_number==305:
    cb_list=[9]
    cct_list=[37]
    val_keep_list_list=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    #val_keep_list_list=[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    #val_keep_list_list=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20]]
    filter_col='0-PictureSet-0'
elif test_number==31:
    cb_list=[15]
    cct_list=range(0,54)
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-PictureSet-0'
elif test_number==32:
    cb_list=range(14,54)
    cct_list=[17]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-PictureSet-0'
if test_number==2:
    cb_list=range(0,54)
    cct_list=range(0,54)
    cct_list=[39]
elif test_number==200:
    cb_list=range(0,54)
    cct_list=[7,8,18,26,33,39]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-Org-PhotoSet'
elif test_number==2001:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-Org-PhotoSet'
elif test_number==20011:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[6,7,8],[10,11,12],[14,15,16]]
    filter_col='0-Org-PhotoSet'
elif test_number==222:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[0],[50],[100],[30],[70]]
    filter_col='0-Org-Yellow'
elif test_number==223:
    cb_list=range(0,54)
    cct_list=[39]
    val_keep_list_list=[[100],[200],[300],[325],[400],[450],[500],[600],[700],[900],[1300],[1400],[1500],[1600],[1700],[2500],[3000],[3500],[1000]]
    filter_col='0-Org-Lux'
elif test_number==21:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==211:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==212:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==213:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==22: #toggle turning on and off thickness and white count, use best CCT
    cb_list=range(0,54)
    cb=0
    cct_list=[[0,277+cb, 333, 334, 343+39],[1,277+cb, 333, 334, 343+39],[277+cb, 333, 334, 343+39]]
elif test_number==3:
    cb_list=range(0,54)
    cct_list=[0]
if test_number==4:
    lux_list=[[100,200,300],[325,400,500,600,700,1000],[1300,1400,1500,1600,2500,3000,3500]]
    lux_filter_col='0-LUX-0'
    yellow_list=[[0],[50],[100]]
    yellow_filter_col='0-Yellow-0'
    cb_list=[9]
    cct_list=[37]
if test_number==5:
    cct_list=range(0,4961)
    cb_list=[0]
elif test_number==51:
    cb_list=[0]
    cct_list=[497,1043,506,2477,1044,1208,507,516,1042,515,1041,2478,1040,1207,498,4963]
if test_number==7:
    cb_list=[0]
    cct_list=[0]
if test_number==8:
    cb_list=[0]
    cct_list=[0]
    filter_col='PhotoSet'
    val_keep_list_list=[[11,12,13,14],[11,12,15,16],[11,12,17,18],[11,12,19,20]]
if test_number==81:
    cb_list=[0]
    cct_list=[0]
    filter_col='PhotoSet'
    val_keep_list_list=[[11,13],[12,14],[11,15],[12,16],[11,17],[12,18],[11,19],[12,20],[11,12,13,14],[11,12,15,16],[11,12,17,18],[11,12,19,20]]
    #val_keep_list_list=[[11,13],[12,14],[11,15],[12,16],[11,17],[12,18],[11,19],[12,20]]
    #val_keep_list_list=[[11,17]]
    csv_file='171112_NewApp_Training_Set.csv'
if test_number==9:
    cb_list=[0]
    cct_list=[0]
    csv_file='171114_ROI_test_679_NOrando.csv'
    val_keep_list_list=[[1]];
#Loop thru all cb bin vals
for cb in cb_list:
    for cct in cct_list:
       # for yellow_list_index in range(0,len(yellow_list)):
           # for lux_list_index in range(0,len(lux_list)):
       for val_keep_list_index in range(0,len(val_keep_list_list)):
            #print val_keep_list_list[val_keep_list_index]
            #for c_val in np.linspace(0.01,1,20):
            #read csv
            #data = pd.read_csv('cbrgb_4.csv', header = 0)
            
            data=pd.read_csv(csv_file, header=0)
            #print(data.head(100))
            # data = data.dropna()
            #print(data.shape)
            # print(list(data.columns))

            ###remove rows as needed - based on column values
            #row_drop_list=[]
            #col_of_interest=332 #PhotoSet
            #for row in range(0,data.shape[0]-1):
            #    keep_row=False
            #    for val_keep in val_keep_list:
            #        if data.iloc[row,col_of_interest]==val_keep:
            #            keep_row=True
            #    if not keep_row:
            #        row_drop_list.append(row)
            #        print row_drop_list
            #data=data.loc[data[lux_filter_col].isin(lux_list[lux_list_index])]
            #data=data.loc[data[yellow_filter_col].isin(yellow_list[yellow_list_index])]
            if not val_keep_list_list[val_keep_list_index]==-1:
                data2=data.loc[data[filter_col].isin(val_keep_list_list[val_keep_list_index])]
            
            #print(data2.head(10))
            #print type(data)
                data=data2.loc[data['Count']==0]
            #print(data.head(10))
            #data.drop(row_drop_list)
            #print(data.shape) 
        
            drop = range(data.shape[1])
            #Only keep Cb, LUX, Mark and Light Warmth columns
            mark_loc=0
            if test_number==0:
                drop = np.delete(drop, [770+cb, 830, 831, 834])
                mark_loc=2
            if test_number==3 or test_number==31 or test_number==301: #drop white count
                drop = np.delete(drop, [0,1,332+cb,388,396,397+cct,453,454]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=7
                #temperature starts at 343
            elif test_number==32: #drop white count
                drop = np.delete(drop, [0,1,66,121,231,332+cb,388,396,397+cct,453,454]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=7
            elif test_number==302: #drop white count
                drop = np.delete(drop, [0,1,332+cb,387,388,390,392,394,395,397+cct,453,454]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=11
            elif test_number==303: #drop white count
                drop = np.delete(drop, [0,1,39,94,204,259,332+cb,387,388,390,392,394,395,397+cct,453,454,455]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=16
                #temperature starts at 343
            elif test_number==304 or test_number==4: #drop white count
                drop = np.delete(drop, [0,1,39,94,204,259,332+cb,387,388,390,392,394,395,397+cct,453,455]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=15
                #temperature starts at 343
                #drop = np.delete(drop, [0,1,39,94,204,259,332+cb,387,388,390,392,395,397+cct,453,455]) #thick;white count;Cb;Lux;Mark;Yellow
                #mark_loc=14
            elif test_number==305: #drop white count
                drop = np.delete(drop, [0,1,39,94,204,332+cb,388,390,392,394,395,397+cct,453,455]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=13
                #temperature starts at 343
            
            if test_number==2:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==200:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==2001:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 337])
                mark_loc=4
                #temperature starts at 343
            elif test_number==20011:
                drop = np.delete(drop, [0,1,66,121,277+cb, 333, 334, 337])
                mark_loc=6
                #temperature starts at 343
            elif test_number==222:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==223:
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 343+cct])
                mark_loc=4
                #temperature starts at 343
            elif test_number==201: #drop thickness
                drop = np.delete(drop, [1,277+cb, 333, 334, 343+cct])
                mark_loc=3
                #temperature starts at 343
            elif test_number==21: #use yellow value instead of cct
                drop = np.delete(drop, [0,1,277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=4
                #temperature starts at 343
            elif test_number==211: #drop thickness
                drop = np.delete(drop, [1,277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=3
                #temperature starts at 343
            elif test_number==212: #drop white count
                drop = np.delete(drop, [0,277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=3
                #temperature starts at 343
            elif test_number==213: #drop white count
                drop = np.delete(drop, [277+cb, 333, 334, 337]) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=2
                #temperature starts at 343
            elif test_number==22: #use yellow value instead of cct
                drop = np.delete(drop, cct) #thick;white count;Cb;Lux;Mark;Yellow
                mark_loc=4
            if test_number==5:
                drop = np.delete(drop, [cct,4963])
                mark_loc=1
            elif test_number==51:
                #cct=253
                #drop = np.delete(drop, [cct,4963])
                drop_list=[0,2522,cct,4963]
                drop_list.sort
                drop=np.delete(drop,drop_list)
                mark_loc=3
                #temperature starts at 343
            if test_number==7:
                drop=np.delete(drop,[0,1,39,94,149,204,231,286,332,333,335,337,339,340,379,398,400])
                mark_loc=16
            if test_number==8 or test_number==81:
                drop=np.delete(drop,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19])
                #drop=np.delete(drop,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18])
                mark_loc=15
            if test_number==9:
                drop=np.delete(drop,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16])
                mark_loc=15
            data.drop(data.columns[drop], axis=1, inplace=True)
            #print(data.head(100))

            #print sns.heatmap(data.corr())
            #plt.show()

            #Y data is the 4th column - mark
            data_print=data.loc[data['Mark'].isin([1])]
            data_blank=data.loc[data['Mark'].isin([0])]

            y = data.iloc[:,mark_loc]
            splitter_col_list2=[['Mark',0],['Mark',1]]
            splitter_col_list=[['Lux_Round',0],['Lux_Round',500],['Lux_Round',1000]]
            x_list=[]
            y_list=[]
            for splitter_index in splitter_col_list:
                data_split0=data.loc[data[splitter_index[0]].isin([splitter_index[1]])]
                for splitter_index2 in splitter_col_list2:
                    data_split=data_split0.loc[data[splitter_index2[0]].isin([splitter_index2[1]])]
                    y_data=data_split.iloc[:,mark_loc]
                    y_list.append(y_data)
                    data_split.drop(data_split.columns[[mark_loc]], axis=1, inplace=True)
                    data_split.drop(data_split.columns[[mark_loc]], axis=1, inplace=True)
                    x_data=data_split.iloc[:,:]
                    x_list.append(x_data)
            y_print=data_print.iloc[:,mark_loc]
            y_blank=data_blank.iloc[:,mark_loc]
            #Drop the 4th col for X data
            data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
            data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
            data_print.drop(data_print.columns[[mark_loc]], axis=1, inplace=True)
            data_blank.drop(data_blank.columns[[mark_loc]], axis=1, inplace=True)
            X = data.iloc[:,:]
            X_print=data_print.iloc[:,:]
            X_blank=data_blank.iloc[:,:]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)
            x_plot_val=[]
            y1_plot_val=[]
            y2_plot_val=[]
            y3_plot_val=[]
            for perc in np.linspace(0,.99,50):
                for iter_split in range(0,len(x_list)):
                    X_train_perc_split, X_test, y_train_perc_split, y_test = train_test_split(x_list[iter_split], y_list[iter_split], test_size=perc, random_state=0)
                   
                    if iter_split==0:
                        X_train_perc=X_train_perc_split
                        y_train_perc=y_train_perc_split
                    else:
                        X_train_perc_temp=X_train_perc.append(X_train_perc_split,ignore_index=True)
                        y_train_perc_temp=y_train_perc.append(y_train_perc_split,ignore_index=True)
                        X_train_perc=X_train_perc_temp
                        y_train_perc=y_train_perc_temp
                #class_dict={'0-PixelPerc-0':0.5,'0-Thickness-0':0.5,'0-Hue-37':0.2,'0-Sat-37':0.2,'0-Blue-37':0.2,'0-Green-37':0.2,'0-CB-9':1,'2-p0-0':0.4,'2-p10-0':0.8,'2-p50-0':0.5,'2-p50-0':0.5,'2-p90-0':0.5,'2-r0-0':0.5,'2-r10-0':0.5,'c-CCT-37':1,'0-LUX-9':1}
                if X_train_perc.shape[0]>5:
                    classifier = LogisticRegression(solver='newton-cg', random_state = 0,fit_intercept=True,class_weight=None)
                    #classifier=LogisticRegressionCV(solver='lbfgs')
                    classifier.fit(X_train_perc, y_train_perc)
                    #print(X_train_perc.head(10))
                    #print(X_train.head(10))
                    y_pred = classifier.predict(X_train)
                    from sklearn.metrics import confusion_matrix
            
                    #for val_print_list_index in range(0,len(val_print_list_list)):
                    #    data2=pd.read_csv('cbrgb_4.csv', header = 0);
                    #    data2=data.loc[data[filter_col].isin(val_keep_list_list[val_keep_list_index])]
                    #    drop = range(data2.shape[1])
                    #    if test_number==304: #drop white count
                    #        drop = np.delete(drop, [0,1,39,94,204,259,332+cb,387,388,390,392,394,395,397+cct,453,455]) #thick;white count;Cb;Lux;Mark;Yellow
                    #        mark_loc=15

                    confusion_matrix = confusion_matrix(y_train, y_pred)
                    #print confusion_matrix
                    tn=confusion_matrix[0,0]
                    fp=confusion_matrix[0,1]
                    tp=confusion_matrix[1,1]
                    fn=confusion_matrix[1,0]
                    tpr=tp/float(tp+fn)
                    tnr=tn/float(tn+fp)
                    fpr=fp/float(fp+tn)
                    #truth=tpr-(1-tnr)
                    truth=tpr-fpr
        
                    #print 'Truth: ' + str(truth)
                    #print 'TPP: '+str(tpr)
                    #print 'fpr: '+str(fpr)

                    #for row in range(0,len(X_train.index)):
                    #    mark_perc= classifier.predict_proba(X_train.iloc[[row]])[0,1]
                    #    LUX= X_train.iloc[[row]].values[0,14]
                    #    print " ",
                    #    mark= y_train.iloc[[row]].values[0]
                    #    if mark_perc<0.5:
                    #        pred_mark=0
                    #        confidence=(0.5-mark_perc)*2
                    #    else:
                    #        pred_mark=1
                    #        confidence=(mark_perc-0.5)*2
                    #    if pred_mark==mark:
                    #        accuracy=1
                    #    else:
                    #        accuracy=0
                    #    multiple1=100
                    #    multiple2=300
                    #    if LUX<=1000:
                    #        MultLux=np.floor((LUX + multiple1/2) / multiple1) * multiple1
                    #    else:
                    #        MultLux=np.floor((LUX + multiple2/2) / multiple2) * multiple2
                    #    MultLux=int(MultLux)
                    #    print str(mark_perc)+","+str(LUX)+","+str(MultLux)+","+str(mark)+","+str(pred_mark)+","+str(accuracy)+","+str(confidence)

                     #print classifier.predict_proba(aray)
                    #print classifier.sparsify()
                    #print(confusion_matrix)
                    #print y_train.mean()
                    #print pd.DataFrame(zip(X.columns, np.transpose(classifier.coef_)))
                    #print classifier.coef_
                    #print classifier.intercept_
                    x_plot_val.append(X_train_perc.shape[0])
                    y1_plot_val.append(truth)
                    y2_plot_val.append(tpr)
                    y3_plot_val.append(fpr)
                    print('Index CB CCT:,'+str(val_keep_list_index)+','+str(cb)+ ','+str(cct)+','+str(data.shape[0])+', {:.5f}'.format(classifier.score(X_train, y_train)))+","+str(perc)+ "," + str(X_train_perc.shape[0])+","+str(truth)+","+str(tpr) + "," + str(fpr) + "," + str(tp) + "," + str(fn) + "," + str(tn) + "," + str(fp)
            plt.plot(x_plot_val,y1_plot_val,x_plot_val,y2_plot_val,x_plot_val,y3_plot_val)
            plt.title(str(val_keep_list_list[val_keep_list_index]))
            axes = plt.gca()
            axes.set_ylim([0,1])
            plt.savefig(str(val_keep_list_list[val_keep_list_index])+ '.jpg', format='jpg', dpi=200)
            plt.clf()

                #if (classifier.score(X_train, y_train) > maxAccuracy):
                #    #print maxAccuracy
                #    maxAccuracy = classifier.score(X_train, y_train)
                #    #print maxAccuracy
                #    #print cb_best
                #    cb_best=cb
                #    cct_best=cct
                #    #val_keep_list_index_best=val_keep_list_index
                #    #print cb_best
                #    #print '*******************************************NEW cb BEST******************'
                #    #from sklearn.metrics import classification_report
                #    #print(classification_report(y_train, y_pred))

#print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(maxAccuracy))
#print 'best cb is' +str(cb_best)
#print 'best cct is' +str(cct_best)
#print 'best index is' +str(val_keep_list_index_best)
##print 'best cct is' +str(cb_best)