
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
print 'running...'
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

maxAccuracy = 0.0
cb_best=0

test_number=2

cb_list=[]
cct_list=[]
if test_number==0:
    cb_list=range(0,54)
    cct_list=[0]
elif test_number==2:
    cb_list=range(0,54)
    cct_list=range(0,54)
    cct_list=[39]
elif test_number==201:
    cb_list=range(0,54)
    cct_list=[7,8,18,26,33,39]
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
#Loop thru all cb bin vals
for cb in cb_list:
    for cct in cct_list:

        #read csv
        data = pd.read_csv('cbrgb_2.csv', header = 0)

        # data = data.dropna()
     #   print(data.shape)
        # print(list(data.columns))


        drop = range(data.shape[1])
        #Only keep Cb, LUX, Mark and Light Warmth columns
        mark_loc=0
        if test_number==0:
            drop = np.delete(drop, [770+cb, 830, 831, 834])
            mark_loc=2
        elif test_number==2:
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
            #temperature starts at 343
        data.drop(data.columns[drop], axis=1, inplace=True)
        #print(data.head())

        # sns.heatmap(data.corr())
        # plt.show()

        #Y data is the 4th column - mark
    

        y = data.iloc[:,mark_loc]
        #Drop the 4th col for X data
        data.drop(data.columns[[mark_loc]], axis=1, inplace=True)
        X = data.iloc[:,:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=0)
        X_train.shape

        classifier = LogisticRegression(solver='newton-cg', random_state = 0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_train)
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(y_train, y_pred)
       # print(confusion_matrix)

        print('CB CCT:,'+str(cb)+ ','+str(cct)+', {:.5f}'.format(classifier.score(X_train, y_train)))
        if (classifier.score(X_train, y_train) > maxAccuracy):
            #print maxAccuracy
            maxAccuracy = classifier.score(X_train, y_train)
            #print maxAccuracy
            #print cb_best
            cb_best=cb
            cct_best=cct
            #print cb_best
            #print '*******************************************NEW cb BEST******************'
            #from sklearn.metrics import classification_report
            #print(classification_report(y_train, y_pred))

print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(maxAccuracy))
print 'best cb is' +str(cb_best)
#print 'best cct is' +str(cb_best)