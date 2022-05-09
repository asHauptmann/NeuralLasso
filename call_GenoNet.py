# Script to call the GenoNet class file
# Andreas Hauptmann, University of Oulu/UCL, 2022


import pandas as pd
import numpy as np
import GenoNet_main as genoNet
import GenoNet_Load as genoNetLoad
from sklearn.model_selection import train_test_split

import sparsitySearch as sps





'''Define experiment settings'''
expName = 'trait3_3k_6em3' #for tensorboard and resulting array name
filePath = 'netData/genoNet_pheno1_ref.ckpt' #Define the network for saving/loading
trait=1 # Choose trait 1,2,3
trainFlag=True #Otherwise only evaluation with trained network
lVal = 1e-3      #Initial learning rate, should not be changed
alpha = 6e-3     #sparsity level, found via sparsity search
maxIter=int(3001) #Training iterations


#Call dataset

#read the genotypes
data = pd.read_csv('genData/rice_geno_mat.txt')
total_geno=data.values
n,p=total_geno.shape

#read the pheno
data_pheno = pd.read_csv('genData/tot_pheno.txt')
total_pheno=data_pheno.values[:,trait]
total_pheno=np.array(total_pheno).reshape((n,1))




#perform sparsity Search
if(False):
    
    cv=5 #Choose one reference data split to find suitable sparsity
    X_train, X_test, y_train, y_test= train_test_split(total_geno, total_pheno, test_size=0.2,random_state=cv)
    dataGeno = genoNetLoad.read_data_sets(X_train, X_test, y_train, y_test)
    alpha = sps.sparsitySearch(dataGeno,maxIter=maxIter)



arr = []

for cv in range(0, 50):
    sess = genoNet.initNet()
    expNameCur = expName + str(cv)
    X_train, X_test, y_train, y_test= train_test_split(total_geno, total_pheno, test_size=0.2,random_state=cv)
    bSize = X_test.shape
    print (X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    print (cv)

    dataGeno = genoNetLoad.read_data_sets(X_train, X_test, y_train, y_test)
    
    [output,loss,corr] = genoNet.callNet(sess,dataGeno,filePath,expNameCur,
            lVal=lVal,        #learning rate
            alpha=alpha,       #sparsity level
            bSize=bSize[0],    #Batch size = full batch for test data
            maxIter=maxIter,   #training iterations/optimisation steps
            tfBoardFlag=False,
            trainFlag=True,
            evalFlag=False)
    arr.append(corr)
    

    print(str(corr))
    print('Current Average: ' + str(sum(arr) / float(len(arr))))
    
    
    genoNet.closeNet(sess)
    
#Get the mean



avgCorr = sum(arr) / float(len(arr))

np.save('results/' + expName + '_CV',arr)

