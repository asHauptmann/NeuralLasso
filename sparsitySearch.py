# Script to call the GenoNet class file
# Andreas Hauptmann, University of Oulu/UCL, 2022


import GenoNet_main as genoNet

def sparsitySearch(dataGeno,
                   step=2e-3,
                   alphaMid=7e-3,
                   TOL = 3e-4,
                   maxIter=int(20001)):            

    
    expName = 'sparsitySearch' #only for tensorboard
    filePath = 'netData/placeholder.ckpt' #Define the network for saving/loading
    
    
    trainFlag = True
    
    
    
    
    alphaLeft  = alphaMid+step
    alphaRight = alphaMid-step
    
    bSize = dataGeno.test.geno.shape[0]
    
    
    
    sess = genoNet.initNet()
    
    if(trainFlag):
       #Call training, will also evaluate test data
       
       [output,loss,corrLeft] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaLeft,       #sparsity level
                bSize=bSize,
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)
       
    
       [output,loss,corrRight] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaRight,       #sparsity level
                bSize=bSize,
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)
       
       
       [output,loss,corrMid] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaMid,       #sparsity level
                bSize=bSize,
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)
       
      
    while step>TOL:    
        
        corrMax = max(corrLeft,corrRight,corrMid)
    
        if corrMax == corrMid:
            print('Middle: reducing step size')
            step=step/2.0
            alphaLeft  = alphaMid+step
            alphaRight = alphaMid-step
            [output,loss,corrLeft] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaLeft,       #sparsity level
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)
       
    
            [output,loss,corrRight] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaRight,       #sparsity level
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)        
            
            #run left right
        
        elif corrMax == corrLeft:
            print('Move left')
            alphaMid=alphaLeft
            alphaLeft  = alphaMid+step
            alphaRight = alphaMid-step
            
            corrRight=corrMid
            corrMid  = corrLeft
            #Run Left
            [output,loss,corrLeft] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaLeft,       #sparsity level
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)        
            
        elif corrMax == corrRight:
            print('Move right')
            alphaMid=alphaRight
            alphaLeft  = alphaMid+step
            alphaRight = alphaMid-step
            
            corrLeft = corrMid
            corrMid  = corrRight
            
            #run right
            [output,loss,corrRight] = genoNet.callNet(sess,dataGeno,filePath,expName,
                lVal = 1e-3,        #learning rate
                alpha = alphaRight,       #sparsity level
                maxIter=maxIter, #training iterations/optimisation steps
                tfBoardFlag=False,   #for tensorboard logging, if set up change path in GenoNet_v1.py
                trainFlag=trainFlag,
                evalFlag=False)        
        
        
        
    #final Check   
    genoNet.closeNet(sess)
                
    
    #Final check
    corrMax = max(corrLeft,corrRight,corrMid)
    
    if corrMax == corrMid:
        alphaFinal = alphaMid
       
    elif corrMax == corrLeft:
            alphaFinal = alphaLeft
            
    elif corrMax == corrRight:
            alphaFinal = alphaRight
    
    
    
    return alphaFinal