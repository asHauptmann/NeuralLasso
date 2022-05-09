# Learned lasso-type network for genetic data
# Andreas Hauptmann, University of Oulu/UCL, 2022

import tensorflow.compat.v1 as tf
import numpy as np
import os
from os.path import exists
import shutil


#import matplotlib.pyplot as plt

tf.disable_eager_execution()
TF_BOARD_DIR = '/scratch1/tensorboard/genoNetAmy/'

name = os.path.splitext(os.path.basename(__file__))[0]
#bSize=int(79)
N=int(96)
NN=N*N
#maxIter=int(20001)   #Iterations for optimisation

#alpha= 5e-3          # weighting value for sparsity
#lVal=2e-3            # Start learning rate (will be halved in first step)
#decayInterval = 900 # intervals in which lVal will be halved (for convergence)



#Create directory for tensorboard
def default_tensorboard_dir(name):
    tensorboard_dir = TF_BOARD_DIR
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return tensorboard_dir

 
#Logging file for tensorbaord
def summary_writers(name, expName ,cleanup=False, session=None):
    if session is None:
        session = tf.get_default_session()
    
    dname = default_tensorboard_dir(name)
    
    if cleanup and os.path.exists(dname):
        shutil.rmtree(dname)    
    
    test_summary_writer = tf.summary.FileWriter(dname + '/test_' + expName, session.graph)
    train_summary_writer = tf.summary.FileWriter(dname + '/train_' + expName)   
    
    return test_summary_writer, train_summary_writer    




# Compute PSNR
def psnr(x_result, x_pheno, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_pheno) - tf.reduce_min(x_pheno)
        mse = tf.reduce_mean((x_result - x_pheno) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

# Compute correlation coefficient
def correlation(x_result, x_pheno):
    upFrac = tf.reduce_sum( (x_pheno- tf.reduce_mean(x_pheno))*(x_result-tf.reduce_mean(x_result)) )
    doFrac = tf.sqrt( tf.reduce_sum( ((x_pheno-tf.reduce_mean(x_pheno))**2) ) * tf.reduce_sum(((x_result-tf.reduce_mean(x_result))**2) ) )
    return upFrac/doFrac


''' genoNet main function, here all learnable parameters are defined and the model A_theta is computed'''
def linearLayer(x_in,bSize,N,layNum):

    
    zero=tf.constant(0.0, shape=[bSize,1])
    x_full=tf.reshape(x_in ,[bSize,N])
    
    # Diagonal vectors to be learned
    diagFac = getKappa(0.1, [N],'diagFac_' + str(layNum))
    rightFac = getKappa(0.1, [N],'rightFac_' + str(layNum))
    leftFac = getKappa(0.1, [N],'leftFac_' + str(layNum))
    right2Fac = getKappa(0.0, [N],'rightFac_' + str(layNum))
    left2Fac = getKappa(0.0, [N],'leftFac_' + str(layNum))
    
    # Bias to be learned
    addValDiag = getKappa(0.0, [N],'diagFac_' + str(layNum))
    addValLeft = getKappa(0.0, [N],'diagFac_' + str(layNum))
    addValRight = getKappa(0.0, [N],'diagFac_' + str(layNum))
    addValLeft2 = getKappa(0.0, [N],'diagFac_' + str(layNum))
    addValRight2 = getKappa(0.0, [N],'diagFac_' + str(layNum))
    
    
    
    # shift, multiplication (sub-diagonals) and nonlinearities with bias
    '''Implementation of the main operation for all components: phi(c * tau(x) + b) (not very elegant, could be done in a loop)''' 

    xLeft = tf.concat([zero,x_full[:,0:N-1]],axis=1)
    xLeft = tf.nn.relu(tf.multiply(leftFac,xLeft)+addValLeft)
    
    xRight = tf.concat([x_full[:,1:N],zero],axis=1)   
    xRight = tf.nn.relu(tf.multiply(rightFac,xRight)+addValRight)
    
    xLeft2 = tf.concat([zero,zero,x_full[:,0:N-2]],axis=1)    
    xLeft2 = (tf.multiply(left2Fac,xLeft2)+addValLeft2)
    
    xRight2 = tf.concat([x_full[:,2:N],zero,zero],axis=1)   
    xRight2 = (tf.multiply(right2Fac,xRight2)+addValRight2)
    
    xDiag  = tf.nn.relu(tf.multiply(diagFac,x_full)+addValDiag)
    
    #Sum and output
    x_update = xDiag + xRight + xLeft  + xRight2 + xLeft2 
    x_out = (x_update)
    
    #l1-norm of learned coefficients
    l1ofVars = tf.reduce_sum(tf.abs(diagFac))+tf.reduce_sum(tf.abs(rightFac))+tf.reduce_sum(tf.abs(leftFac))+tf.reduce_sum(tf.abs(right2Fac))+tf.reduce_sum(tf.abs(left2Fac))
    l1ofVars = l1ofVars + tf.reduce_sum(tf.abs(addValDiag))+tf.reduce_sum(tf.abs(addValLeft))+tf.reduce_sum(tf.abs(addValRight))+tf.reduce_sum(tf.abs(addValLeft2))+tf.reduce_sum(tf.abs(addValRight2))

    
    return x_out, l1ofVars

#Initialisation of learnable parameter
def getKappa(inVal,shape,varName):
    kappa = tf.constant(inVal, shape=shape)
    return tf.Variable(kappa, name=varName)



def initNet():
    sess = tf.InteractiveSession()    
         
    print('--------------------> GenoNet Init <--------------------')        
    return sess
    
'''Main function to be called with parameters'''
def callNet(sess,dataGeno,netPath,expName,
            lVal = 2e-3,
            alpha = 5e-3,
            bSize = 60,
            maxIter=int(5001),
            tfBoardFlag=False,
            trainFlag=False,
            evalFlag=True,
            printFlag = False):
    print('--------------------> GenoNet Train <--------------------')        
    
    
    # Load data  
    inputSize=dataGeno.train.geno.shape
    outSize=dataGeno.train.pheno.shape
         
    # Placeholder for data
    geno = tf.placeholder(tf.float32, [None, inputSize[1]])
    pheno = tf.placeholder(tf.float32, [None, outSize[1]])
        
        
    # Calling the network
    with tf.name_scope('GenoNet'):
          
       x_update=tf.reshape(geno,[bSize,inputSize[1]])
       x_update, l1Vars = linearLayer(x_update,bSize,inputSize[1],0)
       y_diff = tf.reduce_sum(x_update,axis=1,keepdims=True)
            
    #Define optimiser
    with tf.name_scope('optimizer'):
       corr = correlation(y_diff, pheno) 
       loss = tf.norm(tf.subtract(pheno,y_diff))/float(bSize)
       learningRate=tf.constant(1e-3)
       train_step = tf.train.AdamOptimizer(learningRate).minimize(loss+alpha*l1Vars)
    
    if(tfBoardFlag):
        #Summaries for tensorboard    
        with tf.name_scope('summaries'):
             tf.summary.scalar('loss', loss)
             tf.summary.scalar('psnr', psnr(y_diff, pheno))
             tf.summary.scalar('Corr', correlation(y_diff, pheno))
             tf.summary.scalar('l1Vars', l1Vars)
                
             merged_summary = tf.summary.merge_all()
    #         expName='GenoNet_' + fileOutName
             test_summary_writer, train_summary_writer = summary_writers(name, expName ,cleanup=False)        
            
        
        
        
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    lValInit = lVal

    feed_test={geno: dataGeno.test.geno[0:bSize],
                             pheno: dataGeno.test.pheno[0:bSize]}
    
    if(trainFlag):    
        #start training
        startIt=0
        for i in range(maxIter):
                  
              batch = dataGeno.train.next_batch(bSize)
              feed_train={geno: batch[0], pheno: batch[1], learningRate: lVal}
              if(tfBoardFlag):                  
                  _, merged_summary_result_train = sess.run([train_step, merged_summary],
                                              feed_dict=feed_train)
              else:    
                  sess.run([train_step], feed_dict=feed_train)
                  
              if i % 10 == 0:
                  lVal= 1e-5 + 0.5*( np.cos( i *  np.pi/maxIter )+1)*lValInit
              
              # Test data and summary for tensorboard  
              if(printFlag and i % 10 == 0):
              
             
                it=i+startIt

                if(tfBoardFlag):    
                    loss_result, merged_summary_result,corr_result = sess.run([loss, merged_summary,corr],
                                      feed_dict=feed_test)
            
                    if loss_result < 100:
                        train_summary_writer.add_summary(merged_summary_result_train, it)
                        test_summary_writer.add_summary(merged_summary_result, it)
                else: 
                    loss_result,corr_result= sess.run([loss,corr], feed_dict=feed_test)
            
                print('iter={}, loss={}, corr={}, lVal={}'.format(i,loss_result,corr_result,lVal))
#                if corr_result > maxLoss:
#                    maxLoss = corr_result
        
           
        loss_result, corr_result, pheno_predict = sess.run([loss,corr, y_diff],feed_dict=feed_test)        
        
        #save_path = saver.save(sess, netPath)
        #print("Model saved in file: %s" % save_path)
                
        print('--------------------> DONE <--------------------')
                
    
    if(evalFlag):
        
        saver.restore(sess, netPath)
        feed_test={geno: dataGeno.test.geno,
                             pheno: dataGeno.test.pheno}
        loss_result, corr_result, pheno_predict = sess.run([loss,corr, y_diff],feed_dict=feed_test)        
    
        print('Sample processed')
      
#    tf.reset_default_graph()  
#    sess.close()    
    
    return pheno_predict,loss_result, corr_result
    
def closeNet(sess):
    
    
    tf.reset_default_graph()  
    sess.close()    
    
    