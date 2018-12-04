'''
autodidact.ai (Flytxt, Indian Institute of Technology Delhi, CSIR-CEERI)
AutoGBT: "Automatically Optimized Gradient Boosting Trees for Classifying Large
          Volume High Cardinality Data Streams under Concept Drift"

Team:   Jobin Wilson (jobin.wilson@flytxt.com)
        Amit Kumar Meher (amit.meher@flytxt.com)
        Bivin Vinodkumar Bindu (bivin.vinod@flytxt.com)
        Manoj Sharma (mksnith@gmail.com)
        Vishakha Pareek (vishakhapareek@ceeri.res.in)
             
'''
import pickle
import data_converter
import numpy as np  
from os.path import isfile
import random
import time
import os


class setupmgr:
    """
    A simple class to manage installation of dependent libraries in 
    the docker image at run time
    """
    @staticmethod    
    def pip_install(component):
        command = "pip3 install "+component
        print("AutoGBT[setupmgr]:pip3 command:",command)
        os.system(command)
#################################################################    
        
class Model:
    def __init__(self,datainfo,timeinfo):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        # Just print some info from the datainfo variable
        print("The Budget for this data set is: %d seconds" %datainfo['time_budget'],'name:',datainfo['name'])        
        print("Loaded %d time features, %d numerical features, %d categorical " \
              "features and %d multi valued categorical variables" \
              %(datainfo['loaded_feat_types'][0], datainfo['loaded_feat_types'][1],\
                datainfo['loaded_feat_types'][2],datainfo['loaded_feat_types'][3]))
        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]
        print("[***] Overall time spent %5.2f sec" % overall_spenttime)        
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)        
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        ### install hyperopt and lightgbm ###
        print("AutoGBT[Model]:installing hyperopt and lightgbm...")
        setupmgr.pip_install("hyperopt")
        setupmgr.pip_install("lightgbm")
       
        import StreamProcessor
        self.mdl = StreamProcessor.StreamSaveRetrainPredictor()
        
        
        
    def fit(self, F, y, datainfo,timeinfo):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        '''
      
        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)        
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)  
       
        print('AutoGBT[Model]:Calling StreamProcessor - partial_fit')
        self.mdl.partial_fit(F,y,datainfo,timeinfo)
       
    def predict(self, F,datainfo,timeinfo):
        '''
        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. 
        The function predict eventually returns probabilities or continuous values.
        '''
   
        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)        
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)        
        
        print('AutoGBT[Model]:Calling StreamProcessor - predict')
        return self.mdl.predict(F,datainfo,timeinfo)
      

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("AutoGBT[Model]:Model reloaded from: " + modelfile)
        return self
