'''
Module :  StreamProcessor
Authors:  Jobin Wilson (jobin.wilson@flytxt.com)
          Amit Kumar Meher (amit.meher@flytxt.com)
          Bivin Vinodkumar Bindu (bivin.vinod@flytxt.com)
'''

import pandas as pd
from functools import wraps
import time as time
import numpy as np
import random
from libscores import *

class Utils:
    """
    Generic utility functions that our model would require
    """
    @staticmethod
    def random_sample_in_order(X,y,removeperc,seed=1):
        if removeperc==0:
            return X,y
        num_train_samples = len(X)
        rem_samples=int(num_train_samples*removeperc)
        np.random.seed(seed)
        skip = sorted(random.sample(range(num_train_samples),num_train_samples-rem_samples))
        print('AutoGBT[Utils]:Random sample length:',num_train_samples-rem_samples)
        return X[skip,:],y[skip,:]

 
    """
    A function to perform majority downsampling. in case of class-imbalance, 
    pick all examples from minority class and include random samples from 
    majority class to make it balanced at a specific ratio

    """
    @staticmethod
    def majority_undersample(X,y,frac=1.0,seed=1):
        MINORITY_THRESHOLD = 20000
        ## warn if too many samples are present
        y=y.reshape(len(y))
        class_0_freq = len(y[y==0])
        class_1_freq = len(y[y==1])
        majority_class = 0
        if class_1_freq>class_0_freq:
            majority_class = 1
            minority_count = class_0_freq
        else:
            minority_count = class_1_freq

        minority_class = int(not majority_class)

        if  minority_count > MINORITY_THRESHOLD:
            print('AutoGBT[Utils]:Minority samples exceed threshold=',\
                  MINORITY_THRESHOLD,'total minority samples=',minority_count)
        ### do downsampling as per remove percent ###

        indices = np.array(range(len(y)))
        majority_ind = indices[y==majority_class]
        minority_index = indices[y==minority_class]
        np.random.seed(seed)
        if int(minority_count*frac) > len(majority_ind):
            size = len(majority_ind)
        else:
            size = int(minority_count*frac)
        majority_index = np.random.choice(indices[y==majority_class],size=size,replace=False)
        sorted_index = sorted(np.concatenate([minority_index,majority_index]))

        print('AutoGBT[Utils]:Sampled data size:',len(sorted_index))
        return X[sorted_index],y[sorted_index]

    

def simple_time_tracker(log_fun):
    def _simple_time_tracker(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time.time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time.time() - start_time

                # log the result
                log_fun({
                    'function_name': fn.__name__,
                    'total_time': elapsed_time,
                })

            return result

        return wrapped_fn
    return _simple_time_tracker

def _log(message):
    print('[SimpleTimeTracker] {function_name} {total_time:.3f}'.format(**message))

from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, STATUS_OK, Trials
from hyperopt.fmin import fmin
from hyperopt import space_eval
import lightgbm as lgbm

class AutoHyperOptimizer:
    """
    A wrapper for hyperopt to automatically tune hyper-parameters of our model 
    Idea : We use basic SMBO to get to best hyper parameters using a 
    directed search near the neighborhood of a fixed set of hyper-parameters.
    A search window is defined for each hyper-parameter considering the nature
    of the hyper-parameter.Each set of hyper-parameters is eavluated in a cross-validation
    setting on a small fraction of data to determine the fitness. Hyperopt attempts
    to find hyper-parameters that minimize (1.0-AUC) on the validation data.
    
    We finallty compare the cross-validation AUC of the model trained with 
    fixed hyper-parameter set with the AUC of the model trained using hyper-parameters
    returned by hyperopt, and choose the one with higher AUC as the optimal hyper-parameter
    set.
    """

    def __init__(self,max_samples=50000,max_evaluations=25,seed=1,parameter_space={}):
        self.max_samples = max_samples
        self.max_evaluations = max_evaluations
        self.test_size = 0.25 ## fraction of data used for internal validation
        self.shuffle = False
        self.best_params = {}
        self.seed = seed
        self.param_space = parameter_space
  

    def gbc_objective(self,space):

        print('AutoGBT[AutoHyperOptimizer]:Parameter space:',space)
        model = lgbm.LGBMClassifier(random_state=self.seed,min_data=1, min_data_in_bin=1)
        model.set_params(**space)
        model.fit(self.Xe_train,self.ys_train)
        mypreds = model.predict_proba(self.Xe_test)[:,1]
        auc = auc_metric(self.ys_test.reshape(-1,1),mypreds.reshape(-1,1))
        print('AutoGBT[AutoHyperOptimizer] auc=',auc)
        return{'loss': (1-auc), 'status': STATUS_OK }

    def fit(self,X,y,indicator): 
        '''
        indicator=1 means we intend to do just sampling and one-time fitting
        for evaluating a fixed set of hyper-parameters, 
        0 means run hyperopt to search in the neighborhood of the seed 
        hyper-parameters to see if model quality is improving.
        '''
        num_samples = len(X)
        print('AutoGBT[AutoHyperOptimizer]:Total samples passed for'\
              'hyperparameter tuning:',num_samples)
        if num_samples>self.max_samples:
            removeperc = 1.0 - (float(self.max_samples)/num_samples)
            print ('AutoGBT[AutoHyperOptimizer]:Need to downsample for managing time:,'\
                   'I will remove data percentage',removeperc)
            XFull,yFull = Utils.random_sample_in_order(X,y.reshape(-1,1),removeperc)
            print('AutoGBT[AutoHyperOptimizer]:downsampled data length',len(XFull))
        else:
            XFull = X
            yFull = y

        self.Xe_train, self.Xe_test, self.ys_train, self.ys_test = \
        train_test_split(XFull, yFull.ravel(),test_size = self.test_size, random_state=self.seed,shuffle=True)
 
        if indicator == 1: 
            ## just fit lightgbm once to obtain the AUC w.r.t a fixed set of hyper-parameters ##
            model = lgbm.LGBMClassifier(random_state=self.seed,min_data=1, min_data_in_bin=1)
            model.set_params(**self.param_space) 
            model.fit(self.Xe_train,self.ys_train)
            mypreds = model.predict_proba(self.Xe_test)[:,1]
            auc = auc_metric(self.ys_test.reshape(-1,1),mypreds.reshape(-1,1))
            return auc
        else:
            trials = Trials()
            best = fmin(fn=self.gbc_objective,space=self.param_space,algo=tpe.suggest,trials=trials,max_evals=self.max_evaluations)
            params = space_eval(self.param_space, best)
            print('AutoGBT[AutoHyperOptimizer]:Best hyper-parameters',params)
            self.best_params = params
            return params, 1-np.min([x['loss'] for x in trials.results]) 
            #return the best hyper-param with the corresponding AUC


from collections import Counter

class GenericStreamPreprocessor:
    """
    Our generic pre-processing pipeline that uses frequency encoder idea. Categorical and
    Multi-categorical features are encoded with their running frequencies
    Pipeline also handlees Datetime columns. Min non-zero value in such columns would
    be subtracted to make comparison meaningful. Additional derived features  
    (e.g. day of week, time of day etc.) are also generated from such columns
    """

    def __init__(self):
        self.categorical_cols=[]
        self.date_cols=[]
        self.redundant_catcols = []
        self.ohe_cols = []
        self.frequency_encode = True
        self.date_encode = True
        self.colMins = {}
        self.featureMap = {}
        self.dateMap = {}
        self.ohe_col_threshold = 30
        ## no. of unique values to decide if the column need tobe one-hot encoded ##
        ## we didnt finally use OHE as it didnt appear to generalize our pipeline well ##
        self.rows_processed = 0
        self.freqMap = {}
       

    def set_date_cols(self,cols):
        self.date_cols = cols
        for col in cols:
            self.dateMap[col]=0.0


    def set_categorical_cols(self,cols):
        self.categorical_cols = cols
        for col in cols:
            self.featureMap[col]={}

    def set_frequency_encode(self,flag=True):
        self.frequency_encode = flag

    def set_date_encode(self,flag=True):
        self.date_encode = flag


    def set_ohe_col_threshold(self,threshold=30):
        self.ohe_col_threshold = threshold

    def get_ohe_col_threshold(self):
        return self.ohe_col_threshold


    def print_config(self):
        print ('AutoGBT[GenericStreamPreprocessor]:date encoding:',\
               self.date_encode,'columns=',self.date_cols)
        print ('AutoGBT[GenericStreamPreprocessor]:frequency encoding:',\
               self.frequency_encode,'columns=',self.categorical_cols)
   
    @simple_time_tracker(_log)
    def partial_fit(self,X):
        """
        Update frequency count of all categorical/multi-categorical values
        Maintain a map of minimum values in each date column for encoding
        """
        for col in range(X.shape[1]):
            if col in self.categorical_cols and self.frequency_encode ==True :
                if X.shape[0] > 200000: 
                    ## count using pandas if it is a large dataset
                    curr_featureMap = dict(pd.value_counts(X[:,col]))
                    self.featureMap[col] = dict(Counter(self.featureMap[col]) + Counter(curr_featureMap))
                    print('AutoGBT[GenericStreamPreprocessor]:using pandas count ' \
                          'for faster results..updating feature count map for column:',col)
                else:
                    val,freq = np.unique(X[:,col],return_counts=True)
                    curr_featureMap = dict(zip(val,freq))
                    self.featureMap[col] = dict(Counter(self.featureMap[col]) + Counter(curr_featureMap))
                    print('AutoGBT[GenericStreamPreprocessor]:using numpy unique count..'\
                          'updating feature count map for column:',col, len(self.featureMap[col]))

            elif col in self.date_cols and self.date_encode == True:    
                ## find minimum non-zero value corresponding to each date columns ##
                date_col = X[:,col].astype(float)
                non_zero_idx = np.nonzero(date_col)[0]
                if(len(non_zero_idx) > 0):
                    if self.dateMap[col]==0:
                        self.dateMap[col] = np.min(date_col[non_zero_idx])
                    else:
                        self.dateMap[col] = np.min([self.dateMap[col],np.min(date_col[non_zero_idx])])

        self.rows_processed = self.rows_processed + len(X)
        print('AutoGBT[GenericStreamPreprocessor]:featuremap size:',len(self.featureMap))

    @simple_time_tracker(_log)
    def prepareFrequencyEncodingMap(self):

        for col in self.categorical_cols:
            keys = self.featureMap[col].keys()
            vals = np.array(list(self.featureMap[col].values())).astype(float)
            self.freqMap[col] = dict(zip(keys,vals))


    @simple_time_tracker(_log)
    def transform(self,X):
        result = []
        for col in range(X.shape[1]):
            if col in self.categorical_cols:
                ### DO FREQUENCY ENCODING ####
                freq_encoded_col = np.vectorize(self.freqMap[col].get)(X[:,col])
                result.append(freq_encoded_col)

            elif col in self.date_cols:
                transformed_date_col = X[:,col].astype(float) - self.dateMap[col]
                result.append(transformed_date_col)
            else: ### it must be a numeric feature
                result.append(X[:,col])

        ### add dynamic date difference features and other generated features ###
        for i in range(len(self.date_cols)):
            for j in range(i+1,len(self.date_cols)):
                if len(np.nonzero(X[:,i]))>0 and len(np.nonzero(X[:,j]))>0:
                    print('AutoGBT[GenericStreamPreprocessor]:datediff from nonzero cols:',i,j)
                    result.append(X[:,i]-X[:,j])

            dates = pd.DatetimeIndex(X[:,i])
            ## get the date column
            dayofweek = dates.dayofweek.values
            dayofyear = dates.dayofyear.values
            month = dates.month.values
            weekofyear = dates.weekofyear.values
            day = dates.day.values
            hour = dates.hour.values
            minute = dates.minute.values
            year = dates.year.values

            result.append(dayofweek)
            result.append(dayofyear)
            result.append(month)
            result.append(weekofyear)
            result.append(year)
            result.append(day)
            result.append(hour)
            result.append(minute)

        return np.array(result).T

    def get_ohe_candidate_columns(self):
        ohe_candidates = []
        for col in self.categorical_cols:
            unique_categories = len(self.featureMap[col])
            if  unique_categories>1 and unique_categories <= self.ohe_col_threshold:
                ohe_candidates.append(col)
        return ohe_candidates

class StreamSaveRetrainPredictor:
    """
    A Save-Retrain model to combat concept-drift using a two level sampling strategy,
    and by using a generic stream processing pipeline.
    
    Idea: for each incoming batch of data along with label, do a majority undersampling
    and maintain the raw data in a buffer (level-1 sampling strategy). Model training is 
    performed using a lazy strategy (just prior to making predictions) subject to
    the availability of time budget. This way, most recent data is utilized by
    the pre-processing pipeline in performing frequency encoding, datetime column normalization
    etc., to minimze the effect of changes in the underlying data distribution. Automatic
    hyper-parameter tuning is performed using hyperopt SMBO when the very first batch
    of data is encountered. For large datasets,a level-2 downsampling strategty is applied on 
    accumulated training set to keep model training time within the budget.
       
    """
    def __init__(self):
        self.batch=0
        self.max_train_data=400000
        self.min_train_per_batch = 5000
        self.clf=''
        self.best_hyperparams = {}
        self.stream_processor = GenericStreamPreprocessor()
        self.XFull = []
        self.yFull = []
        self.ohe = None
        self.ohe_cols = None
        ### if 80% time budget on a dataset is already spent, donot refit the model - just predict with the existing model###
        self.dataset_budget_threshold = 0.8
        ### Set the delta region for parameter exploration for hyperopt ###
        ### Explore in a small window of hyper-parameters nearby to see if model quality improves ###
        self.delta_n_estimators = 50
        self.delta_learning_rate = 0.005
        self.delta_max_depth = 1
        self.delta_feature_fraction = 0.1
        self.delta_bagging_fraction = 0.1
        self.delta_bagging_freq = 1
        self.delta_num_leaves = 20
        self.current_train_X = {} 
        self.current_train_y = []  
        ## max number of function evaluation for hyperopt ##
        self.max_evaluation = 30    

    def partial_fit(self,F,y,datainfo,timeinfo):
      
        self.current_train_X = F
        self.current_train_y = y

        date_cols = datainfo['loaded_feat_types'][0]
        numeric_cols = datainfo['loaded_feat_types'][1]
        categorical_cols = datainfo['loaded_feat_types'][2]
        multicategorical_cols = datainfo['loaded_feat_types'][3]
        ## date time coulumn indices ###  
        time_cols = np.arange(0,date_cols) 
        ## categorical and multi-categorical column indices ###
        cols = np.arange(date_cols+numeric_cols,date_cols+numeric_cols+categorical_cols+multicategorical_cols)
        print('AutoGBT[StreamSaveRetrainPredictor]:date-time columns:',time_cols)
        print('AutoGBT[StreamSaveRetrainPredictor]:categorical columns:',cols)
        ### extract numerical features first  
        X=F['numerical'] 
        ### replace missing values with zeros ###
        X = np.nan_to_num(X)
        print('AutoGBT[StreamSaveRetrainPredictor]:Numeric Only data shape:',X.shape)
        if categorical_cols > 0: 
           ### replace missing values with string 'nan' ###
           CAT = F['CAT'].fillna('nan').values
           X = np.concatenate((X,CAT),axis=1)
           ## append categorical features
           del CAT

        if multicategorical_cols > 0: 
           ### replace missing values with string 'nan' ###
           MV = F['MV'].fillna('nan').values
           X = np.concatenate((X,MV),axis=1)
           ### append multi-categorical features ###
           del MV


        print('AutoGBT[StreamSaveRetrainPredictor]:Feature Matrix Shape:',X.shape)

        ### INITIALIZE OUR STREAM PROCESSOR PIPELINE###
        if len(self.stream_processor.categorical_cols)==0:
            print('AutoGBT[StreamSaveRetrainPredictor]:initializing categorical columns:')
            self.stream_processor.set_categorical_cols(cols)
            
        if len(self.stream_processor.date_cols)==0:
            print('AutoGBT[StreamSaveRetrainPredictor]:initializing date-time columns:')
            self.stream_processor.set_date_cols(time_cols)
        #### END INITIALIZATION ###
        
        if self.stream_processor.rows_processed == 0: 
            ### we are seeing the first batch of data; process it to make frequency encoder ready ###
            self.stream_processor.partial_fit(X)
            print('AutoGBT[StreamSaveRetrainPredictor]:partial fit of X for first time..')

        train_X,train_y = Utils.majority_undersample(X,y,frac=3.0) 
        ### level-1 of our sampling strategy - sample 1:3 always to handle skewed data ##
        print('AutoGBT[StreamSaveRetrainPredictor]:Level-1 Sampling: undersampling and '\
              'saving raw data for training:length=',len(train_X))

        self.batch = self.batch + 1.0

        if len(self.XFull) == 0: 
            ### first time
            self.XFull = train_X
            self.yFull = train_y

        else: 
            ## we have history, so concatenate to it ##
            self.XFull=np.concatenate((self.XFull,train_X),axis=0)
            self.yFull=np.concatenate((self.yFull,train_y),axis=0)

        num_train_samples = len(self.XFull)
        print('AutoGBT[StreamSaveRetrainPredictor]:Total accumulated training '\
              'data in raw form:',num_train_samples)


    def predict(self,F,datainfo,timeinfo):
        ### extract numerical data first
        X=F['numerical']
        ## replace nan to 0 ##
        X = np.nan_to_num(X) 

        date_cols = datainfo['loaded_feat_types'][0]
        numeric_cols = datainfo['loaded_feat_types'][1]
        categorical_cols = datainfo['loaded_feat_types'][2]
        multicategorical_cols = datainfo['loaded_feat_types'][3]

        if categorical_cols >0:
        ### replace missing values with string 'nan' ###
           CAT = F['CAT'].fillna('nan').values
        ## append categorical features  
           X = np.concatenate((X,CAT),axis=1)
           del CAT

        if multicategorical_cols > 0:
        ### replace missing values with string 'nan' ###
           MV =  F['MV'].fillna('nan').values
        ### append multi-categorical features     
           X = np.concatenate((X,MV),axis=1)
           del MV

        dataset_spenttime=time.time()-timeinfo[1]
        print('AutoGBT[StreamSaveRetrainPredictor]:Dataset Budget threshhold:',self.dataset_budget_threshold ,'safe limit =', \
              datainfo['time_budget']*self.dataset_budget_threshold)
        ## a safe limit for time budget is calculated ## 
        if dataset_spenttime < datainfo['time_budget']*self.dataset_budget_threshold: 
            ### if sufficient time budget exist considering the safe limit, then continue model update ###
            print('AutoGBT[StreamSaveRetrainPredictor]:Sufficient budget available to update the model')
            ### update the stream processor with new data ###
            self.stream_processor.partial_fit(X)
            print('AutoGBT[StreamSaveRetrainPredictor]:partial fit of X in predict function..total rows processed:',self.stream_processor.rows_processed)
            self.stream_processor.prepareFrequencyEncodingMap()
            print('AutoGBT[StreamSaveRetrainPredictor]:FrequencyEncoding Map Prepared')
            num_train_samples = len(self.XFull)
            print('AutoGBT[StreamSaveRetrainPredictor]:About to transform full training data:',num_train_samples)
    
            XTrain = []
            yTrain = []
            if num_train_samples>self.max_train_data:
                removeperc = 1.0 - (float(self.max_train_data)/num_train_samples)
                print('AutoGBT[StreamSaveRetrainPredictor]:Level-2 Sampling...'\
                      'Too much training data..I need to subsample:remove',removeperc)
                XTrain,yTrain = Utils.random_sample_in_order(self.XFull,self.yFull.reshape(-1,1),removeperc)
                print('AutoGBT[StreamSaveRetrainPredictor]:downsampled training data length=',len(XTrain))
            else:
                XTrain = self.XFull
                yTrain = self.yFull
    
            XTrain_transformed = self.stream_processor.transform(XTrain)
            print('AutoGBT[StreamSaveRetrainPredictor]:Training transformed shape:',XTrain_transformed.shape)
            
            ### we didnt find the best hyper-parameters yet
            if len(self.best_hyperparams)==0: 
            #Evaluate at run-time 2 promising choices for Hyper-parameters:
            #Choice1->Fixed set of hyper-parameters, Choice2-> promising solution near a fixed set, found using hyperopt 
                param_choice_fixed = {'n_estimators':600,\
                                      'learning_rate':0.01,\
                                      'num_leaves':60,\
                                      'feature_fraction':0.6,\
                                      'bagging_fraction':0.6,\
                                      'bagging_freq':2,\
                                      'boosting_type':'gbdt',\
                                      'objective':'binary',\
                                      'metric':'auc'}
                
                #Get the AUC for the fixed hyperparameter on the internal validation set
                autohyper = AutoHyperOptimizer(parameter_space=param_choice_fixed)
                best_score_choice1 = autohyper.fit(XTrain_transformed,yTrain.ravel(),1)
                print("---------------------------------------------------------------------------------------------------")
                print("AutoGBT[StreamSaveRetrainPredictor]:Fixed hyperparameters:",param_choice_fixed)
                print("AutoGBT[StreamSaveRetrainPredictor]:Best scores obtained from Fixed hyperparameter only is:",best_score_choice1)
                print("---------------------------------------------------------------------------------------------------")
                
                #Get the AUC for the fixed hyperparameter+Hyperopt combination on the internal validation set
                #Step:1-Define the search space for Hyperopt to be a small delta region over the initial set of fixed hyperparameters 
                n_estimators_low = 50 if (param_choice_fixed['n_estimators'] - self.delta_n_estimators)<50 else param_choice_fixed['n_estimators'] - self.delta_n_estimators
                n_estimators_high = param_choice_fixed['n_estimators'] + self.delta_n_estimators
                
                learning_rate_low = np.log(0.001) if (param_choice_fixed['learning_rate'] - self.delta_learning_rate)<0.001 else np.log(param_choice_fixed['learning_rate'] - self.delta_learning_rate)
                learning_rate_high = np.log(param_choice_fixed['learning_rate'] + self.delta_learning_rate)
                
                num_leaves_low = 5 if (param_choice_fixed['num_leaves'] - self.delta_num_leaves)<5 else param_choice_fixed['num_leaves'] - self.delta_num_leaves
                num_leaves_high = param_choice_fixed['num_leaves'] + self.delta_num_leaves
                
                feature_fraction_low = np.log(0.05) if (param_choice_fixed['feature_fraction'] - self.delta_feature_fraction)<0.05 else np.log(param_choice_fixed['feature_fraction'] - self.delta_feature_fraction)
                feature_fraction_high = np.log(1.0) if (param_choice_fixed['feature_fraction'] + self.delta_feature_fraction)>1.0 else np.log(param_choice_fixed['feature_fraction'] + self.delta_feature_fraction)
                
                bagging_fraction_low = np.log(0.05) if (param_choice_fixed['bagging_fraction'] - self.delta_bagging_fraction)<0.05 else np.log(param_choice_fixed['bagging_fraction'] - self.delta_bagging_fraction)
                bagging_fraction_high = np.log(1.0) if (param_choice_fixed['bagging_fraction'] + self.delta_bagging_fraction)>1.0 else np.log(param_choice_fixed['bagging_fraction'] + self.delta_bagging_fraction)
                
                bagging_freq_low = 1 if (param_choice_fixed['bagging_freq'] - self.delta_bagging_freq)<1 else param_choice_fixed['bagging_freq'] - self.delta_bagging_freq
                bagging_freq_high = param_choice_fixed['bagging_freq'] + self.delta_bagging_freq
                
                boosting_type = param_choice_fixed['boosting_type']
                objective = param_choice_fixed['objective']
                metric = param_choice_fixed['metric']

                #set the search space to be explored by Hyperopt
                param_space_forFixed ={
                'objective': "binary",
                'n_estimators' : hp.choice('n_estimators', np.arange(n_estimators_low, n_estimators_high+50, 50, dtype=int)),
                'num_leaves': hp.choice('num_leaves',np.arange(num_leaves_low, num_leaves_high+10, 10, dtype=int)),
                'feature_fraction': hp.loguniform('feature_fraction', feature_fraction_low, feature_fraction_high),
                'bagging_fraction': hp.loguniform('bagging_fraction', bagging_fraction_low, bagging_fraction_high), 
                'bagging_freq': hp.choice ('bagging_freq',np.arange(bagging_freq_low, bagging_freq_high+1, 1, dtype=int)),
                'learning_rate': hp.loguniform('learning_rate', learning_rate_low, learning_rate_high), 
                'boosting_type' : boosting_type,
                'metric': metric,
                'verbose':-1
                }
                    
                #run Hyperopt to search nearby region in the hope to obtain a better combination of hyper-parameters
                autohyper = AutoHyperOptimizer(max_evaluations=self.max_evaluation, parameter_space=param_space_forFixed) 
                best_hyperparams_choice2, best_score_choice2 = autohyper.fit(XTrain_transformed,yTrain.ravel(),0)
                print("---------------------------------------------------------------------------------------------------")
                print("AutoGBT[StreamSaveRetrainPredictor]:Best hyper-param obtained from Fixed Hyperparameters + Runtime Hyperopt is:",best_hyperparams_choice2)
                print("AutoGBT[StreamSaveRetrainPredictor]:Best score obtained from Fixed Hyperparameter + Runtime Hyperopt is:",best_score_choice2) 
                print("---------------------------------------------------------------------------------------------------")

                #Compare choice-1 & choice-2 and take the better one
                if best_score_choice1 >= best_score_choice2:
                    self.best_hyperparams = param_choice_fixed
                else:
                    self.best_hyperparams = best_hyperparams_choice2

            
            print("AutoGBT[StreamSaveRetrainPredictor]:Best hyper-param obtained is:",self.best_hyperparams)
            #train lightgbm with best hyperparameter obtained     
            self.clf = lgbm.LGBMClassifier(random_state=20,min_data=1, min_data_in_bin=1)
            self.clf.set_params(**self.best_hyperparams) 
            self.clf.fit(XTrain_transformed, yTrain.ravel())
            print('AutoGBT[StreamSaveRetrainPredictor]:LGBM Fit complete on:',XTrain_transformed.shape)

        ## do we need to make prediction in batch mode (chunking) due to memory limit?
        batch_size = 100000
        print('AutoGBT[StreamSaveRetrainPredictor]:predict split size:',X.shape)
        if X.shape[0] <=batch_size: ### if it is relatively small array
            return self.clf.predict_proba(self.stream_processor.transform(X))[:,1]
        else:
            results = np.array([]) ## for chunking results to handle memory limit
            for i in range(0,X.shape[0],batch_size):
                Xsplit = X[i:(i+batch_size),:]
                print('AutoGBT[StreamSaveRetrainPredictor]Chunking Prediction: processing split:'\
                      ,i,i+batch_size,'shape=',Xsplit.shape)
                results = np.append(results,self.clf.predict_proba(self.stream_processor.transform(Xsplit))[:,1])
                del Xsplit
            print("AutoGBT[StreamSaveRetrainPredictor]:RESULTS SHAPE:",np.array(results).shape)
            return np.array(results).T

