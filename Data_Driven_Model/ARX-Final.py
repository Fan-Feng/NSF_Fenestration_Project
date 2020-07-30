#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
import datetime,time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


workingPath = r"C:\Users\Fan\OneDrive - tongji.edu.cn\Project_in_UA\Fan' file\2_UA_Project\ProjectFenestration\Model\Control_Oriented_Model\data"


# <br id="Read_Data">
# 
# ### read data
# #### Initial Data
# 
# <br>
# Read data from originald data file, which are not record in even time intervel

# In[118]:


data = pd.read_csv(workingPath+"\\One-Month.csv")


# #### Q: are all time interval smaller than 10s??

# In[33]:


def test_max_time_interval(data_Index):
    # This function is used to test if the max time interval is 10
    time_interval = np.array(data_Index[1:])-np.array(data_Index[:-1])
    max_time_interval = max(time_interval)
    print(pd.DataFrame({"data":time_interval}).describe())
    plt.hist(time_interval)
    return max_time_interval,max_time_interval==10


# In[34]:


def test_max_time_interval(data):
    # This function is used to test if the max time interval is 10
    time_interval = np.array(data.iloc[1:,0])-np.array(data.iloc[:-1,0])
    max_time_interval = max(time_interval)
    print(pd.DataFrame({"data":time_interval}).describe())
    plt.hist(time_interval)
    return max_time_interval,max_time_interval==10


# In[37]:


test_max_time_interval(data.iloc[:40276,:])


# ##### Read weather data

# In[38]:


Weather_Data = pd.read_csv(workingPath+"\\WeatherData.csv",index_col=0,parse_dates = True)
Weather_Data.head()


# In[39]:


Weather_Data_resampled = Weather_Data.resample("10s").ffill()
Weather_Data_resampled.head()


# #### data preprocessing: transform the data into evenly sampled

# In[40]:


## re_read data
data_with_Timestamps = pd.read_csv(workingPath+"\\One-Month.csv",parse_dates =[0], 
                                   date_parser=lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(float(x)))
                                   ,index_col=[0])
data_with_Timestamps =data_with_Timestamps.iloc[:40276,:]


# In[41]:


data_with_Timestamps.head()


# In[42]:


## drop duplciates, and back fill nan
data_with_Timestamps = data_with_Timestamps.drop_duplicates().bfill()


# In[43]:


data_with_Timestamps_resampled = data_with_Timestamps.iloc[0:1,:].append(data_with_Timestamps.iloc[1:,:].resample("10s",label ="right").mean())


# In[44]:


data_with_Timestamps_resampled = data_with_Timestamps_resampled.bfill()


# In[45]:


data_with_Timestamps_resampled.head()


# In[46]:


data_with_Timestamps_resampled.tail()


# <br id ='Visualization'>
# 
# ### Visualize the data before constructing models

# In[47]:


from pandas.plotting import autocorrelation_plot


# In[19]:


autocorrelation_plot(data_with_Timestamps_resampled.iloc[:,1])


# #### Clearly, there is strong seasonality

# <br id="Construct_ARX_models">
# 
# ### construct ARX models 
# 
# <br id="Feature_Engineering">
# 
# #### Feature Engineering
# ```flow
#  Feature generation --> Feature Selection --> Model fitting 
# ```

# Feature Generation
#   * Category 1:
#      * History data(Autoregressive term)
#          1. t-10s, t-20s, t-30s, ...,t-60s,
#          2. t-1min, t-2min, t-3min,
#          3. t-1hr, t-12hr,t-24hr
#      * Exogenous variables
#      
# | Items| Type  |
# |:------:|------|
# |1. Outdoor air temperature|numerical|
# |2. Solar radiation|numerical|       
# |3. Occupant| Categorical(binary)|
# |4. Equipment|numerical|
# |5. Lighting|numerical|
# |6. Supply air temperature|numerical|
# |7. Supply air flowrate    |numerical|
# 
#          

# In[48]:


## Some Global Variables
TimeInterval = 10
PredictionHorizon = 24*3600


# ##### Feature Generation

# In[49]:


from statsmodels.tools import add_constant 


# In[50]:


## History_Record data
def History_Record_Data(y_Sequence_Updated,time_lags):    
    result = pd.concat(map(lambda x:y_Sequence_Updated.shift(x),time_lags),axis = 1)
    result.columns = ['hist_{}'.format(x) for x in time_lags]
    return result


# In[51]:


'''
## Some Global Variables
TimeInterval = '10s' # 10 mins
PredictionHorizon = "24H"
time_lags = [1,2,3,4,5,6,12,60,360,4320,8640]
data_with_Timestamps_resampled_TenMin = data_with_Timestamps_resampled.resample('10s').mean()
dataset_Hist = History_Record_Data(data_with_Timestamps_resampled_TenMin.iloc[:,1:2],time_lags)
'''


# In[52]:


def TimeInterval_Parser(TimeInterval):
    unit = TimeInterval[-1]
    if unit.upper() == "S":
        return int(TimeInterval[:-1])
    elif unit.upper() == "T":
        return int(TimeInterval[:-1])*60
    elif unit.upper() == "H":
        return int(TimeInterval[:-1])*3600
    else:
        return False


# In[53]:


def pre_Process(data_with_Timestamps_resampled,Weather_Data_resampled,TimeInterval,time_lags):
    data_with_Timestamps_resampled_After = data_with_Timestamps_resampled.resample(TimeInterval).mean()
    dataset_Hist = History_Record_Data(data_with_Timestamps_resampled_After.iloc[:,1:2],time_lags)

    length = data_with_Timestamps_resampled_After.shape[0]

    ## Occupant
    a = ((data_with_Timestamps_resampled_After.index.hour >=6)+0).reshape(-1,1)
    b  = ((data_with_Timestamps_resampled_After.index.hour<=19)+0).reshape(-1,1)
    occupied_OrNot = pd.DataFrame(a*b,columns = ['occup'],index = data_with_Timestamps_resampled_After.index)
    
    n = int(3600/TimeInterval_Parser(TimeInterval))

    # Equipment and Lighting schedule

    ## This is how Internal heat gain defined in modelica
    define_matrix = [[0,0.05],[8,0.05],[9,0.9],[12,0.9],[12,0.8],[13,0.8],[13,1],[17,1],[19,0.1],[24,0.05]] 
    InternalGain = []
    for i,ele in enumerate(define_matrix[:-1]):
        if (define_matrix[i][0] - define_matrix[i+1][0])==0: #
            continue
        else:
            if (define_matrix[i+1][1] - define_matrix[i][1]) == 0:
                temp = [define_matrix[i+1][1]]*(define_matrix[i+1][0] - define_matrix[i][0])*n
                InternalGain = InternalGain + temp
            else:
                temp = np.arange(define_matrix[i][1],define_matrix[i+1][1],(define_matrix[i+1][1] - define_matrix[i][1])/(define_matrix[i+1][0] - define_matrix[i][0])/n).tolist()
                InternalGain = InternalGain + temp
    plt.plot(InternalGain,)
    plt.title('InternalGain')

    InternGain_All =  pd.DataFrame((InternalGain *(int(length/len(InternalGain))+1))[:length],
                                   columns = ['InternalGain'],index = data_with_Timestamps_resampled_After.index)
    
    # Weather data
    Weather_Data_resampled_After = Weather_Data_resampled.resample(TimeInterval).mean()
    Weather_Data_resampled_After.head()
    

    # prepare dateset

    ## Add Constant
    dataSet_MLR = add_constant(data_with_Timestamps_resampled_After.iloc[:,[0,2]].join(dataset_Hist).join(occupied_OrNot).join(
            InternGain_All).join(Weather_Data_resampled_After))
    
    dataSetX_All,dataSetY_All = dataSet_MLR.iloc[max(time_lags):,:],data_with_Timestamps_resampled_After.iloc[max(time_lags):,1:2]
    return dataSetX_All, dataSetY_All


# In[54]:


## Some Global Variables

## You need to specify these three parameters everytime!!!
TimeInterval = '10T' # 10 mins
PredictionHorizon = "6H"
time_lags = [1,2,3,4,5,6,12,18,36,72,144]


# In[55]:


dataSetX_All,dataSetY_All = pre_Process(data_with_Timestamps_resampled,Weather_Data_resampled,TimeInterval,time_lags)


# In[151]:


dataSetX_All.head()


# In[152]:


dataSetX_All.tail()


# In[25]:


dataSetY_All.head()


# In[56]:


## Visualization
featureSet = ['const', 'TSupCor.T', 'VSupCor_flow.V_flow', 'occup', 'InternalGain',
       'Dry Bulb Temperature {C}', 'Dew Point Temperature {C}',
       'Relative Humidity {%}', 'Atmospheric Pressure {Pa}',
       'Extraterrestrial Horizontal Radiation {Wh/m2}',
       'Extraterrestrial Direct Normal Radiation {Wh/m2}',
       'Horizontal Infrared Radiation Intensity from Sky {Wh/m2}',
       'Global Horizontal Radiation {Wh/m2}',
       'Direct Normal Radiation {Wh/m2}',
       'Diffuse Horizontal Radiation {Wh/m2}']
len(featureSet)


# In[57]:


from scipy.stats import zscore
plt.figure(figsize = [10,45])
for i in range(len(featureSet)):
    plt.subplot(len(featureSet),1,i+1)
    plt.plot(zscore(dataSetX_All.loc[:,featureSet[i]].iloc[:144]))
    plt.plot(zscore(dataSetY_All.iloc[:144]))
    plt.title(featureSet[i])


# ##### feature elimination--colinearity removal

# In[58]:


from scipy.stats import pearsonr


# In[59]:


featureSet = ['const', 'TSupCor.T', 'VSupCor_flow.V_flow', 'occup', 'InternalGain',
       'Dry Bulb Temperature {C}', 'Dew Point Temperature {C}',
       'Relative Humidity {%}', 'Atmospheric Pressure {Pa}',
       'Extraterrestrial Horizontal Radiation {Wh/m2}',
       'Extraterrestrial Direct Normal Radiation {Wh/m2}',
       'Horizontal Infrared Radiation Intensity from Sky {Wh/m2}',
       'Global Horizontal Radiation {Wh/m2}',
       'Direct Normal Radiation {Wh/m2}',
       'Diffuse Horizontal Radiation {Wh/m2}']


# In[60]:


# The guidelines underlying this is: if the correlation coefficient of two variables are higher than threhold value, remove the one that is 
#  reletively irrelavant to Y
def Eliminate_Colinearity(dataSetX_All,datasetY_All,featureSet,threshold_For_Colinearity = 0.9):
    n = len(featureSet)

    feature_To_Remove_idx,feature_To_Remove = [],[]

    for i,feature in enumerate(featureSet):
        for j in range(i+1,n):
            coeff = pearsonr(dataSetX_All.loc[:,featureSet[i]],dataSetX_All.loc[:,featureSet[j]])[0]
            #print(coeff)
            if coeff: # if not nan
                if coeff >= threshold_For_Colinearity:

                    coef_i = pearsonr(dataSetX_All.loc[:,featureSet[i]].tolist(),dataSetY_All.iloc[:,0].tolist())[0]
                    coef_j = pearsonr(dataSetX_All.loc[:,featureSet[j]].tolist(),dataSetY_All.iloc[:,0].tolist())[0]

                    feature_To_Remove_idx.append(j) if  coef_i>= coef_j else feature_To_Remove_idx.append(i)
                    print(coeff,feature_To_Remove_idx)
                    #print(featureSet[i],featureSet[j],'\n')        
    feature_To_Remove_idx = list(set(feature_To_Remove_idx ))
    print(feature_To_Remove_idx)
    for idx in feature_To_Remove_idx:
        feature_To_Remove.append(featureSet[idx])
    return feature_To_Remove


# In[61]:


featureSet = list(dataSetX_All.columns)
feature_To_Remove = Eliminate_Colinearity(dataSetX_All,dataSetY_All,featureSet)
featureSet_No_Colinear = featureSet.copy()
for feature in feature_To_Remove:  
    featureSet_No_Colinear.remove(feature)


# In[159]:


featureSet_No_Colinear


# ##### Feature Selection
# Here we will use wrapper method and implement both forward and backward Feature selection seperately
# 
# **Evaluation Metric**: R2_score on testing dataset
# 
# **Question:** Is it necessary to apply Cross-Validation

# In[160]:


import statsmodels
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score


# **Evaluation metrics and prediction function**

# In[161]:


featureSet_NoHist = ['const', 'TSupCor.T', 'VSupCor_flow.V_flow', 'occup', 'InternalGain',
       'Dry Bulb Temperature {C}', 'Dew Point Temperature {C}',
       'Relative Humidity {%}', 'Atmospheric Pressure {Pa}',
       'Extraterrestrial Horizontal Radiation {Wh/m2}',
       'Extraterrestrial Direct Normal Radiation {Wh/m2}',
       'Horizontal Infrared Radiation Intensity from Sky {Wh/m2}',
       'Global Horizontal Radiation {Wh/m2}',
       'Direct Normal Radiation {Wh/m2}',
       'Diffuse Horizontal Radiation {Wh/m2}']


# In[162]:


# Split the data into training/testing sets
## Ratio of training sets to testing set: **5:1**

ratio = 1/5
splitting_Point = int(len(dataSetX_All)*4/5)
dataSetX_train, dataSetX_test = dataSetX_All.iloc[:splitting_Point,:],dataSetX_All.iloc[splitting_Point:,:]

dataSetY_train, dataSetY_test = dataSetY_All.iloc[:splitting_Point,:],dataSetY_All.iloc[splitting_Point:,:]


# In[163]:


dataSetY_train.shape


# In[164]:


dataSetX_test.shape


# In[38]:


splitting_Point


# In[165]:


## Evaluation Metrics
def Score(y,ypred, f = r2_score):
    score = f(y,ypred)
    print("Score:{:.4f}".format(score))
    return score


# **Model Type: Multi-Variate Linear Regression**

# In[45]:


from statsmodels.regression.linear_model import OLS


# In[41]:


time_lags


# In[51]:


def ModelPred_MLR(xTest_FullFeature, ytrain,featureSet,fmodel,time_lags = time_lags):  
    y_Sequence_Updated = ytrain.copy()
    xTest_FullFeature_copy = xTest_FullFeature.copy()
    for i in range(len(xTest_FullFeature)):
        # Append an empty value to the end of y_Sequence
        y_Sequence_Updated = y_Sequence_Updated.append(pd.DataFrame({y_Sequence_Updated.columns[0]:0},
                        columns = y_Sequence_Updated.columns,index =xTest_FullFeature.index[i:i+1] ))
        
        
        dataSetX_replacement = History_Record_Data(y_Sequence_Updated.iloc[-max(time_lags):,:],time_lags)
        # Generat new train dataset based on the predict at current time
        xTest_FullFeature_copy.loc[:,dataSetX_replacement.columns].iloc[i:i+1,:] = dataSetX_replacement.iloc[-1:,:]
        
        #Predict y_value at the next time step
        y_Single = fmodel.predict(xTest_FullFeature_copy.loc[:,featureSet].iloc[i:i+1,:])
        y_Sequence_Updated.iloc[-1:,:]  = y_Single.iloc[:]
    
    return y_Sequence_Updated.iloc[-len(xTest_FullFeature):,:]


# In[52]:


def ModelTrain_MLR(xtrain_FullFeature,ytrain,f_Set,seed = 0,time_lags = time_lags):
    timestart = time.time()
    # Create linear regression object
    #print(ytrain,xtrain_FullFeature.loc[:,f_Set])
    regr = OLS(ytrain,xtrain_FullFeature.loc[:,f_Set])
    #Fit model
    result_Model = regr.fit()    
    return result_Model


# In[44]:


ypred = ModelPred_MLR(xtest,ytrain,f_Set,result_Model,time_lags = time_lags)
#print(ytest)
#int(ypred)
score = Score(ytest,ypred)
print("Time.{:.2f} seconds".format(time.time()-timestart))


# In[40]:


def MLR(dataSetX_train,dataSetY_train,dataSetX_test,dataSetY_test,f_Set,seed = 0,paramodel = {}, time_lags = time_lags,prediction_Horizon = '6H',method = 1):
    n = int(TimeInterval_Parser(prediction_Horizon)/TimeInterval_Parser(TimeInterval))

    ## for this part, this is two ways to predict the Y value on the whole testing set

    # 1. Build a single model on the trainSet, and predict Y values for the next 6 hrs based on measured value  every time
    current_position, prev_Position = 0, 0
    t0 = time.time()
    
    if method == 1:
        print('training model')
        Model = ModelTrain_MLR(dataSetX_train,dataSetY_train,f_Set,seed=0,time_lags = time_lags)
        print('training completed====')
        while True:
            prev_Position,current_position  = current_position, current_position+n  # Move the pointer n steps forward
            if current_position <= len(dataSetY_test):
                ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:current_position,:],
                                      dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),f_Set,Model,time_lags = time_lags)
                print("Time.{:.2f} seconds".format(time.time()-t0))
                if prev_Position == 0:
                    total_Res = ypred
                else:
                    total_Res = total_Res.append(ypred)
            else:
                ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:,:],
                                      dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),f_Set,Model,time_lags = time_lags)
                print("Time.{:.2f} seconds".format(time.time()-t0))
                if prev_Position == 0:
                    total_Res = ypred
                else:
                    total_Res = total_Res.append(ypred)
                break
        # 2. Build a new model everytime, and predict Y values for the next 6 hrs based on measured value  every time
    elif method == 2:
        while True:
            prev_Position,current_position  = current_position, current_position+n  # Move the pointer n steps forward
            if current_position <= len(dataSetY_test):
                Model = ModelTrain_MLR(dataSetX_train.append(dataSetX_test.iloc[:prev_Position,:]),
                                       dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),f_Set,seed=0,time_lags = time_lags)                    
                ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:current_position,:],
                                      dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),f_Set,Model,time_lags = time_lags)
                print("Time.{:.2f} seconds".format(time.time()-t0))
                if prev_Position == 0:
                    total_Res = ypred
                else:
                    total_Res = total_Res.append(ypred)
            else:
                Model = ModelTrain_MLR(dataSetX_train.append(dataSetX_test.iloc[:prev_Position,:]),
                                       dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),f_Set,seed=0,time_lags = time_lags)                    
                ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:,:],
                                      dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),f_Set,Model,time_lags = time_lags)
                print("Time.{:.2f} seconds".format(time.time()-t0))
                if prev_Position == 0:
                    total_Res = ypred
                else:
                    total_Res = total_Res.append(ypred)
                break
    elif method == 3:
        print('1')


    ## Comparison between predicted value and real value
    plt.plot(total_Res)
    plt.plot(dataSetY_test.iloc[:,:])
    s = Score(total_Res,dataSetY_test)
    return total_Res,s


# In[41]:


##Prediction horizon :3600
prediction_Horizon = '6H'


# In[166]:


s = MLR(dataSetX_train,dataSetY_train,dataSetX_test,dataSetY_test,featureSet,seed = 0,paramodel = {}, time_lags = time_lags,prediction_Horizon = '6H',method = 1)


# **Result: Clearly from the top chart, the model runs as I expected**

# **Forward Feature Selection**

# In[57]:


def anyHistData(featureSet):
    res = False
    for feature in featureSet:
        if feature[:4] == 'hist':
            res = True
            break
    return res        


# In[67]:


## Forward Feature Selection
def forward_feature_selection(x_train, x_test, y_train, y_test, featureSet_Original):
    n = len(featureSet_Original)
    feature_set = []
    Scores = []
    for i,num_features in enumerate(range(n)):
        metric_list = [] # Choose appropriate metric based on business problem
        for j,feature in enumerate(featureSet_Original):
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                t_start = time.time()
                if anyHistData(f_set):
                    (ypred,s) =  MLR(x_train,  y_train,x_test, y_test,f_set,seed = 0,
                                paramodel = {}, time_lags = time_lags,prediction_Horizon = '6H',method = 1)
                else:
                    regr = OLS(dataSetY_train,dataSetX_train.loc[:,f_set])
                    #Fit model
                    result_Model = regr.fit()
                    ypred = result_Model.predict(dataSetX_test.loc[:,f_set])
                    s = Score(dataSetY_test,ypred)
                    
                metric_list.append((s, feature))
                print('=======',i,j)
                print(f_set)
                print('=-======',(s))
                print('time elapsed in this round:{:.4f}'.format(time.time()-t_start))

        metric_list.sort(key=lambda x : x[0], reverse = True) # In case metric follows "the more, the merrier"
        feature_set.append(metric_list[0][1])
        Scores.append((i,j,feature_set,metric_list[0][0]))
    return feature_set,Scores


# In[167]:


(feature_Set_r, scores) = forward_feature_selection(dataSetX_train,dataSetX_test,
                         dataSetY_train,dataSetY_test,featureSet_No_Colinear)


# In[170]:


[(ele[0],ele[3])for ele in scores]


# In[169]:


plt.plot(pd.DataFrame([(ele[0],ele[3])for ele in scores]).iloc[:,1])


# In[171]:


scores[8]


# In[83]:


print('{0:4}'.format(4.444))


# ##### test

# In[172]:


feature_Set_re = scores[8][2]


# In[ ]:


['hist_1',
 'VSupCor_flow.V_flow',
 'hist_72',
 'TSupCor.T',
 'occup',
 'Relative Humidity {%}',
 'InternalGain',
 'Atmospheric Pressure {Pa}',
 'hist_36',
 'Extraterrestrial Horizontal Radiation {Wh/m2}',
 'Extraterrestrial Direct Normal Radiation {Wh/m2}',
 'Direct Normal Radiation {Wh/m2}',
 'const',
 'Dry Bulb Temperature {C}']


# In[173]:


feature_Set_re


# In[174]:


Model = ModelTrain_MLR(dataSetX_train,dataSetY_train,feature_Set_re,seed=0,time_lags = time_lags)
n = 36
current_position, prev_Position = 0, 0
t0 = time.time()
while True:
    prev_Position,current_position  = current_position, current_position+n  # Move the pointer n steps forward
    if current_position <= len(dataSetY_test):
        ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:current_position,:],
                              dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),feature_Set_re,Model,time_lags = time_lags)
        print("Time.{:.2f} seconds".format(time.time()-t0))
        if prev_Position == 0:
            total_Res = ypred
        else:
            total_Res = total_Res.append(ypred)
    else:
        ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:,:],
                              dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),feature_Set_re,Model,time_lags = time_lags)
        print("Time.{:.2f} seconds".format(time.time()-t0))
        if prev_Position == 0:
            total_Res = ypred
        else:
            total_Res = total_Res.append(ypred)
        break


# In[175]:


plt.figure(figsize = [10,5])
plt.plot(Model.predict(dataSetX_train.loc[:,feature_Set_re]))
plt.plot(dataSetY_train.iloc[:,:])
s = Score(Model.predict(dataSetX_train.loc[:,feature_Set_re]),dataSetY_train)

plt.plot(total_Res)
plt.plot(dataSetY_test.iloc[:,:])
s = Score(total_Res,dataSetY_test)


# In[95]:


featureSet_NoHist


# In[96]:


feature_Set_re = featureSet_NoHist


# In[97]:


Model = ModelTrain_MLR(dataSetX_train,dataSetY_train,feature_Set_re,seed=0,time_lags = time_lags)
n = 36
current_position, prev_Position = 0, 0
t0 = time.time()
while True:
    prev_Position,current_position  = current_position, current_position+n  # Move the pointer n steps forward
    if current_position <= len(dataSetY_test):
        ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:current_position,:],
                              dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),feature_Set_re,Model,time_lags = time_lags)
        print("Time.{:.2f} seconds".format(time.time()-t0))
        if prev_Position == 0:
            total_Res = ypred
        else:
            total_Res = total_Res.append(ypred)
    else:
        ypred = ModelPred_MLR(dataSetX_test.iloc[prev_Position:,:],
                              dataSetY_train.append(dataSetY_test.iloc[:prev_Position,:]),feature_Set_re,Model,time_lags = time_lags)
        print("Time.{:.2f} seconds".format(time.time()-t0))
        if prev_Position == 0:
            total_Res = ypred
        else:
            total_Res = total_Res.append(ypred)
        break


# In[98]:


plt.figure(figsize = [10,5])
plt.plot(Model.predict(dataSetX_train.loc[:,feature_Set_re]))
plt.plot(dataSetY_train.iloc[:,:])
s = Score(Model.predict(dataSetX_train.loc[:,feature_Set_re]),dataSetY_train)

plt.plot(total_Res)
plt.plot(dataSetY_test.iloc[:,:])
s = Score(total_Res,dataSetY_test)


# <hr style="border-color:royalblue;background-color:royalblue;height:1px;">
# <div style="text-align:center; margin: 40px 0 40px 0; font-weight:bold">
# [Back to Contents](#toc)
# </div>

# In[ ]:




