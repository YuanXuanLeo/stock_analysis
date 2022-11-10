# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:22:04 2022

@author: yuanxuan
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy
import pandas as pd
import math
import os
import numpy as np
from datetime import datetime
import datetime

#step1載入csv檔 一次載入1個資料集
#df =pd.read_csv("DowJ_data_train.csv") # 那斯達克
# df =pd.read_csv("data/DowJ_addT.csv") # 道瓊
# df =pd.read_csv("data/S_P500_addT.csv") # S&P 500

# 技術指標說明
# df.rename(columns={
#     "Close":"收盤價",
#     "X_SMA":"簡單移動平均",經過S相關性觀察刪除
#     "X_WMA":"指數移動平均",經過相關性觀察刪除
#     "X_EMA":"加權移動平均線",經過相關性觀察刪除
#     "X_WILLR":"慢速隨機指標",
#     "X_RSI":"相對強弱指標",
#     "X_CCI":"順勢指標",
#     "X_MOM":"動量指標",
#     "X_STCD":"快速隨機指標%D",
#     "X_STCK":"快速隨機指標%K",
#     "X_MCAD":"平滑異同移動平均線"
#     "increase":"漲跌幅%",新增特徵值
# },inplace=True)

#model 1 先抽牌訓練模型後回復時間序列預測 (保留5天後資料預測)
#預測天數設定說明丟入model參數
#day_pred  = 20 #預測天數
#RandomForest模型
def random_forests(df,day_pred):
    day_range = day_pred+1 #生成預測天數的時間序列範圍=預測天數+1
    new_df = df
    len(new_df)
    #生成資料集要使用的日期index
    dateset = new_df['date'].copy()
    #生成原始預測資料集要使用的日期index
    original_pred_data=new_df.iloc[len(new_df)-day_range:,:-1].copy()
    original_pred_date=original_pred_data['date'].copy()
    original_pred_data=original_pred_data.drop(['date'],axis=1)
    original_pred_data=original_pred_data.set_index(pd.to_datetime(original_pred_date,format='%Y/%m/%d')) 
    #生成預測資料集要使用的未來日期index
    pred_date_set=(df['date'].iloc[-day_pred:])
    pred_date_set=pred_date_set.reset_index(drop=True)

    pred_date_add = pd.Series((pd.date_range(pred_date_set.iloc[-1], periods =2, freq='b')[1]))
    pred_date_set=pd.concat([pred_date_set,pred_date_add],axis=0, ignore_index=True)
    predict_data = new_df.iloc[len(new_df)-day_range:,:-1].copy()
    predict_data.set_index(pd.to_datetime(pred_date_set,format='%Y/%m/%d'),inplace=True)
    predict_data = predict_data.drop(['date'],axis=1)
    #資料集設定日期index
    new_df.set_index(pd.to_datetime(dateset,format='%Y/%m/%d'),inplace=True,drop=True)
    #刪除data欄位
    new_df = new_df.drop(['date'],axis=1)
    #stock_price = new_df['close'].copy()
    change_df=new_df.drop(['updown'],axis=1)

   #對資料集做normalization            
    for i in list(change_df.columns):
   # 获取各个指标的最大值和最小值
       Max = np.max(change_df[i])
       Min = np.min(change_df[i])
       new_df[i] = (change_df[i] - Min)/(Max - Min)
    
    #split feature and lable(用最後五天當驗證資料不加入訓練)
    X = new_df.iloc[:len(new_df)-5,:-1].copy()
    y = new_df.iloc[:len(new_df)-5,len(new_df.columns)-1:].copy()    
    
    # split data into training data and testing data(先用隨機抽象訓練模型80/20分割資料集)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
    # build model 
    #超參數調整
    rfc=RandomForestClassifier(random_state=42)
    params = { 
        'n_estimators': [300],    # 森林裡樹木的數量
        'max_features': ['auto', 'sqrt', 'log2'],    # 每個決策樹最大的特徵數量
        'max_depth' : [5,8,13],    # 樹的最大深度
        'criterion' :['gini', 'entropy']    # 分類依據
        }

    model = GridSearchCV(estimator=rfc, 
                        param_grid=params, 
                        cv= 5,
                        refit = True,
                        n_jobs = -1)

  # verbose=0不輸出訓練過程=2輸出訓練過程 n_job=-1用所有cpu

    model.fit(X_train, y_train.values.ravel())
    
    # (關閉抽樣功能將資料還原成80/20有時間序列的資料,將資料放入訓練好的模型)
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
    
    y_pred = model.predict(X_test)
    new_predict_data = predict_data.copy()
    #對預測集做normalization            
    for i in list(predict_data.columns):
    # 获取各个指标的最大值和最小值
       Max = np.max(predict_data[i])
       Min = np.min(predict_data[i])
       new_predict_data[i] = (predict_data[i] - Min)/(Max - Min)       
    # evaluate model 驗證模型
    accuracy = accuracy_score(y_test, y_pred)#準確率
    test_score = mean_squared_error(y_test, y_pred)#均方誤差
    con_matrix = confusion_matrix(y_test, y_pred)#混淆矩陣
    # 生成原始預測漲跌訊號
    y_pred1=pd.DataFrame(model.predict(new_df.tail(day_pred).copy().drop(['updown'],axis=1)))
    y_pred1.rename(columns={0:'updown'}, inplace=True)
    y_pred1.set_index(pd.to_datetime(dateset.tail(day_pred),format='%Y/%m/%d'),inplace=True)
    #生成預測機率矩陣(漲跌的機率)
    prob_tomorrow=model.predict_proba(new_predict_data)
    # 復原預測資料集
    predict_data_ud = model.predict(new_predict_data)
    predict_data.loc[:,'updown'] = predict_data_ud
    tommor = predict_data
    
    return (accuracy, test_score, prob_tomorrow,tommor, predict_data, pred_date_set, original_pred_data,y_pred1)
#KNN模型
def KNN(df,day_pred):
    day_range = day_pred+1 #生成預測天數的時間序列範圍=預測天數+1
    new_df = df
    len(new_df)
    #生成資料集要使用的日期index
    dateset = new_df['date'].copy()
    #生成原始預測資料集要使用的日期index
    original_pred_data=new_df.iloc[len(new_df)-day_range:,:-1].copy()
    original_pred_date=original_pred_data['date'].copy()
    original_pred_data=original_pred_data.drop(['date'],axis=1)
    original_pred_data=original_pred_data.set_index(pd.to_datetime(original_pred_date,format='%Y/%m/%d')) 
    #生成預測資料集要使用的未來日期index
    pred_date_set=(df['date'].iloc[-day_pred:])
    pred_date_set=pred_date_set.reset_index(drop=True)
    pred_date_add = pd.Series((pd.date_range(pred_date_set.iloc[-1], periods =2, freq='b')[1]))
    pred_date_set=pd.concat([pred_date_set,pred_date_add],axis=0, ignore_index=True)
    predict_data = new_df.iloc[len(new_df)-day_range:,:-1].copy()
    predict_data.set_index(pd.to_datetime(pred_date_set,format='%Y/%m/%d'),inplace=True)
    predict_data = predict_data.drop(['date'],axis=1)
    #資料集設定日期index
    new_df.set_index(pd.to_datetime(dateset,format='%Y/%m/%d'),inplace=True,drop=True)
    #刪除data欄位
    new_df = new_df.drop(['date'],axis=1)
    #stock_price = new_df['close'].copy()
    change_df=new_df.drop(['updown'],axis=1)

   #對資料集做normalization            
    for i in list(change_df.columns):
   # 获取各个指标的最大值和最小值
      Max = np.max(change_df[i])
      Min = np.min(change_df[i])
      new_df[i] = (change_df[i] - Min)/(Max - Min)
    
    #split feature and lable(用最後五天當驗證資料不加入訓練)
    X = new_df.iloc[:len(new_df)-5,:-1].copy()
    y = new_df.iloc[:len(new_df)-5,len(new_df.columns)-1:].copy()    
    
    # split data into training data and testing data(先用隨機抽象訓練模型80/20分割資料集)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 2000)
    
    # build model 
    #超參數調整
    kn = neighbors.KNeighborsClassifier()
    params = {
    'n_neighbors' : [5,10,20],    # 邻居个数
    'weights': ['uniform', 'distance'],    # uniform不带距离权重,distance带有距离权重
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],   # 搜尋數演算法
    }
    model = GridSearchCV(estimator = kn,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 5,    # cv=交叉驗證參數,
                        refit = True,
                        n_jobs = -1)    # verbose=0不輸出訓練過程=2輸出訓練過程 n_job=-1用所有cpu

    model.fit(X_train, y_train.values.ravel())
    
   # 使用預設超參數              
   # model = neighbors.KNeighborsClassifier(n_neighbors=5)
   # model.fit(X_train, y_train.values.ravel())
    # (關閉抽樣功能將資料還原成80/20有時間序列的資料,將資料放入訓練好的模型)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
    
    y_pred = model.predict(X_test)
   # 生成原始預測漲跌訊號
    y_pred1=pd.DataFrame(model.predict(new_df.tail(day_pred).copy().drop(['updown'],axis=1)))
    y_pred1.rename(columns={0:'updown'}, inplace=True)
    y_pred1.set_index(pd.to_datetime(dateset.tail(day_pred),format='%Y/%m/%d'),inplace=True)
    new_predict_data = predict_data.copy()
    #對預測集做normalization            
    for i in list(predict_data.columns):
    # 获取各个指标的最大值和最小值
       Max = np.max(predict_data[i])
       Min = np.min(predict_data[i])
       new_predict_data[i] = (predict_data[i] - Min)/(Max - Min)       
    # evaluate model 驗證模型
    accuracy = accuracy_score(y_test, y_pred)#準確率
    test_score = mean_squared_error(y_test, y_pred)#均方誤差
    con_matrix = confusion_matrix(y_test, y_pred)#混淆矩陣
    #生成預測機率矩陣(漲跌的機率)
    prob_tomorrow=model.predict_proba(new_predict_data)
    # 復原預測資料集
    predict_data_ud = model.predict(new_predict_data)
    predict_data.loc[:,'updown'] = predict_data_ud
    tommor = predict_data
    
    return (accuracy, test_score, prob_tomorrow,tommor, predict_data, pred_date_set, original_pred_data,y_pred1)
