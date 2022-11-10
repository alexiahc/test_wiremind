#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:31:01 2022

@author: alexi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import statistics

###############################################################################

#########################    DATA ANALYSIS      ###############################

###############################################################################

#%%
rep = ""

df = pd.read_csv(rep+"dataset.csv")

print("df\n", df.head())
print("df columns\n", df.columns)
# ChargeableWeight and Revenue will be use as target value
print("df types\n",df.dtypes)
# type object will be change to have just float and int 
# with one-hot encoding or ordinal encoding 
print("df number null values\n",df.isnull().sum().sort_values(ascending=False))
# there is no null values 
print("df describe\n",df.describe())
print("df number unique values\n",df.nunique())
# FlownYear has a unique value (2017)
# Revenue can be 0 (?)

X = df.copy()
# target values : price per weight 
y = X.Revenue / X.ChargeableWeight
X.drop(['Revenue'], axis=1, inplace=True)
X.drop(['ChargeableWeight'], axis=1, inplace=True)

# train and valid sets, 80% in the training set
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

#%% categorical values

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

print("number unique values for train dataset and object type\n",X_train[object_cols].nunique())

# DestinationCode has a unique value
# 3 variables have less then 10 different values
# the rest have more then 15 different values (609 for AgentCode)

#%%

# train set with X_train and y_train for group by 
train = X_train.copy()
train['Price_pw'] = y_train.copy()

#%% figures variables 

print("statistics about some columns")
plt.figure()
sns.countplot(x=pd.qcut(train['Price_pw'],5), hue=train['FlownMonth'])
print("\n mean")
print(train['Price_pw'].groupby(train['FlownMonth']).mean())
print("\n count")
print(train['Price_pw'].groupby(train['FlownMonth']).count())
print("\n std")
print(train['Price_pw'].groupby(train['FlownMonth']).std())
# no great difference of the prices for the three months in term of
# mean, count or standard deviation
# no relevant evolution along time 

plt.figure()
sns.countplot(x=pd.qcut(train['Price_pw'],5), hue=train['DocumentRatingSource'])
print("\n mean")
print(train['Price_pw'].groupby(train['DocumentRatingSource']).mean())
print("\n count")
print(train['Price_pw'].groupby(train['DocumentRatingSource']).count())
print("\n std")
print(train['Price_pw'].groupby(train['DocumentRatingSource']).std())
# shipments with low prices come more from Document rating source AAAA and YYYY
# shipments from document rating source ZZZZ have high prices only 
# from XXX, AACC and AABB are distributed between the 3rd and 5th quantiles, 
# with high mean for AABB and ZZZ have large standard deviation 
# AAAA have half the shipments (around 3000), then AACC and YYYY have 1000 each

plt.figure()
sns.countplot(x=pd.qcut(train['Price_pw'],5), hue=train['CargoType'])
print("\n mean")
print(train['Price_pw'].groupby(train['CargoType']).mean())
print("\n count")
print(train['Price_pw'].groupby(train['CargoType']).count())
print("\n std")
print(train['Price_pw'].groupby(train['CargoType']).std())
# 5/6 of shipments have cargo type ZZZ
# types YYY have prices mostly high, with large standard deviation 
# prices of the type XXX have prices in the three last quantiles mostly

print("\n mean")
print(train['Price_pw'].groupby(train['ProductCode']).mean())
print("\n count")
print(train['Price_pw'].groupby(train['ProductCode']).count())
print("\n std")
print(train['Price_pw'].groupby(train['ProductCode']).std())
plt.figure()
sns.barplot(x=train['ProductCode'], y=train['Price_pw'])
plt.xticks(rotation='vertical')
# most shipments with product code DLJ then with MGK
# some product code with less then 10 shipments 
# large standard deviation for MGV, MGK, SJD, X, XGS
# mean less then 10 for 10 product code 
# 7 others between 10 and 40

plt.figure()
sns.barplot(x=train['POSCountryName'], y=train['Price_pw'])
# 5 country with high mean compared to the others countries 
# uncertainty large for 7 countries (two with high means)
# most countries have prices lower then 10 

#%% categories with many values 

cols = ['POSCountryName', 'POS', 'OriginCode', 'CommodityCode', 'AgentCode', 
        'SpecialHandlingCodeList']
for col in cols:
    group = train.groupby(train[col]).mean()
    plt.figure()
    sns.barplot(x=train[col], y=train['Price_pw'])
    plt.figure()
    sns.barplot(x=group[group.Price_pw < 5].index, y=group[group.Price_pw < 5]['Price_pw'])
    plt.figure()
    sns.barplot(x=group[group.Price_pw > 5].index, y=group[group.Price_pw > 5]['Price_pw'])

# many different categories 
# few with high prices above 100, uncertainty large for many categories 
# most categories have values under a price of 5 

#%% boxplot for variables with no much different values

cols_categ = ['DocumentRatingSource', 'FlownMonth', 'POSCountryName', 
              'CargoType', 'ProductCode' ]

for categ in cols_categ:
    data = pd.concat([train['Price_pw'], train[categ]], axis=1)
    f, ax = plt.subplots(figsize=(16, 10))
    fig = sns.boxplot(x=categ, y="Price_pw", data=data)
    fig.axis(ymin=0, ymax=100) # some values to high to compare 
    xt = plt.xticks(rotation=45)

# for document rating source, flown month and cargo type, boxplots confirm
# the distribution of the variables, and show the repartition of the values
# outside the quarters

# for pos country, the values with the highest median have the largest repartition 
# values are mostly under 10 and their repartition are around little values 

# for product code, median are under 20 for each, SJD has the largest box
# with confirms the standard deviation, and DLJ having more shipments has 
# more outliers above the interquartile range 

#%% agent code 

ACcount = train['AgentCode'].groupby(train.AgentCode).count()
print("agent code count : " , ACcount.describe())
plt.figure()
plt.plot(ACcount)

# two agent code with more shipments, one with 600, one with 250

ACcount = train['Price_pw'].groupby(train.AgentCode).mean()
print("agent code mean of prices : " , ACcount.describe())
plt.figure()
plt.plot(ACcount)

plt.figure()
plt.plot(ACcount[ACcount < 150])

#%% figures target 

plt.figure()
plt.hist(np.log(y_train[y_train!=0]),orientation = 'vertical',histtype = 'bar')

plt.figure()
sns.histplot(np.log(y_train[y_train!=0]), kde=True)

plt.figure()
sns.distplot(np.log(y_train[y_train!=0]), kde=False, fit=stats.lognorm)

plt.figure()
sns.distplot(np.log(y_train[y_train!=0]), kde=False, fit=stats.johnsonsu)

# skewed distribution of y_train
# values are mainly around 0 (plot log(target) without null prices)
# close to a johnsonsu distribution with a pic at the top 

print("target y_train " , y_train.describe())

# large standard deviation (50), mean at 12
# 75% under 2.7 for the price 

#%% FlownYear and DestinationCode have an unique value 

X_train.drop(['FlownYear'], axis=1, inplace=True)
X_train.drop(['DestinationCode'], axis=1, inplace=True)

X_valid.drop(['FlownYear'], axis=1, inplace=True)
X_valid.drop(['DestinationCode'], axis=1, inplace=True)

#%% use one-hot encoding for categorical value with less then 10 unique values 

OH_cols = [col for col in X_train.columns if (X_train[col].dtype == "object") 
                                           and (df.nunique()[col] < 10)]

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_encoder.fit(X[OH_cols])
OH_cols_name = OH_encoder.get_feature_names_out(OH_cols)

OH_cols_train = pd.DataFrame(OH_encoder.transform(X_train[OH_cols]), columns = OH_cols_name)
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[OH_cols]), columns = OH_cols_name)

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

obj_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
# take only numerical values to concatenate 
num_X_train = X_train.drop(obj_cols, axis=1)
num_X_valid = X_valid.drop(obj_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

#%% use ordinal encoding for the other values 

enc_cols = [col for col in X_train.columns if (X_train[col].dtype == "object")
                                           and (df.nunique()[col] >= 10)]

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = 700)
# for the valid set, it is possible to not have encoded value in some columns,
# so we use here the value 700 which cannot be the encoded value of one 
# from the train set (the maximum number of unique values is 679 for the 
# whole daatset)

enc_cols_train = pd.DataFrame(encoder.fit_transform(X_train[enc_cols]), columns = enc_cols)
enc_cols_valid = pd.DataFrame(encoder.transform(X_valid[enc_cols]), columns = enc_cols)

enc_cols_train.index = X_train.index
enc_cols_valid.index = X_valid.index

# concatenate with the dataset of one-hot encoded values and numerical values
nv_X_train = pd.concat([enc_cols_train, OH_X_train], axis=1)
nv_X_valid = pd.concat([enc_cols_valid, OH_X_valid], axis=1)

scaler = StandardScaler()
# standardize the data 
nv_X_train = pd.DataFrame(scaler.fit_transform(nv_X_train), columns = nv_X_train.columns) 
nv_X_valid = pd.DataFrame(scaler.transform(nv_X_valid), columns = nv_X_valid.columns)

#%% transform the target 

# We cut the train target into quantiles, we choose here 30 because the last 
# quantiles are very large with less then 30 (after the price 15, quantiles are 
# (15, 35], (35, 152] and (152, 734]), and the intervals for small values 
# are small so we can easly get a closer price to the reality for the most 
# rows. 
# For each intervals we keep the midpoint to train the models, so models 
# will be accurate with the willing to pay for small values, but they will be 
# less accurate for higher ones. 

y_train_q = pd.qcut(y_train, 30)

# list of sorted 30 intervals from the train data
intervs = y_train_q.unique()
ind = intervs.argsort()
intervs = intervs[ind]

# function find_interv
# input : x, float 
# outuput : interval, Interval
# return the interval corresponding to the value x 
def find_interv(x):
    find = False 
    i=0
    while (not find) and (i<len(intervs)):
        interv = intervs[i]
        find = (x in interv)
        i+=1
    return interv

# apply to the valid dataset 
y_valid_q = y_valid.apply(find_interv)

y_train_f = y_train_q.apply(lambda x : x.mid).astype(float)
y_valid_f = y_valid_q.apply(lambda x : x.mid).astype(float)

# we take here the midpoint of every intervals, so the willing price will be 
# well describe for lower prices, but the quartiles are larger for higher
# prices so the midpoint will not really correspond to the expected price

# we take midpoint to have a float but we can return to the intervals after 
# with apply(find_interv) on train and valid set 

#%% correlation matrix

train = nv_X_train.copy()
train['Price_pw'] = y_train_f.copy()

correlation = train.corr()
k= train.shape[1]
cols = correlation.nlargest(k,'Price_pw')['Price_pw'].index

f, ax = plt.subplots(figsize = (14,12))
sns.heatmap(np.corrcoef(train[cols].values.T), vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

print("correlation with prices : ", correlation.Price_pw.sort_values(ascending = False))

sns.set()
sns.pairplot(train[cols], height = 2 ,kind ='scatter',diag_kind='kde')
plt.show()

# POS and OriginCode correlate (0.92)
# AgentCode and AgentName correlate (0.92)
# DocumentRatingSource_XXXX and CargoType_YYY correlate (0.97)
# correlation is high, can keep just one of each 

#%% 

cols = nv_X_train.columns
print("Pearson’s correlation coef, p-value non-correlation\n")
for col in cols:
    print('\n', col)
    print(stats.pearsonr(y_train_f, train[col]))

# p_value > 0.05 for AgentCode, CommodityCode, DocumentRatingSource_AACC,
# FlownMonth_SEPTEMBER, CargoType_XXX, we cannot reject H0 
# for the last three, we can not keep them because the values of the 
# others columns allow to understand which of the different categorical
# value correspond to the row 
# AgentCode is highly correlate to AgentName so we can not keep it too 

# for the other, p-value < 0.05 that means we reject the hypothesis of 
# non-correlation between the price and the variable, in favor of correlation 

#%%

print("rank ", np.linalg.matrix_rank(nv_X_train))
u, s, vh = np.linalg.svd(nv_X_train)
var_explained = np.round(s**2/np.sum(s**2), decimals=3)
 
sns.barplot(x=list(range(1,len(var_explained)+1)),
            y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('svd_scree_plot.png',dpi=100)

# the fisrt singular values do not contain a lot of information compared
# to the others singular values, we can not reduce the matrice with them 

#%% selection of the variables 

# variable with high correlation with another variable 
X_train_mod = nv_X_train.drop('AgentCode', axis=1).drop('POS',axis=1).drop('DocumentRatingSource_XXXX',axis=1)
X_valid_mod = nv_X_valid.drop('AgentCode', axis=1).drop('POS',axis=1).drop('DocumentRatingSource_XXXX',axis=1)

# variable with discutable correlation 
X_train_mod = X_train_mod.drop('DocumentRatingSource_AACC', axis=1).drop('FlownMonth_SEPTEMBER', axis=1).drop('CargoType_XXX', axis=1)
X_valid_mod = X_valid_mod.drop('DocumentRatingSource_AACC', axis=1).drop('FlownMonth_SEPTEMBER', axis=1).drop('CargoType_XXX', axis=1)

#%%

###############################################################################

###########################     MODELIZATION      #############################

###############################################################################

# the dataset isn't very large so we can use cross-validation on the models
# to compute the r2 score and the mean square error on the validation set 
# for each model 

# we add to the plot also the scores without cross-validation to compare,
# val_ is when validation set is used, train_ is for the train set 

# we print in the shell the most accurate score we can have for each model 

# here we use regressor model because we want to find the value of the midpoint
# for the previous intervals 

#%% ramdom forest 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse 
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_jobs=-1)

# the model is using bootstrap to compute, the is no maximum for the depth so 
# the training can be longer than the others models, and we use n_jobs=-1
# to parallelize the process for fit and predict 
# we will test several number of trees in the forest to keep the best one 

estimators = np.arange(10, 150, 10)
scores = []
scores2 = []
l_mse = []
l_mse2 = []

tscores = []
tscores2 = []
tl_mse = []
tl_mse2 = []

for n in estimators:
    rf.set_params(n_estimators=n)
    rf.fit(X_train_mod, y_train_f)
    
    scores.append(rf.score(X_valid_mod, y_valid_f))
    scores2.append(cross_val_score(rf, X_valid_mod, y_valid_f, cv=5, scoring='r2'))
    l_mse.append(mse(y_valid_f, rf.predict(X_valid_mod)))
    l_mse2.append(-1*cross_val_score(rf, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))
    
    tscores.append(rf.score(X_train_mod, y_train_f))
    tscores2.append(cross_val_score(rf, X_train_mod, y_train_f, cv=5, scoring='r2'))
    tl_mse.append(mse(y_train_f, rf.predict(X_train_mod)))
    tl_mse2.append(-1*cross_val_score(rf, X_train_mod, y_train_f, cv=5, scoring='neg_mean_squared_error'))

plt.figure()
plt.title("Effect of n_estimators RandomForestRegressor")
plt.xlabel("n_estimator")
plt.ylabel("score R2")
plt.plot(estimators, scores, label="val_simple")
plt.plot(estimators, list(map(statistics.mean, scores2)), 
         label="val_cross_validation")
plt.plot(estimators, tscores, label="train_simple")
plt.plot(estimators, list(map(statistics.mean, tscores2)), 
         label="train_cross_validation")
plt.legend()

plt.figure()
plt.title("MSE Evaluation RandomForestRegressor")
plt.xlabel("n_estimators")
plt.ylabel("MSE")
plt.plot(estimators, l_mse, label="val_simple")
plt.plot(estimators, list(map(statistics.mean, l_mse2)), 
         label="val_cross_validation")
plt.plot(estimators, tl_mse, label="tain_simple")
plt.plot(estimators, list(map(statistics.mean, tl_mse2)), 
         label="cross_validation")
plt.legend()

# n_estimators=60 with cross validation score give one of the best approximation 

rf.set_params(n_estimators=60)
rf.fit(X_train_mod, y_train_f)
print("\nRandom forest validation ")
print("Score R2 : ", statistics.mean(cross_val_score(rf, X_valid_mod, y_valid_f, cv=5, scoring='r2')))
print("MSE : ", statistics.mean((-1*cross_val_score(rf, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))))
# print(np.transpose([list(rf.feature_names_in_), list(rf.feature_importances_)]))

#%% k nearest neighbors 

from sklearn.neighbors import KNeighborsRegressor

knr =  KNeighborsRegressor()

# the algorithm to compute the nearest neighbors is auto so the model
# will choose the best one based on the fitting values, for distance computation
# we use de standard euclidiean distance (default one) 
# we will test several number of neighbors and two different weight function 
# to keep the best ones for the model 

n_neighbors = np.arange(2, 10, 1)
scores_unif = []
scores_dist = []
scores2_unif = []
scores2_dist = []
l_mse_unif = []
l_mse2_unif = []
l_mse_dist = []
l_mse2_dist = []

tscores_unif = []
tscores_dist = []
tscores2_unif = []
tscores2_dist = []
tl_mse_unif = []
tl_mse2_unif = []
tl_mse_dist = []
tl_mse2_dist = []

for nnb in n_neighbors:
    for weights in ["uniform", "distance"]:
        knr.set_params(n_neighbors=nnb, weights=weights)
        knr.fit(X_train_mod, y_train_f)
        if weights == "uniform":
            scores_unif.append(knr.score(X_valid_mod, y_valid_f))
            scores2_unif.append(cross_val_score(knr, X_valid_mod, y_valid_f, cv=5, scoring='r2'))
            l_mse_unif.append(mse(y_valid_f, knr.predict(X_valid_mod)))
            l_mse2_unif.append(-1*cross_val_score(knr, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))
            
            tscores_unif.append(knr.score(X_train_mod, y_train_f))
            tscores2_unif.append(cross_val_score(knr, X_train_mod, y_train_f, cv=5, scoring='r2'))
            tl_mse_unif.append(mse(y_train_f, knr.predict(X_train_mod)))
            tl_mse2_unif.append(-1*cross_val_score(knr, X_train_mod, y_train_f, cv=5, scoring='neg_mean_squared_error'))

        else:
            scores_dist.append(knr.score(X_valid_mod, y_valid_f))
            scores2_dist.append(cross_val_score(knr, X_valid_mod, y_valid_f, cv=5, scoring='r2'))
            l_mse_dist.append(mse(y_valid_f, knr.predict(X_valid_mod)))
            l_mse2_dist.append(-1*cross_val_score(knr, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))

            tscores_dist.append(knr.score(X_train_mod, y_train_f))
            tscores2_dist.append(cross_val_score(knr, X_train_mod, y_train_f, cv=5, scoring='r2'))
            tl_mse_dist.append(mse(y_train_f, knr.predict(X_train_mod)))
            tl_mse2_dist.append(-1*cross_val_score(knr, X_train_mod, y_train_f, cv=5, scoring='neg_mean_squared_error'))

# uniform
plt.figure()
plt.title("Effect of n_neighbors KNeighborsRegressor (uniform)")
plt.xlabel("n_neighbors")
plt.ylabel("score R2")
plt.plot(n_neighbors, scores_unif, label="val_simple")
plt.plot(n_neighbors, list(map(statistics.mean, scores2_unif)), 
         label="val_cross_validation")
plt.plot(n_neighbors, tscores_unif, label="train_simple")
plt.plot(n_neighbors, list(map(statistics.mean, tscores2_unif)), 
         label="train_cross_validation")
plt.legend()

plt.figure()
plt.title("MSE Evaluation KNeighborsRegressor (uniform)")
plt.xlabel("n_neighbors")
plt.ylabel("MSE")
plt.plot(n_neighbors, l_mse_unif, label="val_simple")
plt.plot(n_neighbors, list(map(statistics.mean, l_mse2_unif)), 
         label="val_cross_validation")
plt.plot(n_neighbors, tl_mse_unif, label="train_simple")
plt.plot(n_neighbors, list(map(statistics.mean, tl_mse2_unif)), 
         label="train_cross_validation")
plt.legend()

# distance 

plt.figure()
plt.title("Effect of n_neighbors KNeighborsRegressor (distance)")
plt.xlabel("n_neighbors")
plt.ylabel("score R2")
plt.plot(n_neighbors, scores_dist, label="val_simple")
plt.plot(n_neighbors, list(map(statistics.mean, scores2_dist)), 
         label="val_cross_validation")
plt.plot(n_neighbors, tscores_dist, label="train_simple")
plt.plot(n_neighbors, list(map(statistics.mean, tscores2_dist)), 
         label="train_cross_validation")
plt.legend()

plt.figure()
plt.title("MSE Evaluation KNeighborsRegressor (distance)")
plt.xlabel("n_neighbors")
plt.ylabel("MSE")
plt.plot(n_neighbors, l_mse_dist, label="val_simple")
plt.plot(n_neighbors, list(map(statistics.mean, l_mse2_dist)), 
         label="val_cross_validation")
plt.plot(n_neighbors, tl_mse_dist, label="train_simple")
plt.plot(n_neighbors, list(map(statistics.mean, tl_mse2_dist)), 
         label="train_cross_validation")
plt.legend()

# n_neighbors=7 and weight distance with cross validation score give one of 
# the best approximation 

knr.set_params(n_neighbors=7, weights='distance')
knr.fit(X_train_mod, y_train_f)
print("\nK-nearest neighbors validation")
print("Score R2 : ", statistics.mean(cross_val_score(knr, X_valid_mod, y_valid_f, cv=5, scoring='r2')))
print("MSE : ", statistics.mean((-1*cross_val_score(knr, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))))

#%% decision tree 

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

# use criterion squared error to measure the quality of the split
# we change de numbre max_depth to compute the best model 

depths = np.arange(2, 15, 1)
scores = []
scores2 = []
l_mse = []
l_mse2 = []

tscores = []
tscores2 = []
tl_mse = []
tl_mse2 = []

for m in depths:
    dtr.set_params(max_depth=m)
    dtr.fit(X_train_mod, y_train_f)
    scores.append(dtr.score(X_valid_mod, y_valid_f))
    scores2.append(cross_val_score(dtr, X_valid_mod, y_valid_f, cv=5, scoring='r2'))
    l_mse.append(mse(y_valid_f, dtr.predict(X_valid_mod)))
    l_mse2.append(-1*cross_val_score(dtr, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))

    tscores.append(dtr.score(X_train_mod, y_train_f))
    tscores2.append(cross_val_score(dtr, X_train_mod, y_train_f, cv=5, scoring='r2'))
    tl_mse.append(mse(y_train_f, dtr.predict(X_train_mod)))
    tl_mse2.append(-1*cross_val_score(dtr, X_train_mod, y_train_f, cv=5, scoring='neg_mean_squared_error'))

plt.figure()
plt.title("Effect of max_depth DecisionTreeRegressor")
plt.xlabel("max_depth")
plt.ylabel("score R2")
plt.plot(depths, scores, label="val_simple")
plt.plot(depths, list(map(statistics.mean, scores2)), 
         label="val_cross_validation")
plt.plot(depths, tscores, label="train_simple")
plt.plot(depths, list(map(statistics.mean, tscores2)), 
         label="train_cross_validation")
plt.legend()

plt.figure()
plt.title("MSE Evaluation DecisionTreeRegressor")
plt.xlabel("n_estimators")
plt.ylabel("MSE")
plt.plot(depths, l_mse, label="val_simple")
plt.plot(depths, list(map(statistics.mean, l_mse2)), 
         label="val_cross_validation")
plt.plot(depths, tl_mse, label="train_simple")
plt.plot(depths, list(map(statistics.mean, tl_mse2)), 
         label="tain_cross_validation")
plt.legend()

# max_depth=4 with cross validation score give one of the best approximation 

dtr = DecisionTreeRegressor(max_depth=4)
dtr.fit(X_train_mod, y_train_f)
print("\nDecision tree validation")
print("Score R2 : ", statistics.mean(cross_val_score(dtr, X_valid_mod, y_valid_f, cv=5, scoring='r2')))
print("MSE : ", statistics.mean((-1*cross_val_score(dtr, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))))
# print(np.transpose([list(rf.feature_names_in_), list(rf.feature_importances_)]))

#%% gradient boosting 

from sklearn.ensemble import GradientBoostingRegressor

gbc = GradientBoostingRegressor()

# we keep the squared error for loss function, and a learning rate of 0.1 which
# allows to have a more stable response 
# we will change the number of boosting stages to performed to get the best model 

estimators = np.arange(10, 200, 10)
scores = []
scores2 = []
l_mse = []
l_mse2 = []

tscores = []
tscores2 = []
tl_mse = []
tl_mse2 = []
 
for n in estimators:
    gbc.set_params(n_estimators=n)
    gbc.fit(X_train_mod, y_train_f)
    scores.append(gbc.score(X_valid_mod, y_valid_f))
    scores2.append(cross_val_score(gbc, X_valid_mod, y_valid_f, cv=5, scoring='r2'))
    l_mse.append(mse(y_valid_f, gbc.predict(X_valid_mod)))
    l_mse2.append(-1*cross_val_score(gbc, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))

    tscores.append(gbc.score(X_train_mod, y_train_f))
    tscores2.append(cross_val_score(gbc, X_train_mod, y_train_f, cv=5, scoring='r2'))
    tl_mse.append(mse(y_train_f, gbc.predict(X_train_mod)))
    tl_mse2.append(-1*cross_val_score(gbc, X_train_mod, y_train_f, cv=5, scoring='neg_mean_squared_error'))

plt.figure()
plt.title("Effect of n_estimators GradientBoostingRegressor")
plt.xlabel("n_estimator")
plt.ylabel("score R2")
plt.plot(estimators, scores, label="val_simple")
plt.plot(estimators, list(map(statistics.mean, scores2)), 
         label="val_cross_validation")
plt.plot(estimators, tscores, label="train_simple")
plt.plot(estimators, list(map(statistics.mean, tscores2)), 
         label="train_cross_validation")
plt.legend()

plt.figure()
plt.title("MSE Evaluation GradientBoostingRegressor")
plt.xlabel("n_estimators")
plt.ylabel("MSE")
plt.plot(estimators, l_mse, label="val_simple")
plt.plot(estimators, list(map(statistics.mean, l_mse2)), 
         label="val_cross_validation")
plt.plot(estimators, tl_mse, label="train_simple")
plt.plot(estimators, list(map(statistics.mean, tl_mse2)), 
         label="train_cross_validation")
plt.legend()

# n_estimators=80 with cross validation score give one of the best approximation, 
# which is before the mostly constant score of the model 

gbc.set_params(n_estimators=75)
gbc.fit(X_train_mod, y_train_f)
print("\nGradient boosting validation")
print("Score R2 : ", statistics.mean(cross_val_score(gbc, X_valid_mod, y_valid_f, cv=5, scoring='r2')))
print("MSE : ", statistics.mean((-1*cross_val_score(gbc, X_valid_mod, y_valid_f, cv=5, scoring='neg_mean_squared_error'))))


#%%

# the model with the maximum r2 and the minimum MSE is the random forest 
# model, which has close results with the gradient boosting algorithm

# the most stable ones are the gradient boosting model evolving with 
# the number of estimators
# there are also the two models which are near to have constant results with
# cross-validation, which are the K-nearest neighbors model with the weights 
# distance and the random forest model

# the models of decision trees and k-nearest neighbors are the two faster 
# models to train 













