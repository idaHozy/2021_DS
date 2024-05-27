import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)

# Read csv
df = pd.read_excel('IT_3.xlsx')
# Drop 5 columns
df = df.drop(['Age_bucket','EngineHP_bucket','Years_Experience_bucket'
                 ,'Miles_driven_annually_bucket','credit_history_bucket'], axis=1)

# Fill missing values
df.fillna(axis=0, method='ffill',inplace=True)

#make target to category
targets = df['annual_claims'].astype(np.int64)
targets = targets.astype('category')

df_ordinal = df.copy()
df_oneHot = df.copy()
df_label = df.copy()

# Convert 'Marital_Status' feature to numeric values using ordinalEncoder
ordinalEncoder = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Marital_Status'])
ordinalEncoder.fit(X)
df_ordinal['Marital_Status'] = pd.DataFrame(ordinalEncoder.transform(X))

# Convert 'Marital_Status' feature to numeric values using labelEncoder
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(df_label['Marital_Status'])
df_label['Marital_Status'] = labelEncoder.transform(df_label['Marital_Status'])


# Convert 'Vehical_type' feature to numeric values using ordinalEncoder
ordEnc = preprocessing.OrdinalEncoder()
X = pd.DataFrame(df['Vehical_type'])
ordEnc.fit(X)
df_ordinal['Vehical_type'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'Vehical_type' feature to numeric values using labelEncoder
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(df['Vehical_type'])
df_label['Vehical_type'] = pd.DataFrame(labelEnc.transform(df['Vehical_type']))


# Convert 'Gender' features to numeric values using ordinalEncoder
X = pd.DataFrame(df['Gender'])
ordEnc.fit(X)
df_ordinal['Gender']  = pd.DataFrame(ordEnc.transform(X))

# Convert 'Gender' features to numeric values using labelEncoder
labelEnc.fit(X)
df_label['Gender'] = pd.DataFrame(labelEnc.transform(X))


# Convert 'State' features to numeric values using ordinalEncoder
X = pd.DataFrame(df['State'])
ordEnc.fit(X)
df_ordinal['State'] = pd.DataFrame(ordEnc.transform(X))

# Convert 'State' features to numeric values using labelEncoder
labelEnc.fit(df['State'])
df_label['State'] = pd.DataFrame(labelEnc.transform(df['State']))


# Getting all the categorical variables in a list
categoricalColumn = df.columns[df.dtypes == np.object].tolist()
# Convert categorical features to numeric values using oneHotEncoder
for col in categoricalColumn:
    if(len(df_oneHot[col].unique()) == 2):
        df_oneHot[col] = pd.get_dummies(df_oneHot[col], drop_first=True)

df_oneHot = pd.get_dummies(df_oneHot)




# Normalizing the ordinalEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_ordinal_maxAbs = scaler.fit_transform(df_ordinal)
df_ordinal_maxAbs = pd.DataFrame(df_ordinal_maxAbs, columns=df_ordinal.columns)
print(df_ordinal_maxAbs.head(10))



# Normalizing the oneHotEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_oneHot_maxAbs = scaler.fit_transform(df_oneHot)
df_oneHot_maxAbs = pd.DataFrame(df_oneHot_maxAbs, columns=df_oneHot.columns)
print(df_oneHot_maxAbs.head(10))



# Normalizing the labelEncoded dataset using MaxAbsScaler
scaler = preprocessing.MaxAbsScaler()
df_label_maxAbs = scaler.fit_transform(df_label)
df_label_maxAbs = pd.DataFrame(df_label_maxAbs, columns=df_label.columns)
print(df_label_maxAbs.head(10))


# Normalizing the ordinalEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_ordinal_robust = scaler.fit_transform(df_ordinal)
df_ordinal_robust = pd.DataFrame(df_ordinal_robust, columns=df_ordinal.columns)
print(df_ordinal_robust.head(10))



# Normalizing the oneHotEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_oneHot_robust = scaler.fit_transform(df_oneHot)
df_oneHot_robust = pd.DataFrame(df_oneHot_robust, columns=df_oneHot.columns)
print(df_oneHot_robust.head(10))



# Normalizing the labelEncoded dataset using RobustScaler
scaler = preprocessing.RobustScaler()
df_label_robust = scaler.fit_transform(df_label)
df_label_robust = pd.DataFrame(df_label_robust, columns=df_label.columns)
print(df_label_robust.head(10))



# Normalizing the ordinalEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_ordinal_minMax = scaler.fit_transform(df_ordinal)
df_ordinal_minMax = pd.DataFrame(df_ordinal_minMax, columns=df_ordinal.columns)
print(df_ordinal_minMax.head(10))

# Normalizing the oneHotEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_oneHot_minMax = scaler.fit_transform(df_oneHot)
df_oneHot_minMax = pd.DataFrame(df_oneHot_minMax, columns=df_oneHot.columns)
print(df_oneHot_minMax.head(10))

# Normalizing the labelEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_label_minMax = scaler.fit_transform(df_label)
df_label_minMax = pd.DataFrame(df_label_minMax, columns=df_label.columns)
print(df_label_minMax.head(10))

# Normalizing the ordinalEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_ordinal_stand = scaler.fit_transform(df_ordinal)
df_ordinal_stand = pd.DataFrame(df_ordinal_stand, columns=df_ordinal.columns)
print(df_ordinal_stand.head(10))

# Normalizing the oneHotEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_oneHot_stand = scaler.fit_transform(df_oneHot)
df_oneHot_stand = pd.DataFrame(df_oneHot_stand, columns=df_oneHot.columns)
print(df_oneHot_stand.head(10))

# Normalizing the labelEncoded dataset using StandardScaler
scaler = preprocessing.StandardScaler()
df_label_stand = scaler.fit_transform(df_label)
df_label_stand = pd.DataFrame(df_label_stand, columns=df_label.columns)
print(df_label_stand.head(10))

"""
#show result of MaxAbs scaling EngineHP and credit history
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(EngineHp-credit history)')
sns.kdeplot(df_label['EngineHP'],ax=ax1)
sns.kdeplot(df_label['credit_history'],ax=ax1)
ax2.set_title('After_scaling(EngineHp-credit history)')
sns.kdeplot(df_label_minMax['EngineHP'],ax=ax2)
sns.kdeplot(df_label_minMax['credit_history'],ax=ax2)
plt.show()
#show result of MaxAbs scaling Years_Experience and annual_claims
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(Years_Experience-annual_claims)')
sns.kdeplot(df_label['Years_Experience'],ax=ax1)
sns.kdeplot(df_label['annual_claims'],ax=ax1)
ax2.set_title('After_scaling(Years_Experience-annual_claims)')
sns.kdeplot(df_label_minMax['Years_Experience'],ax=ax2)
sns.kdeplot(df_label_minMax['annual_claims'],ax=ax2)
plt.show()
#show result of MaxAbs scaling Gender and Marital Status
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(Gender-Marital_Status)')
sns.kdeplot(df_label['Gender'],ax=ax1)
sns.kdeplot(df_label['Marital_Status'],ax=ax1)
ax2.set_title('After_scaling(Gender-Marital_Status)')
sns.kdeplot(df_label_minMax['Gender'],ax=ax2)
sns.kdeplot(df_label_minMax['Marital_Status'],ax=ax2)
plt.show()
#show result of MaxAbs scaling size_of_family and Miles_driven_annually
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(6,5))
ax1.set_title('Before_scaling(size_of_family-Miles_driven_annually)')
sns.kdeplot(df_label['size_of_family'],ax=ax1)
sns.kdeplot(df_label['Miles_driven_annually'],ax=ax1)
ax2.set_title('After_scaling(size_of_family-Miles_driven_annually)')
sns.kdeplot(df_label_minMax['size_of_family'],ax=ax2)
sns.kdeplot(df_label_minMax['Miles_driven_annually'],ax=ax2)
plt.show()
#bar graph of target
sns.countplot(df['target'])
plt.show()
#bar graph of Gender
sns.countplot(df['Gender'])
plt.show()
#bar graph of annual_claims
sns.countplot(df['annual_claims'])
plt.show()
#bar graph of Marital_Status
sns.countplot(df['Marital_Status'])
plt.show()
#bar graph of Vehical_type
sns.countplot(df['Vehical_type'])
#plt.show()
#bar graph of size_of_family
sns.countplot(df['size_of_family'])
#plt.show()
#bar graph of State
sns.countplot(df['State'])
plt.show()
#Two variable plots
df_f = pd.DataFrame(df[df['Gender']=='F'])
df_m = pd.DataFrame(df[df['Gender']=='M'])
#EngineHP histogram by Gender
plt.hist(df_f['EngineHP'], bins=10)
plt.title('Female')
plt.xlabel('EngineHP')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_m['EngineHP'], bins=10)
plt.title('Male')
plt.xlabel('EngineHP')
plt.ylabel('Frequency')
plt.show()
#credit history histogram by Gender
plt.hist(df_m['credit_history'], bins=10)
plt.title('Male')
plt.xlabel('credit_history')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_f['credit_history'], bins=10)
plt.title('Female')
plt.xlabel('credit_history')
plt.ylabel('Frequency')
plt.show()
#years exprience histogram by Gender
plt.hist(df_f['Years_Experience'], bins=10)
plt.title('Female')
plt.xlabel('Years_Experience')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_m['Years_Experience'], bins=10)
plt.title('Male')
plt.xlabel('Years_Experience')
plt.ylabel('Frequency')
plt.show()
#annual claims histogram by Gender
plt.hist(df_m['annual_claims'], bins=10)
plt.title('Male')
plt.xlabel('annual_claims')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_f['annual_claims'], bins=10)
plt.title('Female')
plt.xlabel('annual_claims')
plt.ylabel('Frequency')
plt.show()
#Miles driven annually histogram by Gender
plt.hist(df_f['Miles_driven_annually'], bins=10)
plt.title('Female')
plt.xlabel('Miles_driven_annually')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_m['Miles_driven_annually'], bins=10)
plt.title('Male')
plt.xlabel('Miles_driven_annually')
plt.ylabel('Frequency')
plt.show()
#Size of family histogram by Gender
plt.hist(df_f['size_of_family'], bins=10)
plt.title('Female')
plt.xlabel('size_of_family')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_m['size_of_family'], bins=10)
plt.title('Male')
plt.xlabel('size_of_family')
plt.ylabel('Frequency')
plt.show()
#State histogram by Gender
plt.hist(df_f['State'], bins=10)
plt.title('Female')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.show()
plt.hist(df_m['State'], bins=10)
plt.title('Male')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.show()
#Pie chart of vehicle type
car = pd.DataFrame(df[df['Vehical_type']=='Car'])
van = pd.DataFrame(df[df['Vehical_type']=='Van'])
truck = pd.DataFrame(df[df['Vehical_type']=='Truck'])
utility = pd.DataFrame(df[df['Vehical_type']=='Utility'])
langs = ['Car','Van','Truck','Utility']
vehicle_level = [len(car), len(van), len(truck),len(utility)]
plt.pie(vehicle_level,labels=langs,autopct='%1.2f%%')
plt.show()
#Pie chart of annual_claim
ann_0 = pd.DataFrame(df[df['annual_claims']==0])
ann_1 = pd.DataFrame(df[df['annual_claims']==1])
ann_2 = pd.DataFrame(df[df['annual_claims']==2])
ann_3 = pd.DataFrame(df[df['annual_claims']==3])
ann_4 = pd.DataFrame(df[df['annual_claims']==4])
langs = ['0','1','2','3','4']
claim_level = [len(ann_0), len(ann_1), len(ann_2),len(ann_3),len(ann_4)]
plt.pie(claim_level,labels=langs,autopct='%1.2f%%')
plt.title('annual claim levels')
plt.show()
#Pie chart of size of family
fam1 = pd.DataFrame(df[df['size_of_family']==1])
fam2 = pd.DataFrame(df[df['size_of_family']==2])
fam3 = pd.DataFrame(df[df['size_of_family']==3])
fam4 = pd.DataFrame(df[df['size_of_family']==4])
fam5 = pd.DataFrame(df[df['size_of_family']==5])
fam6 = pd.DataFrame(df[df['size_of_family']==6])
fam7 = pd.DataFrame(df[df['size_of_family']==7])
fam8 = pd.DataFrame(df[df['size_of_family']==8])
langs = ['1','2','3','4','5','6','7','8']
family_level = [len(fam1), len(fam2), len(fam3),len(fam4),len(fam5),len(fam6),len(fam7),len(fam8)]
plt.pie(family_level,labels=langs,autopct='%1.2f%%')
plt.title('famliy size levels')
plt.show()
#EngineHP and crdit history scatter plot divided by Gender
plt.scatter(df_f['EngineHP'], df_f['credit_history'],color='r')
plt.scatter(df_m['EngineHP'], df_m['credit_history'],color='b')
plt.xlabel('EngineHP')
plt.ylabel('Credit history')
plt.title('Scatter plot divided by Gender')
plt.show()
#Years_Experience and annual claims scatter plot divided by Gender
plt.scatter(df_f['Years_Experience'], df_f['annual_claims'],color='r')
plt.scatter(df_m['Years_Experience'], df_m['annual_claims'],color='b')
plt.xlabel('Years_Experience')
plt.ylabel('Annual claims ')
plt.title('Scatter plot divided by Gender')
plt.show()
#Boxplot of annual claims divided by gender
sns.boxplot(df['Gender'],df['annual_claims'])
plt.show()
#Boxplot of EngineHP divided by gender
sns.boxplot(df['Gender'],df['EngineHP'])
plt.show()
#Boxplot of Years Experience divided by gender
sns.boxplot(df['Gender'],df['Years_Experience'])
plt.show()
#Boxplot of size of family divided by gender
sns.boxplot(df['Gender'],df['size_of_family'])
plt.show()
#heatmap-pearson
sns.heatmap(df.corr(method='pearson'))
plt.title("pearson")
plt.show()
"""



df2 = df_label_stand.copy()
# Rename 'target' and 'annual_claims' features
df2.rename(columns = {'target':'claim_prediction', 'annual_claims':'target'}, inplace=True)
# Drop 'ID' feature
df2.drop(['ID'], 1, inplace=True)

# Split the dataset
X = df2.drop(['target'], 1)
y = targets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = BaggingClassifier() #grid search 해야함
params = {'n_estimators': [100,125,150],
              'max_features': [0.1,0.4, 0.5,1],
              'max_samples':[0.1, 0.2, 0.3,0.5,1]
          };

print("Do bagging regressor grid search")
model_gscv = GridSearchCV(model,param_grid = params,cv=5,scoring='accuracy')
model_gscv.fit(X_train,y_train)
print("Best param : ",model_gscv.best_params_)
print("Best score : ",model_gscv.best_score_)
prediction = model_gscv.predict(X_test)
print(model_gscv.score(X_test,y_test))


# Using KNN algorithm
# Create and train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_prec = knn.predict(X_test)
print("\n---------- KNN algorithm ----------")
print(y_prec[0:100])
print("Score: %.2f" % knn.score(X_test, y_test))


rfModel = RandomForestRegressor()
params = {'n_estimators': [100,125,150],
              'max_depth': [2,4,6,8],
              'max_features': [0.1,0.4, 0.5,1],
              'max_samples':[0.1, 0.2, 0.3,0.5,1]
          };
print("Do randomforest regressor grid search")
rfModel_gscv = GridSearchCV(rfModel,params,scoring = 'r2')
rfModel_gscv.fit(X_train,y_train)
y_predict = rfModel_gscv.predict(X_test)
print("Best param : ",rfModel_gscv.best_params_)
print("Best score : ",rfModel_gscv.best_score_)
print(rfModel_gscv.score(X_test,y_test))
print("\n\n\n")

# LinearRegression
line_reg = LinearRegression();
line_reg.fit(X_train, y_train)
y_predict = line_reg.predict(X_test)

print("y_predict: \n", y_predict)
print("Score: %.2f" % line_reg.score(X_test, y_test))

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
pol_reg = LinearRegression()
pol_reg.fit(X_poly_train, y_train)
y_predict = line_reg.predict(X_test)

print("y_predict: \n", y_predict)
print("Score: %.2f" % pol_reg.score(X_poly_test, y_test))




# K-mean clustering
df3 = df_label.copy()
# Rename 'target' and 'annual_claims' features
df3.rename(columns = {'target':'claim_prediction', 'annual_claims':'target'}, inplace=True)
# Drop 'ID' feature
df3.drop(['ID'], 1, inplace=True)

# Split the dataset
X_3 = df3.drop(['target'], 1).astype(float)
y_3 = df3['target']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_3, y_3, random_state=0)

X_train3 = np.array(X_train3)
X_test3 = np.array(X_test3)
y_test3 = np.array(y_test3)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train3)
correct = 0
for i in range(len(X_test3)):
    predict_me = np.array(X_test3[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = kmeans.predict(predict_me)
    if(prediction[0]==y_test3[i]):
        correct += 1
print("Score: %.2f" % (correct/len(X_test3)))





#Make own module to predict
def process_module(df, targetName):
   maxAbsScaler = preprocessing.MaxAbsScaler()
   minmaxScaler = preprocessing.MinMaxScaler()
   robustScaler = preprocessing.RobustScaler()
   standardScaler = preprocessing.StandardScaler()
   
   df_maxAbs_scaled = maxAbsScaler.fit_transform(df)
   df_maxAbs_scaled = pd.DataFrame(df_maxAbs_scaled, columns=df.columns)
   
   df_minMax_scaled = minmaxScaler.fit_transform(df)
   df_minMax_scaled = pd.DataFrame(df_minMax_scaled, columns=df.columns)
   
   df_robust_scaled = robustScaler.fit_transform(df)
   df_robust_scaled = pd.DataFrame(df_robust_scaled, columns=df.columns)
   
   df_standard_scaled = standardScaler.fit_transform(df)
   df_standard_scaled = pd.DataFrame(df_standard_scaled, columns=df.columns)

   print("\n------------------------- Using maxAbs scaled dataset -------------------------")
   max_score_maxAbs = algorithm_model(df_maxAbs_scaled, targetName)
   print("\n------------------------- Using minMax scaled dataset -------------------------")
   max_score_minMax = algorithm_model(df_minMax_scaled, targetName)
   print("\n------------------------- Using robust scaled dataset -------------------------")
   max_score_robust = algorithm_model(df_robust_scaled, targetName)
   print("\n------------------------- Using standard scaled dataset -------------------------")
   max_score_standard = algorithm_model(df_standard_scaled, targetName)

   max_score_result = max(max_score_maxAbs, max_score_minMax, max_score_robust, max_score_standard)
   print("\n\n============================== Result ==============================")
   print("Final maximum score: %.6f" % max_score_result)


def algorithm_model(df, targetName):
    # Split the dataset
    X = df.drop([targetName], 1)
    y = df[targetName]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Linear Regression
    line_reg = LinearRegression()
    line_reg.fit(X_train, y_train)
    y_prec_linear = line_reg.predict(X_test)
    score_linear = line_reg.score(X_test, y_test)
    print("\ny_predict_linear: \n", y_prec_linear[0:50])
    print("Score: %.6f" % score_linear)

    # Polynomial Regression
    poly_reg = PolynomialFeatures(degree=2)
    X_poly_train = poly_reg.fit_transform(X_train)
    X_poly_test = poly_reg.fit_transform(X_test)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly_train, y_train)
    y_prec_poly = line_reg.predict(X_test)
    score_poly = pol_reg.score(X_poly_test, y_test)
    print("\ny_predict_poly: \n", y_prec_poly[0:50])
    print("Score: %.6f" % score_poly)

    # KNN algorithm
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_prec_knn = knn.predict(X_test)
    score_knn = knn.score(X_test, y_test)
    print("\ny_predict_KNN: \n", y_prec_knn[0:50])
    print("Score: %.6f" % score_knn)

    # Random Forest
    random_forest = RandomForestRegressor(max_depth=4, random_state=0)
    random_forest.fit(X_train, y_train)
    y_predict_rf = random_forest.predict(X_test)
    score_rf = random_forest.score(X_test, y_test)
    print("\ny_predict_RF: \n", y_predict_rf[0:50])
    print("Score: %.6f" % score_rf)

    max_score = max(score_linear, score_poly, score_knn, score_rf)
    return max_score


# Test our model using the ordinal encoded dataset
df_test_model = df_ordinal.copy()
# Rename 'target' and 'annual_claims' features
df_test_model.rename(columns = {'target':'claim_prediction', 'annual_claims':'target'}, inplace=True)
# Drop 'ID' feature
df_test_model.drop(['ID'], 1, inplace=True)

print("\n\n============================== Using own module ==============================")
process_module(df_test_model, 'target')
