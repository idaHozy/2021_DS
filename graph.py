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

pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)

# Read csv
df = pd.read_excel('IT_3.xlsx')
# Drop 5 columns
df = df.drop(['Age_bucket','EngineHP_bucket','Years_Experience_bucket'
                 ,'Miles_driven_annually_bucket','credit_history_bucket'], axis=1)

# Fill missing values
df.fillna(axis=0, method='ffill',inplace=True)

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


##################


# Normalizing the labelEncoded dataset using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
df_label_minMax = scaler.fit_transform(df_label)
df_label_minMax = pd.DataFrame(df_label_minMax, columns=df_label.columns)
print(df_label_minMax.head(10))

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

