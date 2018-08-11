
# %%-----------------------------------------------------------------------
# Import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %%-----------------------------------------------------------------------
# Import the data
data=pd.read_csv("application_train.csv")
print(data.head())
print(data.isnull().sum())
print('\n')
data.info()
print('\n')
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])
print('\n')
#print(data.describe(include='all'))
print('\n')
print(list(data))
print('\n')
print(data.dtypes)
print('\n')

# %%-----------------------------------------------------------------------
# Plot the data
# Plot group means for numerical variables
def num_var_plot(var):
    graphdata=data.groupby('TARGET')[var].mean().reset_index()
    print(graphdata)
    x=graphdata['TARGET']
    y=graphdata[var]
    plt.bar(x, y, align='center', alpha=0.5,color='navy')
    plt.xlabel('TARGET')
    plt.ylabel(var)
    plt.title(var+' BY TARGET')
    plt.rc('font',family='Cambria')
    plt.show()

num_var_plot('AMT_CREDIT')

# %%-----------------------------------------------------------------------
# Data pre-processing
# Encode the categorical variables
obj_columns = data.select_dtypes(include=['object']).columns
print(obj_columns)
data[obj_columns] = data[obj_columns].astype('category')
data[obj_columns] = data[obj_columns].apply(lambda x: x.cat.codes)
print(data.dtypes)


# %%-----------------------------------------------------------------------
# split the dataset
# separate the target variable
x = data.values[:, 2:]
y = data.values[:, 1]

# encloding the class with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(y)

# split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# %%-----------------------------------------------------------------------
# perform training
# SVC
# creating the classifier object
clf = SVC(kernel="linear")
# performing training
clf.fit(x_train, y_train)
# make predictions
# predicton on test
y_pred = clf.predict(x_test)



