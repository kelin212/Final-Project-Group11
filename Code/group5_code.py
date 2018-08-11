
# %%-----------------------------------------------------------------------
# Import packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
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
# Drop rows with missing values
data=data.dropna()
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
print(data["TARGET"].value_counts())
print('\n')

# %%-----------------------------------------------------------------------
# Plot the data
# Plot group means for numerical variables
def num_var_plot(var):
    graphdata=data.groupby('TARGET')[var].mean().reset_index()
    target = {0:"No default", 1:"Default"}
    graphdata['TARGET'].replace(target, inplace=True)
    print(graphdata)
    x=graphdata['TARGET']
    y=graphdata[var]
    plt.bar(x, y, align='center',width=0.6,color='navy')
    plt.xticks(x)
    plt.ylabel(var)
    plt.title(var+' BY TARGET')
    plt.rc('font',family='Times New Roman')
    plt.show()

num_var_plot('AMT_CREDIT')
num_var_plot('AMT_INCOME_TOTAL')
num_var_plot('AMT_ANNUITY')
num_var_plot('AMT_GOODS_PRICE')

# Plot for categorical variables
def cat_var_plot(var):
    data['freq']=1
    graphdata=data.groupby(['TARGET',var])['freq'].count().reset_index()
    target = {0:"No default", 1:"Default"}
    graphdata['TARGET'].replace(target, inplace=True)
    graphdata=graphdata.pivot(index='TARGET',columns=var,values='freq')
    print(graphdata)
    #x=graphdata['TARGET']
    #y=graphdata[var]
    #plt.bar(x, y, align='center',width=0.6,color='navy')
    #plt.xticks(x)
    #plt.ylabel(y)
    #plt.title(var+' BY TARGET')
    #plt.rc('font',family='Times New Roman')
    #plt.show()

cat_var_plot('NAME_CONTRACT_TYPE')

# %%-----------------------------------------------------------------------
# Data pre-processing
# Normalize the numerical variables
nor_columns=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']
for var in nor_columns:
    data[var] = (data[var] - data[var].min()) / (data[var].max() - data[var].min())
    print(data[var].describe())

# Encode all the categorical variables
obj_columns = data.select_dtypes(include=['object']).columns
print(obj_columns)
data[obj_columns] = data[obj_columns].astype('category')
data[obj_columns] = data[obj_columns].apply(lambda x: x.cat.codes)
print(data.dtypes)

# %%-----------------------------------------------------------------------
# split the dataset
# separate the target variable
x = data.values[:, 2:10]
y = data.values[:, 1]

# encloding the class with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(y)

# split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# %%-----------------------------------------------------------------------
# perform training: SVC
# creating the classifier object
clf = SVC(kernel="linear")
# performing training
clf.fit(x_train, y_train)
# make predictions
# predicton on test
y_pred = clf.predict(x_test)

# calculate metrics
print("\n")
print("Classification Report")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy: ",accuracy_score(y_test,y_pred)*100)
print("\n")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['TARGET'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

# Plot ROC Area Under Curve
y_pred_proba = clf.decision_function(x_test)
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Loan Default')
plt.legend(loc="lower right")
plt.show()

# %%-----------------------------------------------------------------------
# perform training:


