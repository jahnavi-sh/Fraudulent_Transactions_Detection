#import the libraries 

#numericla python for building arrays 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import imblearn
from imblearn.under_sampling import NearMiss

#load the dataset 
df = pd.read_csv(r'C:\Users\jahna\Downloads\Fraud.csv')

#view the dataset 
#Exploratory data analysis 

#view total number of rows and columns 
df.shape

#there are 6362620 rows and 11 columns 

#view the top five rows of the dataset 
df.head()

#exploratory data analysis 
print (df.isnull().sum())

#the dataset contains zero missing values 

#get more information about the dataset 
df.info()

#there are 7 columns of integer data type and 3 columns of object data type 

#view the statistical measures 
df.describe()

#check for missing values 
df.isnull().sum()

#There are no mmissing values 

#checking for duplicated values 
df[df.duplicated()]

#There are no duplicate values 

#dropping the unnecessary column that are not required 
df.drop(columns=['nameOrig','nameDest'],inplace=True)

#checking for outliers 
sns.boxenplot(df['amount'])
plt.ylabel('distribution')
plt.show()

sns.boxenplot(df['oldbalanceOrg'])
plt.ylabel('distribution')

sns.boxenplot(df['newbalanceOrig'])
plt.ylabel('distribution')

sns.boxenplot(df['oldbalanceDest'])
plt.ylabel('distribution')

sns.boxenplot(df['newbalanceDest'])
plt.ylabel('distribution')

#Removing the outliers 
def remove_outliers(df,col):
    lower_quantile = df[col].quantile(0.25)
    upper_quantile = df[col].quantile(0.75)
    IQR = upper_quantile - lower_quantile
    lower_whisker = lower_quantile - 1.5 * IQR
    upper_whisker = upper_quantile + 1.5 * IQR
    temp = df.loc[(df[col]>lower_whisker)&(df[col]<upper_whisker)]
    return temp[col]

df['amount'] = remove_outliers(df,'amount')
df['oldbalanceOrg'] = remove_outliers(df,'oldbalanceOrg')
df['newbalanceOrig'] = remove_outliers(df,'newbalanceOrig')
df['oldbalanceDest'] = remove_outliers(df,'oldbalanceDest')
df['newbalanceDest'] = remove_outliers(df,'newbalanceDest')

#data visualisation 

#checking correlation between the features 
df.corr()
sns.heatmap(df.corr(),annot=True,cmap='plasma')

#From the heatmap, it is seen that the transaction recipient's old and new balances are strongly positively correlated with each other. 

df.groupby('isFraud').describe()

values = df['type'].value_counts().values
labels = df['type'].value_counts().keys()
explode = (0.1,0,0,0,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.show()

#The vast majority of transactions are of the type 'CASH_OUT' - overall proportion of more than one-third
#Next, 'Payment mode' - 34%
#'CASH_IN' - one-fifth 
#debit and normal transfer transactions - less than one-tenth 

values = df['isFraud'].value_counts().values
labels = ['Not Fraud','Fraud']
explode = (0.1,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.show()

#only tiny fraction is fraudulent transactions

#getting the exact value

legit_transaction = len(df[df.isFraud == 0])
fraud_transaction = len(df[df.isFraud == 1])
legit_transaction_percent = (legit_transaction/(fraud_transaction+legit_transaction))*100
fraud_transaction_percent = (fraud_transaction/(fraud_transaction+legit_transaction))*100

print("Number of Legitimate transactions: ", legit_transaction)
print("Number of Fraudulent transactions: ", fraud_transaction)
print("Percentage of Legitimate transactions: {:.4f} %".format(legit_transaction_percent))
print("Percentage of Fraudulent transactions: {:.4f} %".format(fraud_transaction_percent))

#Output - 
#Percentage of Legitimate transactions: 99.8709 %
#Percentage of Fraudulent transactions: 0.1291 %

#The results show that the data is highly unbalanced. The percentage of fraud transactions is barely 0.13%

#Let's find the maximum transferred amount by type 
max_amount_type = df.groupby('type')['amount'].max().sort_values(ascending=False).reset_index()[:10]
max_amount_type

#visualizing
sns.barplot(x='type',y='amount',data=max_amount_type,palette='magma')

#According to the graph, highest amount was transferred through normal transfer mode 
#least amount was transferred by payment. 

sns.countplot(df['isFraud'])
#imbalanced dataset because almost all the samples belong to the majority class label 'Not Fraud'

sns.distplot(df['amount'],bins=50)
#The graph as positive right skewed distribution. Need to perform normalization on the variables 

positive_fraud_case = df[df['isFraud']==1]
sns.distplot(positive_fraud_case['amount'],bins=50)

non_fraud_case = df[df['isFraud']==0]
sns.distplot(non_fraud_case['amount'],bins=50)

sns.regplot(x='oldbalanceDest',y='newbalanceDest',data=df.sample(100000))

#Min - max normalization 
df['amount'].fillna(df['amount'].mean(),inplace=True)
df['oldbalanceOrg'].fillna(df['oldbalanceOrg'].mean(),inplace=True)
df['newbalanceOrig'].fillna(df['newbalanceOrig'].mean(),inplace=True)
df['oldbalanceDest'].fillna(df['oldbalanceDest'].mean(),inplace=True)
df['newbalanceDest'].fillna(df['newbalanceDest'].mean(),inplace=True)

payment_types = pd.get_dummies(df['type'],prefix='type',drop_first=True)
df = pd.concat([df,payment_types],axis=1)
df.head()

df.drop('type',axis=1,inplace=True)

df['type_CASH_OUT'] = df['type_CASH_OUT'].astype(np.int64)
df['type_DEBIT'] = df['type_DEBIT'].astype(np.int64)
df['type_PAYMENT'] = df['type_PAYMENT'].astype(np.int64)
df['type_TRANSFER'] = df['type_TRANSFER'].astype(np.int64)

x = df.drop('isFraud',axis=1)
y = df['isFraud']

nm = NearMiss()
x_nm, y_nm = nm.fit_resample(x,y)

#Usually for imbalanced classes, ensemble machine learning algorithms such as Decision Tree and Random Forest work well. 

#train test split 
X = x_nm
y = y_nm
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,stratify=y,random_state=2)

X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

#Model building and evaluation 

#Logistic Regression 
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

print('ROC AUC Score:',roc_auc_score(y_test,lr_pred))
print('F1 Score:',f1_score(y_test,lr_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,lr_pred))
print('Classification Report:\n',classification_report(y_test,lr_pred))
print('Accuracy Score:',accuracy_score(y_test,lr_pred))

#Random Forest Classifier 
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

print("Confusion Matrix:\n",confusion_matrix(y_test,rfc_pred))
print("Classification Report:\n",classification_report(y_test,rfc_pred))
print("ROC AUC Score:",roc_auc_score(y_test,rfc_pred))
print("F1 Score:",f1_score(y_test,rfc_pred))
print('Accuracy Score:',accuracy_score(y_test,rfc_pred))

#Decision Tree classifier 
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_pred = dtree.predict(X_test)
dtree_pred

print("ROC AUC Score:",roc_auc_score(y_test,dtree_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,dtree_pred))
print("F1 Score:",f1_score(y_test,dtree_pred))
print("Classification Report:\n",classification_report(y_test,dtree_pred))
print("Accuracy Score:",accuracy_score(y_test,dtree_pred))

#Gaussian naive bayes 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
gnb_pred

print("ROC AUC Score:",roc_auc_score(y_test,gnb_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,gnb_pred))
print("F1 Score:",f1_score(y_test,gnb_pred))
print("Classification Report:\n",classification_report(y_test,gnb_pred))
print("Accuracy Score:",accuracy_score(y_test,gnb_pred))

#K-nearest Neighbour classifier 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

print("ROC AUC Score:",roc_auc_score(y_test,knn_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,knn_pred))
print("F1 Score:",f1_score(y_test,knn_pred))
print("Classification Report:\n",classification_report(y_test,knn_pred))
print("Accuracy Score:",accuracy_score(y_test,knn_pred))

#Support vector machine classifier 
svm = SVC()
svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)

print("ROC AUC Score:",roc_auc_score(y_test,svm_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,svm_pred))
print("F1 Score:",f1_score(y_test,svm_pred))
print("Classification Report:\n",classification_report(y_test,svm_pred))
print("Accuracy Score:",accuracy_score(y_test,svm_pred))

#XGBoost classifier 
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)

print("ROC AUC Score:",roc_auc_score(y_test,xgb_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,xgb_pred))
print("F1 Score:",f1_score(y_test,xgb_pred))
print("Classification Report:\n",classification_report(y_test,xgb_pred))
print("Accuracy Score:",accuracy_score(y_test,xgb_pred))