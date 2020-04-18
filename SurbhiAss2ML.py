#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Assignment 2 ####

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as Sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.metrics import accuracy_score , confusion_matrix 
from xgboost import XGBClassifier
import time


# # Dataset 1 Exploratory Data Analysis

# In[28]:


df1 = pd.read_csv('sgemm_product.csv')


# In[31]:


df1.head()


# In[29]:


df1["avg_run"] = df1.iloc[:,14:].mean(axis=1)
df1["log_avg_run"] = np.log(df1.avg_run) 
df1.drop(['Run1 (ms)', 'Run2 (ms)','Run3 (ms)', 'Run4 (ms)'], axis = 1, inplace = True)


# In[4]:


df1.head()


# In[34]:


df1.describe()


# In[30]:


df1.drop(['avg_run'], axis = 1, inplace = True)


# In[31]:


df1.log_avg_run.median()


# In[32]:


true_false = df1.log_avg_run > 4.245490
values = true_false.value_counts()
print(values)


# In[10]:


corrmat = df1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,13))
g=Sb.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[33]:


# Find outliers
plt.rcParams['figure.figsize'] = (12, 6)
Sb.boxplot(x="log_avg_run", data=df1)
plt.xlabel('Log of Average Run Time', fontsize=12)
plt.title("Average Run Time distribution", fontsize=15)


# In[34]:


#Deleting outliers
df1 = df1[df1['log_avg_run'] <= 8]


# In[35]:


X = df1.iloc[:,1:14]
Y_org = df1['log_avg_run']


# In[36]:


Y = Y_org.apply(lambda x : 0 if x <= 4.245490 else 1)
Y.value_counts()


# In[169]:


#Checking for missing values
df1.isnull().sum()


# # Dataset 2 Exploratory Data Analysis

# In[2]:


df2 = pd.read_csv('adult.csv')


# In[3]:


df2.head()


# In[13]:


df2.describe()


# In[4]:


df2['Salary'].value_counts()


# In[45]:


corrmat = df2.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(8,6))
g=Sb.heatmap(df2[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[120]:


Sb.pairplot(df2,hue='Salary',palette='Set1')


# In[5]:


X_adt = df2.iloc[:,0:14]
Y_adt = df2['Salary']


# In[6]:


Y_adt.shape


# In[171]:


#Checking for missing values
df2.isnull().sum()


# In[10]:


X_adt


# In[7]:


# Finding categorical data columns
obj_df = df2.select_dtypes(include=['object']).copy()
obj_df.head()


# In[8]:


obj_df[obj_df.isnull().any(axis=1)]


# In[10]:


# Encoding the categorical columns
obj_df["Work class"] = obj_df["Work class"].astype('category')
obj_df["WorkClass_cat"] = obj_df["Work class"].cat.codes
obj_df["Education"] = obj_df["Education"].astype('category')
obj_df["Education_cat"] = obj_df["Education"].cat.codes
obj_df["Marital status"] = obj_df["Marital status"].astype('category')
obj_df["MaritalStatus_cat"] = obj_df["Marital status"].cat.codes
obj_df["Occupation"] = obj_df["Occupation"].astype('category')
obj_df["Occupation_cat"] = obj_df["Occupation"].cat.codes
obj_df["Relationship"] = obj_df["Relationship"].astype('category')
obj_df["Relationship_cat"] = obj_df["Relationship"].cat.codes
obj_df["Race"] = obj_df["Race"].astype('category')
obj_df["Race_cat"] = obj_df["Race"].cat.codes
obj_df["Sex"] = obj_df["Sex"].astype('category')
obj_df["Sex_cat"] = obj_df["Sex"].cat.codes
obj_df["native-country"] = obj_df["native-country"].astype('category')
obj_df["NativeCountry_cat"] = obj_df["native-country"].cat.codes


# In[11]:


#Adding the new encoded columns to dataset
df2['WorkClass_cat'] = obj_df['WorkClass_cat']
df2['Education_cat'] = obj_df['Education_cat']
df2['MaritalStatus_cat'] = obj_df['MaritalStatus_cat']
df2['Occupation_cat'] = obj_df['Occupation_cat']
df2['Relationship_cat'] = obj_df['Relationship_cat']
df2['Race_cat'] = obj_df['Race_cat']
df2['Sex_cat'] = obj_df['Sex_cat']
df2['NativeCountry_cat'] = obj_df['NativeCountry_cat']
df2.drop(['Work class','Education','Marital status','Occupation','Relationship','Race','Sex','native-country'], axis = 1, inplace = True)


# In[12]:


df2.head()
lst = ['Age','fnlwgt','Edu-num','capital-gain','capital-loss','hours-per-week','WorkClass_cat','Education_cat','MaritalStatus_cat','Occupation_cat','Relationship_cat','Race_cat','Sex_cat','NativeCountry_cat','Salary']
A_df = df2[lst]
A_df.head()


# In[13]:


X_adt = A_df.iloc[:,0:14]
Y_adt = A_df['Salary']
X_adt.head()


# # SVM for Dataset 1

# In[40]:


# scaling for SVM
X_sc = X.copy()
std_sc = StandardScaler()
X_sc = std_sc.fit_transform(X_sc)


# In[32]:


X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X_sc,Y,test_size = 0.6,random_state = 0)


# In[25]:


from sklearn.svm import SVC


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train_1,y_train_1)


# In[112]:


model_rbf = SVC(kernel='rbf', C=1, gamma = 0.1)
model_rbf.fit(X_train_1,y_train_1)


# In[113]:


predictions = model_rbf.predict(X_test_1)


# In[114]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test_1,predictions))


# In[115]:


print(classification_report(y_test_1,predictions))


# In[34]:


### Polynomial degree eperimentation
degree = np.arange(1,15,1)
acc_train_1 = []
acc_test_1 = []
for i in degree:
    poly_svm = SVC(kernel='poly', degree= i, random_state=20, gamma = 'scale')
    poly_svm.fit(X_train_1 , y_train_1)
    y_pred_train1 = poly_svm.predict(X_train_1)
    y_pred_test1 = poly_svm.predict(X_test_1)
    acc_train_1.append(accuracy_score(y_train_1,y_pred_train1))
    acc_test_1.append(accuracy_score(y_test_1,y_pred_test1))


# In[ ]:


## Plotting the graph for the different degrees of polynomial
plt.plot(degree,acc_train_1,color='orange',label='training')
plt.plot(degree,acc_test_1,color='red',label='test')
plt.title(" Accuracies for various degrees of polynomial ")
plt.xlabel("Degrees")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[ ]:


### developing the models
classifier_lin = SVC(kernel = 'linear', random_state = 20, gamma = 'scale')
classifier_lin.fit(X_train_1, y_train_1)

classifier_rbf = SVC(kernel = 'rbf', random_state = 20, gamma = 0.1, C = 1)
classifier_rbf.fit(X_train_1, y_train_1)

classifier_poly = SVC(kernel = 'poly', degree = 4, random_state = 20, gamma = 'scale')
classifier_poly.fit(X_train_1, y_train_1)


# In[ ]:


# Predicting the Test set results
y_pred_lin = classifier_lin.predict(X_test_1)
y_pred_rbf = classifier_rbf.predict(X_test_1)
y_pred_poly = classifier_poly.predict(X_test_1)


# In[ ]:


# Making the Confusion Matrix
cm_linear = confusion_matrix(y_test_1, y_pred_lin)
cm_rbf = confusion_matrix(y_test_1, y_pred_rbf)
cm_poly = confusion_matrix(y_test_1, y_pred_poly)

acc_linear = accuracy_score(y_test_1, y_pred_lin)
acc_rbf = accuracy_score(y_test_1, y_pred_rbf)
acc_poly = accuracy_score(y_test_1, y_pred_poly)


# In[ ]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_lin = cross_val_score(estimator = classifier_lin, X = X_train_1, y = y_train_1, cv = 10)
mean_acc_lin = accuracies_lin.mean()
accuracies_rbf = cross_val_score(estimator = classifier_rbf, X = X_train_1, y = y_train_1, cv = 10)
mean_acc_rbf = accuracies_rbf.mean()
accuracies_poly = cross_val_score(estimator = classifier_poly, X = X_train_1, y = y_train_1, cv = 10)
mean_acc_poly = accuracies_poly.mean()


# In[ ]:


# Plotting the mean accuracies found from cross validtion and test set accuracy
mean_acc = [mean_acc_lin, mean_acc_rbf,mean_acc_poly]
test_acc = [acc_linear,acc_rbf,acc_poly]
kernel_svm = ['Linear','Rbf','Polynomial']
plt.plot(kernel_svm , mean_acc, color ='red',label='k-fold')
plt.plot(kernel_svm, test_acc, color ='blue',label='Test accuracy')
plt.title(" Accuracies of mean Folds Vs Test accuracy ")
plt.xlabel("Kernels")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[ ]:


# Graph for different accuracies of different kernels
degree = np.arange(1,11,1)
plt.plot(degree, accuracies_lin, color = 'yellow',label='linear')
plt.plot(degree, accuracies_rbf, color = 'blue',label='rbf')
plt.plot(degree, accuracies_poly, color = 'red' , label='poly')
plt.title(" Accuracies of different Folds for differnet Kernels ")
plt.xlabel("Fold Number")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # SVM for Dataset 2

# In[18]:


X_sc_adt = X_adt.copy()
std_sc = StandardScaler()
X_sc_adt = std_sc.fit_transform(X_sc_adt)


# In[19]:


X_train_2,X_test_2,y_train_2,y_test_2 = train_test_split(X_sc_adt,Y_adt,test_size = 0.3,random_state = 0)


# In[20]:


from sklearn.svm import SVC


# In[121]:


from sklearn.model_selection import GridSearchCV


# In[122]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# In[123]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train_2,y_train_2)


# In[126]:


grid.best_params_


# In[128]:


svc_model = SVC(C=1, gamma= 0.01, kernel = 'rbf')


# In[129]:


svc_model.fit(X_train_2,y_train_2)


# In[131]:


svc_pred = svc_model.predict(X_test_2)


# In[133]:


acc_score_1 = accuracy_score(y_test_2 , svc_pred)
acc_score_1


# In[41]:


### Polynomial degree eperimentation
degree = np.arange(1,9,1)
acc_train_2 = []
acc_test_2 = []
for i in degree:
    poly_svm_adt = SVC(kernel='poly' , degree= i , random_state=0, gamma = 'scale')
    poly_svm_adt.fit(X_train_2 , y_train_2)
    y_pred_train_2 = poly_svm_adt.predict(X_train_2)
    y_pred_test_2 = poly_svm_adt.predict(X_test_2)
    acc_train_2.append(accuracy_score(y_train_2,y_pred_train_2))
    acc_test_2.append(accuracy_score(y_test_2,y_pred_test_2))


# In[42]:


## Plotting the graph for the different degrees of polynomial
plt.plot(degree,acc_train_2,color='orange',label='training')
plt.plot(degree,acc_test_2,color='red',label='test')
plt.title(" Accuracies for various degrees of polynomial ")
plt.xlabel("Degrees")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[21]:


### developing the models
classifier_lin = SVC(kernel = 'linear', random_state = 0, gamma = 'scale')
classifier_lin.fit(X_train_2, y_train_2)

classifier_rbf = SVC(kernel = 'rbf', random_state = 0, gamma = 0.01, C=1)
classifier_rbf.fit(X_train_2, y_train_2)

classifier_poly = SVC(kernel = 'poly', degree = 3, random_state = 0, gamma = 'scale')
classifier_poly.fit(X_train_2, y_train_2)


# In[22]:


# Predicting the Test set results
y_pred_lin = classifier_lin.predict(X_test_2)
y_pred_rbf = classifier_rbf.predict(X_test_2)
y_pred_poly = classifier_poly.predict(X_test_2)


# In[24]:


# Making the Confusion Matrix
cm_adt_linear = confusion_matrix(y_test_2, y_pred_lin)
cm_adt_rbf = confusion_matrix(y_test_2, y_pred_rbf)
cm_adt_poly = confusion_matrix(y_test_2, y_pred_poly)

acc_adt_linear = accuracy_score(y_test_2 , y_pred_lin)
acc_adt_rbf = accuracy_score(y_test_2 , y_pred_rbf)
acc_adt_poly = accuracy_score(y_test_2 , y_pred_poly)


# In[25]:


print(acc_adt_linear, acc_adt_rbf, acc_adt_poly)


# In[26]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_lin = cross_val_score(estimator = classifier_lin, X = X_train_2, y = y_train_2, cv = 10)
mean_acc_lin = accuracies_lin.mean()
accuracies_rbf = cross_val_score(estimator = classifier_rbf, X = X_train_2, y = y_train_2, cv = 10)
mean_acc_rbf = accuracies_rbf.mean()
accuracies_poly = cross_val_score(estimator = classifier_poly, X = X_train_2, y = y_train_2, cv = 10)
mean_acc_poly = accuracies_poly.mean()


# In[47]:


# Plotting the mean accuracies found from cross validtion and test set accuracy
mean_acc_adt = [mean_acc_lin, mean_acc_rbf,mean_acc_poly]
test_acc_adt = [acc_adt_linear,acc_adt_rbf,acc_adt_poly]
kernel_svm = ['Linear','Rbf','Polynomial']
plt.plot(kernel_svm , mean_acc_adt, color ='red',label='k-fold')
plt.plot(kernel_svm, test_acc_adt, color ='blue',label='Test accuracy')
plt.title(" Accuracies of mean Folds Vs Test accuracy ")
plt.xlabel("Kernels")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[48]:


# Graph for different accuracies of different kernels
degree = np.arange(1,11,1)
plt.plot(degree, accuracies_lin,marker = 'o' , markersize = 6 , color = 'skyblue',label='linear')
plt.plot(degree, accuracies_rbf,marker = 'o' , markersize = 6 , color = 'orange',label='rbf')
plt.plot(degree, accuracies_poly,marker = 'o' , markersize = 6 , color = 'red' , label='poly')
plt.title(" Accuracies of different Folds for differnet Kernels ")
plt.xlabel("Fold Number")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[49]:


print(mean_acc_lin, mean_acc_rbf,mean_acc_poly)


# # Decision Tree Classification for Dataset 1

# In[72]:


from sklearn.tree import DecisionTreeClassifier


# In[73]:


dtree = DecisionTreeClassifier()


# In[77]:


X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X,Y,test_size = 0.3,random_state = 0)


# In[78]:


dtree.fit(X_train_1,y_train_1)
predictions = dtree.predict(X_test_1)


# In[79]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_1,predictions))


# In[80]:


print(confusion_matrix(y_test_1,predictions))


# In[150]:


from sklearn.tree import DecisionTreeClassifier
depth = np.arange(2,20,1)
acc_train = []
acc_test = []
for i in depth:   
    dtree = DecisionTreeClassifier(criterion= "entropy", max_depth= i, random_state= 0 )
    dtree.fit(X_train_1 , y_train_1)
    y_pred_train = dtree.predict(X_train_1)
    y_pred_test = dtree.predict(X_test_1)
    acc_train.append(accuracy_score(y_train_1,y_pred_train))
    acc_test.append(accuracy_score(y_test_1,y_pred_test))


# In[151]:


plt.plot(depth,acc_train,color='orange',label='train')
plt.plot(depth,acc_test,color='blue',label='test')
plt.title(" Accuracies for different Depth ")
plt.xlabel("Depths")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[152]:


from sklearn.tree import DecisionTreeClassifier
acc_train = []
acc_test = [] 
dtree = DecisionTreeClassifier(criterion="entropy", max_depth= 10, random_state= 0)
dtree.fit(X_train_1 , y_train_1)
y_pred_train = dtree.predict(X_train_1)
y_pred_test = dtree.predict(X_test_1)
acc_train.append(accuracy_score(y_train_1,y_pred_train))
acc_test.append(accuracy_score(y_test_1,y_pred_test))


# In[153]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_1,y_pred_test))


# In[154]:


# Experiment pruning by changing the min_samples count
acc_train = []
acc_test = []
for i in np.arange(0,100,10):
    DTClas = DecisionTreeClassifier(criterion= "entropy" ,max_depth= 10,min_samples_leaf= i+1 ,  random_state= 0 )
    DTClas.fit(X_train_1 , y_train_1)
    y_pred_train = DTClas.predict(X_train_1)
    y_pred_test = DTClas.predict(X_test_1)
    acc_train.append(accuracy_score(y_train_1,y_pred_train))
    acc_test.append(accuracy_score(y_test_1,y_pred_test))


# In[155]:


plt.plot(np.arange(0,100,10),acc_train,color='orange',label='train')
plt.plot(np.arange(0,100,10),acc_test,color='red',label='test')
plt.title(" Accuracies for different Pruning Nodes ")
plt.xlabel("Pruning Count")
plt.grid()
plt.ylabel("Accuracy")
plt.legend() 


# In[156]:


### Final Decision Tree model
Dtree_final = DecisionTreeClassifier(criterion= "entropy" , random_state= 29 , max_depth=10, min_samples_leaf = 40)
Dtree_final.fit(X_train_1 , y_train_1)
y_pred_dt = Dtree_final.predict(X_test_1)


# In[157]:


print(confusion_matrix(y_test_1,y_pred_dt))
print(classification_report(y_test_1,y_pred_dt))


# In[158]:


### Implementing cross validation and finding the mean accuracy of folds
from sklearn.model_selection import cross_val_score
accuracy_1 = cross_val_score(estimator = Dtree_final, X = X_train_1, y = y_train_1, cv = 10)
mean_accuracy_1 = accuracy_1.mean()
mean_accuracy_1


# In[159]:


# Plotting the mean accuracies found from cross validtion and test set accuracy
plt.plot(np.arange(0,100,10),accuracy_1, color = 'red',label='Folds')
plt.title(" Accuracies of different Folds ")
plt.xlabel("Folds")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# # Decision Tree Classification for Dataset 2

# In[60]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
dtree_a = DecisionTreeClassifier()


# In[58]:


X_train_2,X_test_2,y_train_2,y_test_2 = train_test_split(X_sc_adt,Y_adt,test_size = 0.3,random_state = 0)


# In[61]:


dtree_a.fit(X_train_2,y_train_2)
predictions = dtree_a.predict(X_test_2)


# In[64]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_2,predictions))


# In[63]:


print(confusion_matrix(y_test_2,predictions))


# In[61]:


from sklearn.tree import DecisionTreeClassifier
depth_adt = np.arange(2,20,1)
acc_train_adt = []
acc_test_adt = []
for i in depth_adt:   
    dtree_adt = DecisionTreeClassifier(criterion= "entropy", max_depth= i, random_state= 0 )
    dtree_adt.fit(X_train_2, y_train_2)
    y_pred_train_2 = dtree_adt.predict(X_train_2)
    y_pred_test_2 = dtree_adt.predict(X_test_2)
    acc_train_adt.append(accuracy_score(y_train_2,y_pred_train_2))
    acc_test_adt.append(accuracy_score(y_test_2,y_pred_test_2))


# In[62]:


plt.plot(depth_adt,acc_train_adt,color='orange',label='train')
plt.plot(depth_adt,acc_test_adt,color='blue',label='test')
plt.title(" Accuracies for different Depth ")
plt.xlabel("Depths")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[65]:


from sklearn.tree import DecisionTreeClassifier
dtree_adt2 = DecisionTreeClassifier(criterion="entropy", max_depth= 9, random_state= 0)
dtree_adt2.fit(X_train_2 , y_train_2)
y_pred_train_2 = dtree_adt2.predict(X_train_2)
y_pred_test_2 = dtree_adt2.predict(X_test_2)


# In[66]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_2,y_pred_test_2))


# In[67]:


# Experiment pruning by changing the min_samples count
acc_train_adt2 = []
acc_test_adt2 = []
for i in np.arange(0,100,10):
    DTClas_adt2 = DecisionTreeClassifier(criterion= "entropy" ,max_depth= 9,min_samples_leaf= i+1 ,  random_state= 0 )
    DTClas_adt2.fit(X_train_2 , y_train_2)
    y_pred_train_adt2 = DTClas_adt2.predict(X_train_2)
    y_pred_test_adt2 = DTClas_adt2.predict(X_test_2)
    acc_train_adt2.append(accuracy_score(y_train_2,y_pred_train_adt2))
    acc_test_adt2.append(accuracy_score(y_test_2,y_pred_test_adt2))


# In[68]:


plt.plot(np.arange(0,100,10),acc_train_adt2,color='orange',label='train')
plt.plot(np.arange(0,100,10),acc_test_adt2,color='red',label='test')
plt.title(" Accuracies for different Pruning Nodes ")
plt.xlabel("Pruning Count")
plt.grid()
plt.ylabel("Accuracy")
plt.legend() 


# In[69]:


### Final Decision Tree model
Dtree_final_adt = DecisionTreeClassifier(criterion= "entropy" , random_state= 29 , max_depth=10, min_samples_leaf = 50)
Dtree_final_adt.fit(X_train_2 , y_train_2)
y_pred_finadt = Dtree_final_adt.predict(X_test_2)


# In[70]:


print(confusion_matrix(y_test_2,y_pred_finadt))
print(classification_report(y_test_2,y_pred_finadt))


# In[71]:


### Implementing cross validation and finding the mean accuracy of folds
from sklearn.model_selection import cross_val_score
accuracy_2 = cross_val_score(estimator = Dtree_final_adt, X = X_train_2, y = y_train_2, cv = 10)
mean_accuracy_2 = accuracy_2.mean()
mean_accuracy_2


# In[72]:


# Plotting the mean accuracies found from cross validtion and test set accuracy
plt.plot(np.arange(0,100,10),accuracy_2, color = 'green',label='Folds')
plt.title(" Accuracies of different Folds ")
plt.xlabel("Folds")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# # Boosting for Dataset 1

# In[51]:


from xgboost import XGBClassifier


# In[ ]:


# Experimentation with depth
depth = np.arange(2,20,1)
acc_train = []
acc_test = []
for i in depth:   
    BTree = XGBClassifier(max_depth= i, random_state= 20)
    BTree.fit(X_train_1, y_train_1)
    y_pred_train = BTree.predict(X_train_1)
    y_pred_test = BTree.predict(X_test_1)
    acc_train.append(accuracy_score(y_train_1,y_pred_train))
    acc_test.append(accuracy_score(y_test_1,y_pred_test))


# In[105]:


# Plotting depth vs accuracy rate of training and test    
plt.plot(depth,acc_train,color='orange',label='training')
plt.plot(depth,acc_test,color='red',label='test')
plt.title(" Accuracies for different Depth")
plt.xlabel("Depths")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[106]:


# Experimentation with boosting levels
from sklearn.ensemble import AdaBoostClassifier
boosting_level = np.arange(1,100,10)
acc_train_bst = []
acc_test_bst = []
for i in boosting_level:   
    Bst = AdaBoostClassifier(n_estimators = i, random_state= 20)
    Bst.fit(X_train_1 , y_train_1)
    y_pred_train_bst = Bst.predict(X_train_1)
    y_pred_test_bst = Bst.predict(X_test_1)
    acc_train_bst.append(accuracy_score(y_train_1,y_pred_train_bst))
    acc_test_bst.append(accuracy_score(y_test_1,y_pred_test_bst))


# In[107]:


# Plotting the graph of boosting level Vs accuracy
plt.plot(boosting_level,acc_train_bst,color='orange',label='training')
plt.plot(boosting_level,acc_test_bst,color='red',label='test')
plt.title(" Accuracies for different Boosting Level ")
plt.xlabel("Boosting Level")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[108]:


# Experimentation with learning rate parameter
learning_rate = np.arange(0.01,0.60,0.05)
acc_train = []
acc_test = []
for i in learning_rate:   
    BTree = XGBClassifier(learning_rate = i, random_state= 20)
    BTree.fit(X_train_1, y_train_1)
    y_pred_train = BTree.predict(X_train_1)
    y_pred_test = BTree.predict(X_test_1)
    acc_train.append(accuracy_score(y_train_1,y_pred_train))
    acc_test.append(accuracy_score(y_test_1,y_pred_test))


# In[109]:


# Plotting learning rate versus the accuracies of train and test set
plt.plot(learning_rate,acc_train,color='orange',label='training')
plt.plot(learning_rate,acc_test,color='red',label='test')
plt.title(" Accuracies for different Learning Rate ")
plt.xlabel("Learning Rate")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[54]:


### Final Boosted Tree
Boosted_tree = XGBClassifier(random_state=10, n_estimators = 40, learning_rate = 0.16, max_depth = 5)
Boosted_tree.fit(X_train_1,y_train_1)
y_pred_bst_1 = Boosted_tree.predict(X_test_1)


# In[55]:


### confusion matrix
Con_mat = confusion_matrix(y_test_1, y_pred_bst_1)
print(Con_mat) 


# In[56]:


accuracy_adt = accuracy_score(y_test_1,y_pred_bst_1)
accuracy_adt


# In[57]:


### Cross validation with 15 folds
accuracies_bst_1 = cross_val_score(estimator = Boosted_tree, X = X_train_1, y = y_train_1, cv = 15)
mean_acc_bst_1 = accuracies_bst_1.mean()
mean_acc_bst_1


# # Boosting for Dataset 2

# In[ ]:


# Experimentation with depth
depth = np.arange(3,20,1)
acc_train = []
acc_test = []
for i in depth:   
    BTree = XGBClassifier(max_depth= i, random_state= 20)
    BTree.fit(X_train_2, y_train_2)
    y_pred_train = BTree.predict(X_train_2)
    y_pred_test = BTree.predict(X_test_2)
    acc_train.append(accuracy_score(y_train_2,y_pred_train))
    acc_test.append(accuracy_score(y_test_2,y_pred_test))


# In[74]:


# Plotting depth vs accuracy rate of training and test    
plt.plot(depth,acc_train,color='orange',label='training')
plt.plot(depth,acc_test,color='red',label='test')
plt.title(" Accuracies for different Depth")
plt.xlabel("Depths")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[77]:


# Experimentation with boosting levels
from sklearn.ensemble import AdaBoostClassifier
boosting_level = np.arange(1,100,10)
acc_train_bst = []
acc_test_bst = []
for i in boosting_level:   
    Bst_adt = AdaBoostClassifier(n_estimators = i, random_state= 0 )
    Bst_adt.fit(X_train_2 , y_train_2)
    y_pred_train_bst = Bst_adt.predict(X_train_2)
    y_pred_test_bst = Bst_adt.predict(X_test_2)
    acc_train_bst.append(accuracy_score(y_train_2,y_pred_train_bst))
    acc_test_bst.append(accuracy_score(y_test_2,y_pred_test_bst))


# In[78]:


# Plotting the graph of boosting level Vs accuracy
plt.plot(boosting_level,acc_train_bst,color='orange',label='training')
plt.plot(boosting_level,acc_test_bst,color='red',label='test')
plt.title(" Accuracies for different Boosting Level ")
plt.xlabel("Boosting Level")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[79]:


# Experimentation with learning rate parameter
learning_rate = np.arange(0.01,0.60,0.05)
acc_train = []
acc_test = []
for i in learning_rate:   
    BTree = XGBClassifier(learning_rate = i, random_state= 20)
    BTree.fit(X_train_2, y_train_2)
    y_pred_train = BTree.predict(X_train_2)
    y_pred_test = BTree.predict(X_test_2)
    acc_train.append(accuracy_score(y_train_2,y_pred_train))
    acc_test.append(accuracy_score(y_test_2,y_pred_test))


# In[80]:


# Plotting learning rate versus the accuracies of train and test set
plt.plot(learning_rate,acc_train,color='orange',label='training')
plt.plot(learning_rate,acc_test,color='red',label='test')
plt.title(" Accuracies for different Learning Rate ")
plt.xlabel("Learning Rate")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()


# In[81]:


### Final Boosted Tree
Boosted_tree = XGBClassifier(random_state=10, n_estimators = 50, learning_rate=0.15)
Boosted_tree.fit(X_train_2,y_train_2)
y_pred_bst_2 = Boosted_tree.predict(X_test_2)


# In[82]:


### confusion matrix
Con_mat = confusion_matrix(y_test_2, y_pred_bst_2)
print(Con_mat) 
accuracy_adt = accuracy_score(y_test_2,y_pred_bst_2)
accuracy_adt


# In[83]:


### Cross validation with 10 folds
accuracies_bst_2 = cross_val_score(estimator = Boosted_tree, X = X_train_2, y = y_train_2, cv = 10)
mean_acc_bst_2 = accuracies_bst_2.mean()
mean_acc_bst_2


# In[144]:


# Plotting the mean accuracies found from cross validtion and test set accuracy
plt.plot(np.arange(0,100,10),accuracies_bst_2, color = 'green',label='Folds')
plt.title(" Accuracies of different Folds ")
plt.xlabel("Folds")
plt.grid()
plt.ylabel("Accuracy")
plt.legend()

