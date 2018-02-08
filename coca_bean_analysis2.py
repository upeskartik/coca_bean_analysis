
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
coca_df = pd.read_csv(r"C:\Users\kartik singh\Desktop\glass classification\flavors_of_cacao.csv", encoding='mac_roman')
coca_np = np.array(coca_df)
coca_np


# In[2]:

coca_df.head()


# In[3]:

Company=[]
Origin=[]
REF=[]
Review_Date=[]
Cocoa_percent=[]
Company_loc=[]
Rating=[]
Type=[]
Broad_origin=[]
for index,i in coca_df.iterrows():
    Company.append(i["Company"])
    Origin.append(i["Bean_Origin"])
    REF.append(i["REF"])
    Review_Date.append(i["Review_Date"])
    Cocoa_percent.append(i["Cocoa_Percent"])
    Company_loc.append(i["Company_Location"])
    Rating.append(i["Rating"])
    Type.append(i["Type"])
    Broad_origin.append(i["Broad_Bean_Origin"])


# In[60]:

Company_unique=np.unique(Company)
Unique_Type = np.unique(Type)
Unique_Type


# In[5]:

for i in range(0,416):
    c=0
    for j in Company:
        if j==Company_unique[i]:
            Company[c]=i
        c=c+1 
        


# In[6]:

for i in range(0,42):
    c=0
    for j in Type:
        if j==Unique_Type[i]:
            Type[c]=i
        c=c+1 


# In[41]:

Unique_Origin=np.unique(Origin)
Unique_loc = np.unique(Company_loc)
Unique_Broad = np.unique(Broad_origin)
Review_Date_Unique = np.unique(Review_Date)
Broad_origin_unique = np.unique(Broad_origin)
index=0
for i in Broad_origin:
    if i==:
        print(index)
    index=index+1 
Broad_origin_unique[100]


# In[8]:

for i in range(0,1039):
    c=0
    for j in Origin:
        if j==Unique_Origin[i]:
            Origin[c]=i
        c=c+1 


# In[9]:

for i in range(0,60):
    c=0
    for j in Company_loc:
        if j==Unique_loc[i]:
            Company_loc[c]=i
        c=c+1   


# In[10]:

for i in range(0,101):
    c=0
    for j in Broad_origin:
        if j==Unique_Broad[i]:
            Broad_origin[c]=i
        c=c+1   
max(Broad_origin)        


# In[11]:

from statistics import median,mean
np.array(Review_Date).std()


# In[12]:

def Z_score(x,mean,std):
    Z_value=(x-mean)/std 
    return Z_value 
Z_score_Company_loc = []
Z_score_Rating=[]
for i in Company_loc:
    Z_score_Company_loc.append(Z_score(i,37.9125348189415,20.770603664387068))
for i in Rating:
    Z_score_Rating.append(Z_score(i,3.185933147632312,0.47792921002236244))


# In[13]:

p=0
for i in range(0,len(Company_loc)):
    p=p+(Z_score_Company_loc[i]*Z_score_Rating[i])
p=p/(len(Company_loc)-1)
p


# In[ ]:




# In[14]:

Z_score_Cocoa_percent = []
for i in Cocoa_percent:
    Z_score_Cocoa_percent.append(Z_score(i,0.7170027855153204,0.06321240219833289))
p=0
for i in range(0,len(Cocoa_percent)):
    p=p+(Z_score_Cocoa_percent[i]*Z_score_Rating[i])
p=p/(len(Cocoa_percent)-1)
p    


# In[15]:

Z_score_Company = []
for i in Company:
    Z_score_Company.append(Z_score(i,206.25849582172702,124.7459190593882))
p=0
for i in range(0,len(Company)):
    p=p+(Z_score_Company[i]*Z_score_Rating[i])
p=p/(len(Company)-1)
p    


# In[16]:

Z_score_Review_Date= []
for i in Review_Date:
    Z_score_Review_Date.append(Z_score(i,2012.325348189415,2.9263947851670165))
p=0
for i in range(0,len(Review_Date)):
    p=p+(Z_score_Review_Date[i]*Z_score_Rating[i])
p=p/(len(Review_Date)-1)
p 


# In[17]:

Z_score_Origin= []
for i in Origin:
    Z_score_Origin.append(Z_score(i,523.8846796657382,291.78459556832297))
p=0
for i in range(0,len(Origin)):
    p=p+(Z_score_Origin[i]*Z_score_Rating[i])
p=p/(len(Origin)-1)
p 


# In[83]:

Company_df=pd.DataFrame(Company,columns=["Company"]).astype(np.float32)
Company_loc_df=pd.DataFrame(Company_loc,columns=["Company_loc"]).astype(np.float32)
Cocoa_percent_df=pd.DataFrame(Cocoa_percent,columns=["Cocoa_percent"]).astype(np.float32)
Review_Date_df=pd.DataFrame(Review_Date,columns=["Na"]).astype(np.float32)
Origin_df=pd.DataFrame(Origin,columns=["Origin"]).astype(np.float32)
Broad_origin_df=pd.DataFrame(Broad_origin,columns=["Broad_origin"]).astype(np.float32)
Type_df = pd.DataFrame(Type,columns=["Type"])


# In[ ]:

Broad_origin_df=Broad_origin_df.fillna(0)
Type_df = Type_df.fillna(0)


# In[108]:

df_train=Cocoa_percent_df.join(Review_Date_df)
df_train=df_train.join(Company_loc_df)
df_train=df_train.join(Origin_df)
#df_train=df_train.join(Type_df)
df_train=df_train.join(Broad_origin_df)


# In[109]:

Rank = []
for j,i in coca_df.iterrows():
    if i["Rating"]>=1 and i["Rating"]<2:
        Rank.append(1)

    elif i["Rating"]>=2 and i["Rating"]<3:
        Rank.append(2)   
    elif i["Rating"]>=3 and i["Rating"]<3.25:
        Rank.append(3)
    elif i["Rating"]>=3.25 and i["Rating"]<3.5:
        Rank.append(4)
    elif i["Rating"]>=3.5 and i["Rating"]<3.75:
        Rank.append(5) 
    elif i["Rating"]>=3.75 and i["Rating"]<4:
        Rank.append(6) 
    elif i["Rating"]>=4 and i["Rating"]<5:
        Rank.append(7) 
    else:
        Rank.append(8)
Rank_df=pd.DataFrame(Rank,columns=["Rank"]).astype(np.float32)


# In[110]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df_train, Rank_df, test_size = 0.3, random_state = 100)


# In[111]:

from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=5, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[112]:

y_pred=clf_gini.predict(X_test)


# In[113]:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[114]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion="gini",max_depth=3, random_state=40)
clf.fit(X_train,y_train)


# In[115]:

y_pred_RF = clf.predict(X_test)


# In[116]:

accuracy_score(y_test, y_pred_RF)


# In[103]:

from sklearn.svm import SVC
clf = SVC(shrinking=False)
clf.fit(X_train,y_train)


# In[104]:

y_pred_SVM = clf.predict(X_test)


# In[105]:

accuracy_score(y_test, y_pred_SVM)


# In[91]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



