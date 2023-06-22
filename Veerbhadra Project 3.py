#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as np
import numpy as rp
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


#importing dataset


# In[19]:


file=np.read_excel("~/desktop/Google ads hourly analysis 20th june.xlsx")


# In[20]:


file


# In[22]:


# the data set gives us how many people who clicked and which was cold lead and hot lead and warm lead of advertisment with ctr and cpc .


# In[ ]:


# It contains user impression and clicks and this adertisment data is hot , cold and warm lead is determine 


# In[ ]:


#list of first five rows


# In[21]:


file. head()


# In[ ]:


#list of last five rows


# In[23]:


file.tail()


# In[24]:


#data preprocessing 


# In[25]:


#check number of unique value from all data set 


# In[26]:


file.select_dtypes(include='object').nunique()


# In[27]:


file.shape


# In[ ]:


# there is drop or unwanted column is drop outin data set.


# In[28]:


data=file.drop("Sr no",axis=1)


# In[29]:


data


# In[ ]:


# to check any missing or null value in the dataset 


# In[30]:


data.isnull().sum()


# In[ ]:


# is there 4 missing or non value find out in data set 


# In[31]:


file.isnull().sum()


# In[ ]:


# these 4 missing value or none value is drop out in dataset 


# In[104]:


at=data.dropna()


# In[105]:


at


# In[106]:


at.isnull().sum()


# In[ ]:


# in that there is no none data or missing value .


# In[ ]:


#find out mean value .


# In[108]:


rl=at.fillna(rt.mean())


# In[109]:


rl


# In[39]:


# descriptive analysis.


# In[ ]:


# in that descriptive anlaysis there is findout summary about the data set .


# In[110]:


rl.info()


# In[111]:


rl.describe()


# In[45]:


#the statistical summary of the dataset givs us following information .
#there is all count is simlilar there no. of count is 44.00
#there is mean of cold lead is more than that hot lead is genertion of consumer interset is more but in that data there is less than cold lead .
#there is warm is minimum for the data set .
#there is 50% of pepole or custmer there is impression is more but there is no awarness about that advertisment.
#there is std.devation is impreession 3693.944
#there is warm lead is less than cold lead because there is ctr and ctc is affect on it .


# In[112]:


rl.nunique()


# In[113]:


rl.sum()


# In[ ]:


# there is all sum of data for easy overview the data .


# In[114]:


rl.mean()


# In[ ]:


#there is average of all data like lead and impression and clicks .
#there is average of all data to easily study that difference in data set. 


# In[115]:


rl.isnull().sum()


# In[116]:


x=rl.iloc[:,:-1].values
y=rl.iloc[:,-1].values


# In[117]:


x


# In[118]:


y


# In[119]:


#linear regreesion 


# In[120]:


#exploratory data analysis.


# In[121]:


from sklearn.model_selection import train_test_split 
x_train, x_test,  Y_train, Y_test = train_test_split(x,y, test_size = 0.30, random_state = 0) 


# In[122]:


print(x_train)


# In[60]:


print(Y_train)


# In[61]:


print(x_test)


# In[123]:


print(Y_test)


# In[124]:


from sklearn.linear_model import LinearRegression 


# In[125]:


lr= LinearRegression()


# In[126]:


lr


# In[127]:


lr.fit(x_train,Y_train)


# In[128]:


y_predict=lr.predict(x_test)


# In[129]:


y_predict


# In[130]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[131]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[132]:


print(x_train)


# In[133]:


print(x_test)


# In[134]:


from sklearn.tree import DecisionTreeClassifier


# In[135]:


ct= DecisionTreeClassifier(criterion="gini",random_state=0)



# In[136]:


ct.fit(x_train,Y_train)


# In[137]:


print(ct.predict(sc.transform([[3000,221,400,0.234,3.23,2,0.2]])))


# In[139]:


print(ct.predict(sc.transform([[1000,232,100,0.00121,2.232,1,0.1]]))) 


# In[140]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[141]:


Y_prediction =vt.predict(x_test)


# In[142]:


Y_prediction


# In[143]:


gm= confusion_matrix(Y_test,Y_prediction)


# In[144]:


accuracy_score(Y_test,Y_prediction)


# In[ ]:


# we obtain 57 percentage of accuracy of dataset 


# In[85]:


#main alogorithms 


# In[86]:


#Correlations


# In[87]:


# For whole dataset


# In[145]:


rl.corr()


# In[90]:


# For some selected coulmns or attributes


# In[146]:


ct=don[['Clicks','CTR']].corr()


# In[148]:


ct


# In[149]:


wr=don[['CTR','CPC']].corr()


# In[150]:


wr


# In[151]:


sns.lineplot(x='CTR',y='CPC',data=data)


# In[169]:


sns.lineplot(x='Clicks',y='CTR',data=data)


# In[165]:


corr = don.corr()

plt.figure(figsize=(8,4))
sns.heatmap(corr,cmap="Greens",annot=True)


# In[154]:


corr = don.corrwith(don['CTR']).sort_values(ascending = False ).to_frame()
corr.columns =['CTR']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('CTR Correlation')


# In[155]:


corr = don.corrwith(don['CPC']).sort_values(ascending = False ).to_frame()
corr.columns =['CPC']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('CPC Correlation')


# In[156]:


corr = don.corrwith(don['Hot Leads']).sort_values(ascending = False ).to_frame()
corr.columns =['Hot Leads']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('HOT Leads Correlation')


# In[163]:


#conclusion of analysis.

#various phases of data analysis including data collection, cleaning and analysis are discussed briefly.

#Explorative data analysis is mainly studied here. 

#For the implementation, Python programming language is used.

#For detailed research, jupyter notebook is used. Different Python libraries and packages are introduced.

# We can see that the Impression ,Clicks and Sales units there are interrelation between them 

#  we can see that when clicks increase Sales also increase .

# The  clicks had the best sales.

# we can see that the DecisionTreeClassifier is used for accuaracy define in dataset. 

#the statistical summary of the dataset givs us following information .

#there is all count is simlilar there no. of count is 44.00

#there is mean of cold lead is more than that hot lead is genertion of consumer interset is more but in that data there is less than cold lead .

#there is warm is minimum for the data set .

#there is 50% of pepole or custmer there is impression is more but there is no awarness about that advertisment.

#there is std.devation is impreession 3693.944

#there is warm lead is less than cold lead because there is ctr and ctc is affect on it .

# we can see that hot leads correlation 

# we can see that warm lead correlation.

# we can see that how the leads are relation between this cliks and impression 

# We can see that the linerregression to increase sales with increase the clicks and impression 

# we can see that the average std deviation of impression very spread due to marketing canablize.

# We can see that the std deviation of clicks is some low than Impression because there is low awarnees of this ads.

# it is all about To analyses how many people who clicked on the advertisement enrolled in our course.

#in that data set we learn about data is there ctc and ctr data analysis after clicking ad .on the time which choose the have you leads like warm , cold and and hot lead which will be consider.

# there is hot lead is less than cold lead . 

# for there is loss of advertisment there is hot lead is overall 11.00 and cost is 135 per advertisment . 
# there is time


# In[158]:


#insights


# In[159]:


# In all about analaysis dataset to inform that general marketing and how the people was aware about advertisment 
# this advertisment was 6th june. 
# It main think that there was tuesday is a working day .
# people mindset was to do  workholic or motivated 
# that day they search or aware about cources
# some people was went house from office that time is about 12.am 
# some people go to saw this particular ads but not click .
# to all dataset analysis there was impression was slightly peak but not click this ads .
# some people to aware this ads more information was find to click them this ads then this ads useful for this.
# those people want to sale this course.
# by the analysis is found that there is intereltaion between CTR, and CPC of paticualr advertisement 
# and to analyais of what is hot leads and cold and warm leads genertion of particualar advertisement 
#for the analysis there is 0.573 accuracy of data is obtained it menas that there is 57% customer refer or see the this advertsisment for course .
# i suggest to marketing head to increses ad. and disply reptabley for marketing and awaerness purpose . 
# on the time which choose the have you leads like warm , cold and and hot lead which will be consider.
# there is hot lead is less than cold lead . 
# for there is loss of advertisment there is hot lead is overall 11.00 and cost is 135 per advertisment . 


# In[ ]:





# In[ ]:





# In[ ]:




