#pandas use in data analysis 
import pandas as pd
#numpy use to working with arrays
import numpy as np
#matplotlib use in data visualizations
import matplotlib.pyplot as plt
#Seaborn use in data visualization based on matplotlib
import seaborn as sns
#sklearn use in data analysis
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report








#Read the CSV file
data = pd.read_csv("Applais project.csv")
#print the first 5 rows of each column
data.head()


data.info()



#calculate the missing data in each column
Missing_data = data.isnull().sum()
#printthe calculated missing data
print(Missing_data)







#Replace the null values in Mobile_wt with the Mean
data['mobile_wt'] = data['mobile_wt'].fillna(data['mobile_wt'].mean())




#Check for the Dublicates
duplicates =sum(data.duplicated()) == 0
#print the result of the checking
print(duplicates)






#print the data description and check for the outlayers 
data.describe()



#Put from 1 to 12 Clusters in K
K = range(1,12)
#List to save in it clusters inertia
wss = []
for k in K:
  #Predict the closest cluster each sample in K belongs
  kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
  #put data that kmeans work on it
  kmeans=kmeans.fit(data)
  #wss equal to cluster inertia
  wss_iter = kmeans.inertia_
  #add cluster inertia to wss list
  wss.append(wss_iter)





#create My_Centers data frame to store in cluster and its inertia
My_Centers = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
#orint My_Centers data frame
print(My_Centers)







#plot elbow method to check the number of Clusters needed
#we use the data My_Centers that contain cluster and its inertia
sns.scatterplot(x = 'Clusters', y = 'WSS' , data = My_Centers, marker="+")





#this method do more check on elbow method to know number of Clusters needed
for i in range(3,13):
  #Predict the closest cluster each sample in K belongs
  label = cluster.KMeans(n_clusters=i,init='k-means++',random_state=200).fit(data).labels_
  #print the mean Silhouette Coefficient of all samples
  print("Sihouette score for k(clusters) = "+str(i)+" is "
        +str(metrics.silhouette_score(data,label,metric="euclidean",sample_size=1000,random_state=200)))






#number_clusters = 3 that calculated from Elbow and Sihouette method
number_clusters = 3
#apply k-means on data
kmeans = cluster.KMeans(n_clusters=number_clusters,init='k-means++')
kmeans = kmeans.fit(data)





#Add column Clusters to the data
#that contain each row belongs to any clusters
data['Clusters'] = kmeans.labels_





#View the data after adding Clusters column
data.head(20)



#plot pairplot gragh to the data 
#to decide which coulmn that affect by cluster column
#sns.pairplot(data)



#import the columns that affect by cluster column
X = data.iloc[:, [0,2,4,6,7,8,9,10,11,12,13,14,15]].values
#import cluster column
y = data.iloc[:, 20].values








# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 200)



# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#classifier acording to logistic regression
classifier = LogisticRegression(random_state = 200)
classifier.fit(X_train, y_train)




# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)



#do the classification report
cr=classification_report(y_test, y_pred)
print(cr)
