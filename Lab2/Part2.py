#Assigning features and labels variables
#First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 
'Rainy','Sunny','Overcast','Overcast','Rainy']
#Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
#Label or target variable
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#Import LabelEncoder
from sklearn import preprocessing
#Creating LabelEncoder
le = preprocessing.LabelEncoder()
#Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print ("Weather:",weather_encoded)

#converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

#Combining Features
features=list(zip(weather_encoded,temp_encoded))

#Generating Model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
#Train the model using the training sets
model.fit(features,label)
#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted) #Output: [1]

# Import scikit-learn dataset library
from sklearn import datasets
# Load dataset
wine = datasets.load_wine()
# print the names of the features
print("Features: ", wine.feature_names)
#print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target_names)
#print the wine data (top 5 records)
print("Data: ", wine.data[0:5])
#print the wine labels (0:Class_0, 1:class_2, 2:class_2)
print("Labels: ", wine.target)
#print data(feature)shape
print(wine.data.shape)
#print target(or label)shape
print(wine.target.shape)

# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy for k=5:",metrics.accuracy_score(y_test, y_pred))

#Create a new KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy for k=7:",metrics.accuracy_score(y_test, y_pred))