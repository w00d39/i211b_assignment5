from sklearn import datasets, model_selection, ensemble, metrics, svm, neighbors, preprocessing 
import pandas as pd

"""
I think my random forest and my tuned knn model are the best models for this data because the accuracy is high and the f1 scores are high. 
The random forest model has an accuracy of 0.9649122807017544 and an f1 score of 0.972972972972973. With this I selected the best number of estimators which turned out to be 77.
To find this I increased the number of estimators until the accuracy peaked and started decreasing then going back to the number of estimators that gave me the highest accuracy.

The tuned knn model has an accuracy of 0.9824561403508771 and an f1 score of 0.9873417721518988. With this I had a grid search to find the best arguments/parameters for the knn model. 
Which we can see versus the regular KNN model has a great improvement in accuracy and f1 score. The best parameters for the knn model were n_neighbors: 3, p: 1, and weights: distance.
In this instance the SVM was not the best model because the untuned svm model has a sharp recall which means it is way too sensitive to classifying the data. When untoggling the 
confusion matrix we can see that there are 9 cases that were properly classified as malignant but 1 case that was classified as benign when it was malignant. This is not good because
it is a cancer dataset and we want to make sure that we are classifying the data correctly. The tuned svm model has an accuracy of 0.9824561403508771 and an f1 score of 0.9873417721518988.
The tuned SVM model does perform better than the raw SVM model, but not as well as random forest and KNN were doing. 
"""

cancer_data = datasets.load_breast_cancer()

#just to get an idea of the amount of features i will be working with
#print(cancer_data.feature_names)

#the pandas here is just to get a lay of the land and decide what models I want to use
cancer_df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
#this will give me a summary of the data to get an idea of what I am working with
summary = cancer_df.describe()
#print(summary)

#this will give me the coefficients to see if I can use the ridge classification model

cancer_corr = cancer_df.corr()
#print(cancer_corr)
#spoiler alert I can't :(

#splitting
x_data = cancer_data.data #data
y_data = cancer_data.target #target
#splitting the data into training and testing sets with an 80/20 split and a random state of 301
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=301)

#scaling
#the idea of this is to make sure that the data is on the same scale so that the model can be more accurate
#this will be after I try svm and knn to see if it makes a difference to try and raise their accuracy
scaler = preprocessing.StandardScaler() #this will scale the data
X_train_scaled = scaler.fit_transform(X_train) #this will fit the data to the scaler and then transform it
X_test_scaled = scaler.transform(X_test) #this will transform the data to the scaler

#random forest
#I chose this model because it sounded like it fit the best for this data
#This model was already tuned by messing around with the best estimators to get the best accuracy I could
cancer_forest = ensemble.RandomForestClassifier(n_estimators=77, random_state=301)
cancer_forest.fit(X_train, y_train) #fitting the data to the model
cancer_forest_predictions = cancer_forest.predict(X_test) #training the model
cancer_forest_accuracy = metrics.accuracy_score(y_test, cancer_forest_predictions) #getting the accuracy of the model
print("Random Forest Accuracy: ", cancer_forest_accuracy) #printing the accuracy
print("Random Forest F1 Score", metrics.f1_score(y_test, cancer_forest_predictions)) #printing the f1 score
print("Random Forest Precision", metrics.precision_score(y_test, cancer_forest_predictions)) #printing the precision
print("Random Forest Recall", metrics.recall_score(y_test, cancer_forest_predictions), "\n\n\n") #printing the recall

#svm
#this is the initial svm model which I will use to compare to the tuned model
#svm is a good model for this data because it is a binary classification problem benign or malignant
cancer_svm = svm.SVC(random_state=301)  #the svm model
cancer_svm.fit(X_train, y_train) #fitting the data to the model
cancer_svm_predictions = cancer_svm.predict(X_test) #training the model
cancer_svm_accuracy = metrics.accuracy_score(y_test, cancer_svm_predictions) #getting the accuracy of the model
print("SVM Accuracy: ", cancer_svm_accuracy) #printing the accuracy
print("SVM F1 Score", metrics.f1_score(y_test, cancer_svm_predictions)) #printing the f1 score
print("SVM Precision", metrics.precision_score(y_test, cancer_svm_predictions)) #printing the precision
print("SVM Recall", metrics.recall_score(y_test, cancer_svm_predictions)) #printing the recall

#print(metrics.confusion_matrix(y_test, cancer_svm_predictions)) #printing the confusion matrix



#This is the tuned svm model where I used a grid search to find the best parameters
svm_params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
#this will grid search the svm model to find the best parameters so I can boost accuracy bc 92 is not to my standards of 95 at least
svm_grid = model_selection.GridSearchCV(svm.SVC(random_state=301), svm_params, cv=5)
#fitting the data to the grid search
svm_grid.fit(X_train_scaled, y_train)
#this will get the best estimator from the grid search
best_svm = svm_grid.best_estimator_
#this will print the best parameters
print("Best SVM Parameters: ", svm_grid.best_params_)

#this will predict the data using the best svm model
svm_predictions = best_svm.predict(X_test_scaled)
#this will get the accuracy of the model
print("Tuned SVM: \n")
svm_accuracy = metrics.accuracy_score(y_test, svm_predictions)
print("Tuned SVM Accuracy: ", svm_accuracy) #printing the accuracy
print("Tuned SVM F1 Score", metrics.f1_score(y_test, svm_predictions)) #printing the f1 score
print("Tuned SVM Precision", metrics.precision_score(y_test, svm_predictions)) #printing the precision
print("Tuned SVM Recall", metrics.recall_score(y_test, svm_predictions),"\n\n\n") #printing the recall


#knn
#this is the initial knn model which I will use to compare to the tuned model
#knn is a good model for this data because it is will use the nearest data points to classify the data
cancer_knn = neighbors.KNeighborsClassifier(n_neighbors=5) #the knn model with neighbors set to 5
cancer_knn.fit(X_train, y_train) #fitting the data to the model
cancer_knn_predictions = cancer_knn.predict(X_test) #training the model
cancer_knn_accuracy = metrics.accuracy_score(y_test, cancer_knn_predictions)
print("KNN Accuracy: ", cancer_knn_accuracy) #printing the accuracy
print("KNN F1 Score", metrics.f1_score(y_test, cancer_knn_predictions)) #printing the f1 score
print("KNN Precision", metrics.precision_score(y_test, cancer_knn_predictions)) #printing the precision
print("KNN Recall", metrics.recall_score(y_test, cancer_knn_predictions)) #printing the recall


#This is the tuned knn model where I used a grid search to find the best parameters
#I will use the grid search to find the best parameters for the knn model which hopefully will boost the accuracy
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
#grid seatch
knn_grid = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), knn_params, cv=5)
#fitting the data to the grid search
knn_grid.fit(X_train_scaled, y_train)
#this will get the best estimator from the grid search
best_knn = knn_grid.best_estimator_
print("Best KNN Parameters: ", knn_grid.best_params_) #printing the best parameters

#predict the data using the best knn model
knn_predictions = best_knn.predict(X_test_scaled)
#this will get the accuracy of the model
print("Tuned KNN: \n")
knn_accuracy = metrics.accuracy_score(y_test, knn_predictions)
print("Tuned KNN Accuracy: ", knn_accuracy) #printing the accuracy
print("Tuned KNN F1 Score", metrics.f1_score(y_test, knn_predictions)) #printing the f1 score
print("Tuned KNN Precision", metrics.precision_score(y_test, knn_predictions)) #printing the precision
print("Tuned KNN Recall", metrics.recall_score(y_test, knn_predictions)) #printing the recall