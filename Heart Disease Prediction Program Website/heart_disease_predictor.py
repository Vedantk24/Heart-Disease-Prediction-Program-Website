# -*- coding: utf-8 -*-

try:
    #importing the libraries
    import warnings
    import pickle
    import pandas                 as pd
    import numpy                  as np
    from sklearn.preprocessing    import OneHotEncoder
    from sklearn.compose          import ColumnTransformer
    from sklearn.model_selection  import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
    from sklearn.ensemble         import RandomForestClassifier
    from sklearn.linear_model     import LogisticRegression
    from sklearn.neighbors        import KNeighborsClassifier
    from sklearn.metrics          import accuracy_score, confusion_matrix, precision_score, recall_score
    warnings.filterwarnings('ignore')
    
except Exception as e:
    print("Unable to import the libraries",e)

#==============================Data Preprocessing================================
#loading the dataset
dataset=pd.read_csv('heart_data.csv', sep='\t' )

#to check the diferrent unique values in the dataset
for index in dataset.columns:
    print(index,dataset[index].unique())
print(dataset.dtypes)
print('The number of missing dataset:', dataset.isnull().sum().sum())
    
#splitting the dataset to independent and dependent sets
dataset_X=dataset.iloc[:,  0:13].values
dataset_Y=dataset.iloc[:, 13:14].values

#columns to be encoded: cp(2), restecg(6), slope(10), ca(11), thal(12)
ct=ColumnTransformer([('encoder', OneHotEncoder(drop='first'), [2,6,10,11,12])], remainder='passthrough')
dataset_X = ct.fit_transform(dataset_X)


#splitting data to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(dataset_X, dataset_Y, test_size=0.3 , random_state=0)

#==============================Evaluation=======================================
#scores

def scores(pred,test,model):
    print(('\n==========Scores for {} ==========\n').format(model))
    print(f"Accuracy Score   : {accuracy_score(pred,test) * 100:.2f}% " )
    print(f"Precision Score  : {precision_score(pred,test) * 100:.2f}% ")
    print(f"Recall Score     : {recall_score(pred,test) * 100:.2f}% " )
    print("Confusion Matrix :\n" ,confusion_matrix(pred,test))

    
#====================================LR_Tunned==========================================   
#logistic regression

lr       = LogisticRegression()
solvers  = ['newton-cg', 'lbfgs', 'liblinear']
penalty  = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid_lr ={"solver":solvers,
          "penalty":penalty,
          "C":c_values}


grid_search_lr = GridSearchCV(estimator=lr, param_grid=grid_lr, n_jobs=-1, cv=10, scoring='accuracy',error_score=0)

#getting the best parameters
grid_result_lr = grid_search_lr.fit(X_train,Y_train)
best_grid_lr   =grid_result_lr.best_estimator_
best_grid_lr.fit(X_train, Y_train)
Y_pred_knn_t    =best_grid_lr.predict(X_test)
scores(Y_pred_knn_t,Y_test,'KNeighbors_Classifier_Tunned')


#===================================KNN==========================================
#K Nearest Neighbour

#tunning the k value
accuracy=[]
for index in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=index)
    score=cross_val_score(knn,X_train, Y_train,cv=10)
    accuracy.append(score.mean())

best_k=accuracy.index(max(accuracy))

#fitting the model with best k value
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(X_train, Y_train)
Y_pred_knn=knn.predict(X_test)
scores(Y_pred_knn,Y_test,'KNeighbors_Classifier')

#===================================KNN_Tunned====================================
#K Nearest Neighbour with Hyper parameter

knn_t = KNeighborsClassifier()
n_neighbors = range(1,20)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

# define grid search
grid = {'n_neighbors' : n_neighbors,
        'weights'     : weights,
        'metric'      : metric}

grid_search_knn = GridSearchCV(estimator=knn_t, param_grid=grid, n_jobs=-1, cv=10 ,scoring='accuracy',error_score=0)

#getting the best parameters
grid_result_knn = grid_search_knn.fit(X_train,Y_train)
best_grid_knn   =grid_result_knn.best_estimator_
best_grid_knn.fit(X_train, Y_train)
Y_pred_knn_t    =best_grid_knn.predict(X_test)
scores(Y_pred_knn_t,Y_test,'KNeighbors_Classifier_Tunned')

#====================================RF==========================================
#Random Forest

n_estimators        = [int(x) for x in np.linspace(start =10, stop = 200, num = 10)]
max_features        = ['auto', 'sqrt','log2']
max_depth           = [int(x) for x in np.linspace(10, 1000,10)]
min_samples_split   = [2, 5, 10,14,None]
min_samples_leaf    = [1, 2, 4,6,8,None]

random_grid = {'n_estimators'      : n_estimators,
               'max_features'      : max_features,
               'max_depth'         : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf'  : min_samples_leaf}

rf=RandomForestClassifier()

randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=300,cv=10,
                               random_state=100,n_jobs=-1)
#getting the best parameters
randomcv.fit(X_train,Y_train)
best_grid=randomcv.best_estimator_
best_grid.fit(X_train,Y_train)
pred_2=best_grid.predict(X_test)
scores(pred_2,Y_test,'RandomForestClassifier')


#=============================Saving the models==================================
#saving model to disk

pickle.dump(best_grid_lr, open('ml_model.pkl', 'wb'))
pickle.dump(ct,           open('encoder.pkl',  'wb'))

#==============================Testing model response============================
#test the pickle file

def test_model(row_number):
    model=pickle.load(open('ml_model.pkl', 'rb'))
    value,real=dataset_X[row_number,:].reshape(1,-1),dataset_Y[row_number,:]
    print(("\n The value predicted is : {} and the real value is : {} ").format(model.predict(value), real))


test_model(102)
print('\nCompleted')