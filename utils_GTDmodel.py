import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def dataset_creation(filename, filename_data, features_dataset, features_class):
    #LOADING DATASET FROM FILE
    df = pd.read_excel(filename)
    dataset = df.loc[:,features_dataset]
    target = df.loc[:,features_class]
    #SAVE DATASET AND TARGET SET ON FILE
    np.savez_compressed(filename_data, dataset=dataset, target=target)
    return_labels = ['dataset', 'target']
    return np.array(return_labels, dtype=np.str)

def normalize_features(dataset):
    dataset_norm = np.copy(dataset)
    #LABELS NORMALIZATION - LABELENCODER
    le = preprocessing.LabelEncoder()
    for c in range(0,dataset_norm.T.shape[0]):
        col = dataset_norm.T[c]
        col_norm = le.fit_transform(col)
        print('feature '+str(c))
        print('max: '+str(np.max(col)))
        print('min: '+str(np.min(col)))
        print('num_values: '+str(le.classes_.size))
        print(col_norm)
        print(le.classes_)
        print('\n')
        np.put(dataset_norm.T[c], range(0,dataset_norm.shape[0]), col_norm)
    return dataset_norm

def one_hot_encoding(filename, dataset): 
    #ONE HOT ENCODING
    enc = OneHotEncoder()
    X_dataset = enc.fit_transform(dataset)
    #SAVE the encoder to use later
    dump(enc, filename)
    print(enc.n_values_)
    print(enc.active_features_)
    print(enc.feature_indices_)
    print(enc.categorical_features)
    return X_dataset

def svc_param_selection(filename_tuning, Cs, gamma, class_weight, n_folds, X_train, y_train):
    print('start parameters tuning')
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf', class_weight=class_weight), param_grid, cv=n_folds, scoring='f1_micro', verbose=10, n_jobs=5)
    grid_search.fit(X_train, y_train)
    print('Best parameters for RBF kernel with BALANCED weight classes:')
    print(grid_search.best_params_) 
    #class_weight='balanced': C = 1000; gamma = 0.001; best_f1 = 0.69223775 
    np.savez_compressed(filename_tuning, mean_f1_micro_test_score=mean_f1_micro_test_score, C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])

    