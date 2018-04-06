import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
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

def svc_param_selection(filename_tuning, Cs, gammas, class_weight, n_folds, X_train, y_train):
    print('start parameters tuning')
    param_grid = {'C': Cs, 'gamma' : gammas}
    if class_weight=='noWeight':
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=n_folds, scoring='f1_micro', verbose=10, n_jobs=5)
    else:
        grid_search = GridSearchCV(SVC(kernel='rbf', class_weight=class_weight), param_grid, cv=n_folds, scoring='f1_micro', verbose=10, n_jobs=5)
    grid_search.fit(X_train, y_train)
    print('Best parameters for RBF kernel with BALANCED weight classes:')
    print(grid_search.best_params_) 
    #class_weight='balanced': C = 1000; gamma = 0.001; best_f1 = 0.69223775 
    np.savez_compressed(filename_tuning, mean_f1_micro_test_score=grid_search.cv_results_['mean_test_score'], C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
    return grid_search
    
def imbalanced_resampling(X_set, Y_target):
    #RE-BALANCED DATSET USING IMBALANCED-LEARN
    #print('plotting sparse matrix dataset')
    #plt.figure(figsize=(14,12))
    #plt.spy(X_set, aspect='auto')
    print('starting resample')
    Y_arr_target = [elem[0] for elem in Y_target]
    smote_enn = SMOTEENN(random_state=42)
    smote_enn.fit(X_set, Y_arr_target)
    X_resampled, y_resampled = smote_enn.sample(X_set, Y_arr_target)
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled

# function to compute label weights 
def func_threshold(p_curr_label, p_i_threshold, delta_threshold):
    if p_curr_label <= (p_i_threshold+delta_threshold) and p_curr_label >= (p_i_threshold-delta_threshold):
        #print 'p_curr_label: '+str(p_curr_label)+' p_i_threshold: '+str(p_i_threshold)+' return 1'        
        return 1
    else:
        #print 'p_curr_label: '+str(p_curr_label)+' p_i_threshold: '+str(p_i_threshold)+' return 0'
        return 0
    
def print_plot_distribution(new_info_y_target, new_count_y_target, tot_examples):
    sorted_labels = np.sort(new_count_y_target)
    index_sorted_labels = np.argsort(new_count_y_target)
    perc_labels = (sorted_labels/float(tot_examples))*100
    
    print('new info label distribution:')
    print('list sorted occurence labels')
    print(str(list(sorted_labels)))
    print('frequence labels')
    print(str(list(perc_labels)))
    print('max occurence: '+str(np.max(new_count_y_target)))
    print('min occurence: '+str(np.min(new_count_y_target)))
    #print('mean occurence: '+str(np.mean(new_count_y_target)))
    print('median: '+str(np.median(new_count_y_target)))
    mode = stats.mode(new_count_y_target)
    print('mode occurence: '+str(mode[1]))
    print('the most frequent occurence '+str(mode[0]))
    #PLOT DISTRUBUTION
    norm = []
    for i in range(len(new_count_y_target)):
        norm.append(list(new_count_y_target)[i]/(float(sum(list(new_count_y_target)))))
    fig = plt.figure(figsize=(9,6))
    ind = np.arange(new_count_y_target.shape[0])                # the x locations for the groups
    width = 0.45                      # the width of the bars
    plt.bar(ind, norm, width, color='green')
    plt.xlim(-width,len(ind)+width)
    plt.ylim(0,0.12)
    plt.title("Target Labels Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Normalized Frequency Labels")
    plt.show()
    plt.savefig('labelDistr.png')
    
