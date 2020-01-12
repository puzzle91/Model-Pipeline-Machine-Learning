
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Isolation forest for outlier removal
#and sampler for outlier removal
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from imblearn import FunctionSampler

from sklearn.model_selection import train_test_split

import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import make_classification
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

#from imblearn.datasets import fetch_datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

#from sklearn.neighbors import KNeighborsClassifier as KNN



from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC )

from imblearn.pipeline import make_pipeline

from collections import Counter

from sklearn.svm import LinearSVC

from imblearn.pipeline import make_pipeline



#SMOTE & OTHER OVERSAMPLING METHODS:
from imblearn.over_sampling import ADASYN

from imblearn.over_sampling import (SMOTE, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, SMOTENC)


#from kmeans_smote import KMeansSMOTE


#SMOTE EXTENSIONS THAT COMBINE OVER & UNDER SAMPLING 

from imblearn.combine import SMOTEENN, SMOTETomek


from imblearn.base import BaseSampler

#from imblearn.over_sampling import KMeansSMOTE


# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Ensembles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import (EasyEnsembleClassifier,BalancedRandomForestClassifier, RUSBoostClassifier, BalancedBaggingClassifier)

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import gc
from imblearn.pipeline import pipeline


#Metrics:
from sklearn.metrics import roc_curve

from sklearn.metrics import (accuracy_score,confusion_matrix, classification_report, roc_auc_score,
f1_score, recall_score, precision_score, SCORERS)
from imblearn.metrics import (sensitivity_score, specificity_score, geometric_mean_score, classification_report_imbalanced)
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


import warnings
warnings.filterwarnings("ignore")

print(__doc__)










df = pd.read_csv('https://datahub.io/machine-learning/creditcard/r/creditcard.csv')

#Using a uniform random state number for reproducability.
rand_state = 42


print("----"*10)
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
print("NotFraud:", df['Class'].value_counts()[0])
print ("fraud:", df['Class'].value_counts()[1])
print("----"*10)
#  No Null Values!
print("NULL VALUES", df.isnull().sum().max())


print(df['Amount']) 

print("Min (0 - transaction amount):", min(df['Amount']))
print("Max:", max(df['Amount']))


print("done")


print(df['Amount'].cummin())



original_df = df
# print(df.head(5))

df = df.sample(frac=1, random_state=rand_state)



X = df.drop('Class', axis=1)
y = df['Class']


#SCALING DATA
std_scaler = StandardScaler()
rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)


scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

#The two features Time and Amount are scaled



# Due to the pervious insert of new colums scaled amount and scaled time 
# The dataset will be shuffled before creating training and test sets 
# 
df = df.sample(frac=1, random_state=rand_state)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1] #Taking all fraud cases
non_fraud_df = df.loc[df['Class'] == 0] #Taking all  non_fraud cases

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffling again due to the concatenation above 

new_df = normal_distributed_df.sample(frac=1, random_state=rand_state)

print("NEW DF:", new_df.head(5))

print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()) 
print(len(new_df))



X = new_df.drop('Class', axis=1)
y = new_df['Class']

X_vals = X.values
y_vals = y.values

# This is explicitly used for the data-cleaning process (outlier rejection using isolation forest) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


##This will be our function used to resample our dataset.
# Using a naive ISOLATION FOREST ALOGIRTHM with default params
# TO REMOVE OUTLIERS during stratified fold 


 

def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = IsolationForest(max_samples=1000,
                                contamination='auto',
                                random_state=rand_state,
                                behaviour='new')
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]



def plot_scatter(X, y, title):
    """Function to plot some data as a scatter plot."""
    plt.figure( figsize=(16,16))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', label='Class #1 - Fraud')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', label='Class #0 - Non-Fraud')
    plt.legend()
    plt.title(title)

reject_sampler = FunctionSampler(func=outlier_rejection)
X_vals = X.values
y_vals =y.values
X_inliers, y_inliers = reject_sampler.fit_resample(X_vals, y_vals)
plot_scatter(X_inliers, y_inliers, 'Training data without outliers')

print("Total outliers removed: {:}".format(len(X_vals) - len(X_inliers)))
print("New lenght of X: {} ; new length of y {}".format(len(X_inliers),  len(y_inliers)))








def nosampling_pipeline(data=[],  verbose=False, clean=False, plot=False):
          
          results_table=[]
          results=[]
          rand_state =42
          
          if clean:
              X = data.drop('Class', axis=1)
              y = data['Class']
              X_vals=X.values
              y_vals=y.values
              X_inliners, y_inliners = reject_sampler.fit_resample(X_vals, y_vals)
              X=X_inliners
              y=y_inliners
          else:
              X = data.drop('Class', axis=1)
              y = data['Class']
              X=X.values  
              y=y.values
              pass

          sss = StratifiedKFold(n_splits=10, random_state=rand_state, shuffle=False)
          print("StratKFold:",sss)


          #List of models to be used
          models=[DecisionTreeClassifier(random_state=rand_state), RUSBoostClassifier(random_state=rand_state),
                  LogisticRegression(random_state=rand_state), BalancedBaggingClassifier(random_state=rand_state),
                RandomForestClassifier(random_state=rand_state),
                 EasyEnsembleClassifier(base_estimator=RandomForestClassifier(random_state=rand_state),random_state=rand_state),
                  BalancedRandomForestClassifier(random_state=rand_state)]

        
          results_table = pd.DataFrame(columns=['models', 'fpr','tpr','auc'])
          #Create training and testing data sets depending on wheather or not they have been generated previously.
          #Instantiate lists to store each of the models results
          strategy =[]
          classifier = []
          strategy=[]
          samp_technique=[]
          accuracy = []
          f1 = []
          auc = []
          recall = []
          precision = []
          g_mean = []
          start = time.time()
          #Run thorugh each of the models to get their performance metrics
          
         
          sampling_strat = 'no_sampling'
          
    

          
          for train_index, test_index in sss.split(X,y):
          
          
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

          # X_train=X_train.values
          # X_test=X_test.values
          # y_train=y_train.values
          # y_test=y_test.values

          for model in models: 
            print("Using lentgh of X for training: {}; Using Length of Y for training: {}".format(len(X_train), len(y_train)))
            print("Using lentgh of X for testing: {}; Using Length of Y for test: {}".format(len(X_test), len(y_test)))
            
            print("Currently training model - {} using sampling strategy - {}".format(model.__class__.__name__, sampling_strat))
            print("--"*20)


            clf = model
           
            pipe = make_pipeline(clf) # LOG_REG_MODEL WITH BOTHER
            pipe.fit(X_train, y_train)
            
            
            test_preds = pipe.predict(X_test)
            #yproba = pipe.predict_proba(X_test)[::,1] 
            
            classifier.append(model.__class__.__name__)
            samp_technique.append(sampling_strat)
            strategy.append(" %s+%s " %(str(model.__class__.__name__), sampling_strat))

            f1.append(f1_score(y_test, test_preds))
            accuracy.append(accuracy_score(y_test, test_preds))
            auc.append(roc_auc_score(y_test, test_preds))
            recall.append(recall_score(y_test, test_preds))
            precision.append(precision_score(y_test, test_preds))
            g_mean.append(geometric_mean_score(y_test, test_preds, average='binary'))

  
            fpr, tpr, _ = roc_curve(y_test,  test_preds)
            auc_score = roc_auc_score(y_test, test_preds)
      
            results_table = results_table.append({'classifiers':model.__class__.__name__,
                                          'fpr':fpr, 
                                          'tpr':tpr, 
                                          'auc_score':auc_score}, ignore_index=True)
                                                                            

           
        #Print the model and its report
            if verbose:
                print('Classification Model: ', model.__class__.__name__,'\n')
                print ('Sampling Strategy Model: ', sampling_strat,'\n')
                print(confusion_matrix(y_test, test_preds),'\n')
                print(classification_report_imbalanced(y_test, test_preds),'\n')

          #round the results for convenience
          f1 = [float(round(n, 4)) for n in f1]
          auc = [float(round(n, 4)) for n in auc]
          g_mean = [float(round(n, 4)) for n in g_mean]
          accuracy = [float(round(n, 4)) for n in accuracy]
          precision = [float(round(n, 4)) for n in precision]
          recall = [float(round(n, 4)) for n in recall]

          #store results in dataframe
           
          results = pd.DataFrame([classifier, strategy, samp_technique, f1, auc, g_mean, accuracy, precision, recall],
                        index= ['classifier', 'strategy', 'samp_technique', 'f1','roc_auc','g_mean', 'accuracy','precision','recall'],
          columns=['DecisionTreeClassifier','RUSBoostClaassifier','LogisiticRegression', 'BalancedBaggingClassifier', 'RandomForestClassifier',
                                      'EasyEnsembleClassifier', 'BalancedRandomForestClassifier'])
          

          
    

          
          if plot:
              
              results_table.set_index('classifiers', inplace=True) 
              fig = plt.figure(figsize=(8,6))
              results_table.sort_values(by=['auc_score'],ascending=False)

              for i in results_table.index:
                  
                  plt.plot(results_table.loc[i]['fpr'], 
                          results_table.loc[i]['tpr'], 
                          label="{}, AUC={:.4f}".format(i, results_table.loc[i]['auc_score']))
                  
                  plt.plot([0,1], [0,1], color='orange', linestyle='--')

                  plt.xticks(np.arange(0.0, 1.1, step=0.1))
                  plt.xlabel("Flase Positive Rate", fontsize=15)

                  plt.yticks(np.arange(0.0, 1.1, step=0.1))
                  plt.ylabel("True Positive Rate", fontsize=15)

                  plt.title('ROC Curve for classifiers using Full data split using sampling technique: {}'.format(sampling_strat), fontweight='bold', fontsize=15)
                  plt.legend(prop={'size':13}, loc='lower right')

          plt.show()
          
      
          
          #Change orientation of the dataframe

          end = time.time()
          print("Time elapsed:", start-end)
          
          return results.transpose()