# Model-Pipeline-Machine-Learning
Fraud Machine Learning Pipelining for experimenting with sampling techniques
(SMOTE + SMOTE extensions available on Imbalanced Learn)

Link to creditcard fraud Dataset:
https://datahub.io/machine-learning/creditcard/r/creditcard.csv

            :Requirements:
            
            imbalanced-learn==0.6.1
            imblearn==0.0
           
            matplotlib==3.1.2
            numpy==1.17.4
            pandas==0.25.3
            

     
          
            scikit-learn==0.22
            scikit-posthocs==0.6.2
            
            scipy==1.4.1
            seaborn==0.9.0
            sklearn==0.0
            statsmodels==0.10.2

To reproduce results from thesis set rand_state = 42
and evaluate SMOTE, SVM-SMOTE, SMOTETomek, BorderlineSMOTE, RandomOversampling, SMOTEEN and No sampling (separate pipeline)

List of classifiers that are tested:

1) DecisionTreeClassifier(random_state=rand_state)

2) RUSBoostClassifier(random_state=rand_state)
3) LogisticRegression(random_state=rand_state),
4) BalancedBaggingClassifier(random_state=rand_state)
5) RandomForestClassifier(random_state=rand_state)
6) EasyEnsembleClassifier(base_estimator=RandomForestClassifier(random_state=rand_state),random_state=rand_state),
(EasyEnsembleClassifier has base estimator of RFC -> only non-default setting)

7) BalancedRandomForestClassifier(random_state=rand_state)


#Example of how to run the pipeline:


1. Firstly, create the sampling technique that you wish to analyze 
like so:

Smote = SMOTE(random_state=rand_state)

2. Create a results dataframe from the base_pipeline function:
like so:

results_smote = base_pipeline(data=new_df, sampling_technique=Smote, clean=False, verbose=False, plot=False)

Setting for Basepipeline
(1) data => the dataset that will be used (can be another other than the fraud dataset but the Class of the Target/Dependent class must be named 'Class'
   and the categories must be 0 (for negative) and 1 (for positive)
   
   
(2) sampling_technique => The sampling technique that will be tested 
(3) clean: Boolean that decides whether to run outlier removal (isolation forest)
 If clean=True -> Will run Isolation Forest to remove any outliers that are detected (up max 1000, this setting can be adjusted in the isolation forest function)
 
 If clean=False -> will not run any outlier removal

(4) Verbose: Boolean that decides whether or not to give information for each learning step as well a confusion matrix and a classification report (from imbalanced-learn library) that gives some of the metrics used with imbalanced datasets.
(5) Plot: Boolean that decides whether or not the pipeline should produce one graph with the ROC_AUC curves for each classifier that is tested (a key with the associated ROC values is also given in the graph)

