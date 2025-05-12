# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import itertools
import joblib
# import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
# from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
# from sklearn.metrics import confusion_matrix, r2_score
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
# from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
# from imblearn.over_sampling import SMOTE, RandomOverSampler, SVMSMOTE, BorderlineSMOTE
from sklearn.multiclass import OneVsRestClassifier
# from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids
# from sklearn.feature_selection import RFE, SelectKBest, f_classif
# from xgboost import XGBClassifier
# from collections import Counter
# from imblearn.combine import SMOTETomek, SMOTEENN

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # drop missing values
    df = df.dropna()
    
    # feature mapping
    grade_mapping = { 'Below Average (2.75 - 5)': 1, 'Average (1.75 - 2.5)': 2, 'Above Average (1.0 - 1.5)': 3, 'P (Passed - pandemic period)': 2 }
        
    df['PracticumGrade'] = df['PracticumGrade'].map(grade_mapping)
    df['WebDevGrade'] = df['WebDevGrade'].map(grade_mapping)
    df['DSAGrade'] = df['DSAGrade'].map(grade_mapping)
    df['FundamentalsProgGrade'] = df['FundamentalsProgGrade'].map(grade_mapping)
    df['OOPGrade'] = df['OOPGrade'].map(grade_mapping)
    df['FoundationsCSGrade'] = df['FoundationsCSGrade'].map(grade_mapping)
    df['NetworkingGrade'] = df['NetworkingGrade'].map(grade_mapping)
    df['NumericComputationGrade'] = df['NumericComputationGrade'].map(grade_mapping)
    df['ExtracurricularsLevel'] = df['ExtracurricularsLevel'].map({'Low': 1, 'Moderate': 2, 'High': 3})
    df['LatinHonors'] = df['LatinHonors'].map({'No': 0, 'Yes': 1})
    
    # imputation
    df['PracticumGrade'] = df['PracticumGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['WebDevGrade'] = df['WebDevGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['DSAGrade'] = df['DSAGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['FundamentalsProgGrade'] = df['FundamentalsProgGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['OOPGrade'] = df['OOPGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['FoundationsCSGrade'] = df['FoundationsCSGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['NetworkingGrade'] = df['NetworkingGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 
    df['NumericComputationGrade'] = df['NumericComputationGrade'].replace('Not applicable and/or was not offered during time of study', np.nan) 

    imputer = SimpleImputer(strategy='median')
    
    df['PracticumGrade'] = imputer.fit_transform(df[['PracticumGrade']]) 
    df['WebDevGrade'] = imputer.fit_transform(df[['WebDevGrade']]) 
    df['DSAGrade'] = imputer.fit_transform(df[['DSAGrade']]) 
    df['FundamentalsProgGrade'] = imputer.fit_transform(df[['FundamentalsProgGrade']]) 
    df['OOPGrade'] = imputer.fit_transform(df[['OOPGrade']]) 
    df['FoundationsCSGrade'] = imputer.fit_transform(df[['FoundationsCSGrade']]) 
    df['NetworkingGrade'] = imputer.fit_transform(df[['NetworkingGrade']]) 
    df['NumericComputationGrade'] = imputer.fit_transform(df[['NumericComputationGrade']]) 

    # feature transformations
    df['AcademicGrade'] = df['WebDevGrade'] + df['DSAGrade'] + df['FundamentalsProgGrade'] + df['OOPGrade'] + df['FoundationsCSGrade'] + df['NetworkingGrade'] + df['NumericComputationGrade']
    df['ExternalMetrics'] = df[['PracticumGrade', 'ExtracurricularsLevel', 'LatinHonors']].mean(axis=1)
    df['OverallGrade'] = df[['PracticumGrade', 'WebDevGrade', 'DSAGrade', 'FundamentalsProgGrade', 'OOPGrade', 'FoundationsCSGrade', 'NetworkingGrade', 'NumericComputationGrade']].mean(axis=1)
    df['FundamentalsGrade'] = df[['DSAGrade', 'FundamentalsProgGrade', 'OOPGrade', 'FoundationsCSGrade']].mean(axis=1)
    df['SpecializedGrade'] = df[['WebDevGrade', 'NetworkingGrade', 'NumericComputationGrade']].mean(axis=1)

    df['WebDevStrength'] = df['WebDevGrade'] - df['OverallGrade']
    df['DSAStrength'] = df['DSAGrade'] - df['OverallGrade']
    df['FundamentalsProgStrength'] = df['FundamentalsProgGrade'] - df['OverallGrade']
    df['OOPStrength'] = df['OOPGrade'] - df['OverallGrade']
    df['FoundationsCSStrength'] = df['FoundationsCSGrade'] - df['OverallGrade']
    df['NetworkingStrength'] = df['NetworkingGrade'] - df['OverallGrade']
    df['NumericComputationStrength'] = df['NumericComputationGrade'] - df['OverallGrade']

    # identifying columns
    # X = df[['PracticumGrade','WebDevGrade','DSAGrade','FundamentalsProgGrade','OOPGrade','FoundationsCSGrade','NetworkingGrade','NumericComputationGrade','ExtracurricularsLevel','LatinHonors']]
    # X = df[['WebDevStrength', 'DSAStrength', 'FundamentalsProgStrength', 'OOPStrength', 'FoundationsCSStrength', 'NetworkingStrength', 'NumericComputationStrength', 'ExtracurricularsLevel', 'LatinHonors']]
    # X = df[['OverallGrade', 'ExtracurricularsLevel', 'LatinHonors']]
    X = df[['WebDevGrade', 'FundamentalsProgGrade','FoundationsCSGrade', 'ExtracurricularsLevel', 'LatinHonors']] # based on select K-Best
    y = df['CategorizedJobTitle']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # feature scaling
    sc_X = StandardScaler()
    X_scaled = sc_X.fit_transform(X)

    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    # dummy classifier (baseline model)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    dummy_f1 = f1_score(y_test, y_dummy, average='weighted')
    print(f"Dummy F1: {dummy_f1:.4f}\n")

    '''
    CLASSIFICATION PROBLEM (JOB TITLES)
    '''
    # param grid for hyperparameter tuning
    param_grid = {
        'C': [2**-5, 2**-3, 2**-1, 2, 2**3, 2**5, 2**7],
        'gamma': [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2, 2**1, 2**3]
    }

    param_combinations = list(itertools.product(param_grid['C'], param_grid['gamma']))

    best_score = 0
    best_params = None
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=3)

    for C, gamma in param_combinations:
        accuracies, weighted_f1s, weighted_precisions, weighted_recalls = [], [], [], []
        train_accuracies, train_f1s = [], []

        for train_index, test_index in skf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_fold, y_train_fold)
            
            model = OneVsRestClassifier(SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced'))
            model.fit(X_train_resampled, y_train_resampled)
            
            # predict test
            y_pred = model.predict(X_test_fold) 

            accuracies.append(accuracy_score(y_test_fold, y_pred))
            weighted_f1s.append(f1_score(y_test_fold, y_pred, average='weighted'))
            weighted_precisions.append(precision_score(y_test_fold, y_pred, average='weighted', zero_division=0))
            weighted_recalls.append(recall_score(y_test_fold, y_pred, average='weighted', zero_division=0))

            # predict training
            y_train_pred = model.predict(X_train_resampled)
            train_f1 = f1_score(y_train_resampled, y_train_pred, average='weighted')
            train_accuracy = accuracy_score(y_train_resampled, y_train_pred)

            train_accuracies.append(train_accuracy)
            train_f1s.append(train_f1)

        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(weighted_f1s)
        avg_precision = np.mean(weighted_precisions)
        avg_recall = np.mean(weighted_recalls)
        avg_train_accuracy = np.mean(train_accuracies)
        avg_train_f1 = np.mean(train_f1s)

        print(f"Params: C={C}, gamma={gamma} --> "
            f"Accuracy: {avg_accuracy:.4f}, "
            f"Weighted F1: {avg_f1:.4f}, "
            f"Weighted Precision: {avg_precision:.4f}, "
            f"Weighted Recall: {avg_recall:.4f}")
    
        if avg_f1 > best_score:
            best_accuracy = avg_accuracy
            best_score = avg_f1
            best_precision = avg_precision
            best_recall = avg_recall
            best_train_acc = avg_train_accuracy
            best_train_f1 = avg_train_f1
            best_params = {'C': C, 'gamma': gamma}

    print("\nBest SVM configuration found:")
    print(best_params, "\n")
    print(f"Train Accuracy: {best_train_acc:.4f}, Train F1: {best_train_f1:.4f}")
    print(f"Test Accuracy: {best_accuracy:.4f}, F1 Score: {best_score:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    print(f"Performance Increase over Baseline Model: {((best_score-dummy_f1)/dummy_f1) * 100:.4f}%\n")

    '''
    REGRESSION PROBLEM (TIME TO EMPLOYMENT)
    '''
    y_reg = df['TimeToEmployment'].values

    # param grid for hyperparameter tuning
    param_grid_reg = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2]
    }

    reg_param_combinations = list(itertools.product(param_grid_reg['C'], param_grid_reg['epsilon']))

    best_reg_score = float('inf') 
    best_reg_params = None

    for C, epsilon in reg_param_combinations:
        loo = LeaveOneOut()
        mse_scores = []

        for train_index, test_index in loo.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_reg[train_index], y_reg[test_index]

            reg_model = SVR(C=C, epsilon=epsilon, kernel='rbf')
            reg_model.fit(X_train, y_train)
            y_pred = reg_model.predict(X_test)

            mse_scores.append(mean_squared_error(y_test, y_pred))

        avg_mse = np.mean(mse_scores)
        print(f"Reg Params: C={C}, epsilon={epsilon} --> Avg MSE: {avg_mse:.4f}")

        if avg_mse < best_reg_score:
            best_reg_score = avg_mse
            best_reg_params = {'C': C, 'epsilon': epsilon}

    print("\nBest SVR configuration found:")
    print(best_reg_params)
    print(f"Best LOO-CV MSE: {best_reg_score:.4f}")
    
    '''
    export models
    '''
    final_svm_model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
    final_svm_model.fit(X_scaled, y_encoded)

    final_svr_model = SVR(C=best_reg_params['C'], epsilon=best_reg_params['epsilon'], kernel='rbf')
    final_svr_model.fit(X_scaled, y_reg)

    # save to .pkl
    joblib.dump(final_svr_model, 'svr_model.pkl')
    joblib.dump(final_svm_model, 'svm_model.pkl')
    joblib.dump(sc_X, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

def main():
    load_and_preprocess_data('dataset.xlsx')

if __name__ == '__main__':
    main()
