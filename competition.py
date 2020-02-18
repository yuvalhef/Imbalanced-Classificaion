import sys
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import Imputer, OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from keras import optimizers
from keras.layers import Dropout
import keras
# from sklearn.pipeline import make_pipeline
# from tpot.builtins import ZeroCount
from sklearn.feature_selection import RFE, RFECV, univariate_selection
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
from mlxtend.regressor import StackingCVRegressor
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler
import xgboost
import lightgbm.sklearn as lgbm
import csv
import os
# import seaborn as sns
# sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from scipy.special import boxcox1p
# import seaborn as sns
from scipy import stats
# from yellowbrick.features.importances import FeatureImportances
from sklearn.ensemble.forest import RandomForestRegressor
# from bayes_opt import BayesianOptimization
# from bayes_opt import UtilityFunction
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def load__competition_ds():
    train = pd.read_csv('datasets/saftey_efficay_myopiaTrain.csv').iloc[:30451]
    testing_features = pd.read_csv('datasets/saftey_efficay_myopiaTest.csv')
    training_target = train.iloc[:, -1]
    training_features = train.iloc[:, :-1]
    return training_features, testing_features, training_target


def load_train_ds():
    ds = pd.read_csv('datasets/saftey_efficay_myopiaTrain.csv').iloc[:30451]
    attributes = ds.iloc[:, :-1]
    targets = ds.iloc[:, -1]
    train, testing_features, training_target, y_test = train_test_split(attributes, targets,
                                                                        train_size=0.8, test_size=0.2)
    return train, testing_features, training_target, y_test


def balance_ds(training_features, training_target, amount=12):

    training_features['Class'] = training_target.reset_index(drop=True)
    df_majority = training_features[training_features['Class'] == 0].reset_index(drop=True)
    df_minority = training_features[training_features['Class'] == 1].reset_index(drop=True)
    df_majority = df_majority.sample(df_minority.shape[0] * 5, replace=False)
    training_features = df_majority.append(df_minority).reset_index(drop=True)
    training_features = shuffle(training_features).reset_index(drop=True)
    training_target = training_features['Class']
    training_features = training_features.iloc[:, :-1]

    return training_features, training_target


# use straight after balance_ds
def get_dummies(train, testing_features):
    unified_ds = train.append(testing_features).reset_index(drop=True)
    dum_unified = pd.get_dummies(unified_ds)
    training_features = dum_unified[:train.shape[0]].reset_index(drop=True)
    testing_features = dum_unified[train.shape[0]:].reset_index(drop=True)
    return training_features, testing_features


# use straight after balance_ds if didn't use get_dummies
def one_hot_encoder(training_features, testing_features):
    unified_ds = training_features.append(testing_features).reset_index(drop=True)
    enc = OneHotEncoder()
    enc.fit(unified_ds.values)
    training_features = enc.transform(training_features.values)
    testing_features = enc.transform(testing_features.values)
    return training_features, testing_features


# Impute missing values
def missing_imputer(training_features, testing_features, strategy='median'):
    imputer = Imputer(strategy="mean")
    imputer.fit(training_features)
    training_features = imputer.transform(training_features)
    testing_features = imputer.transform(testing_features)
    return training_features, testing_features


def min_max_scaler(training_features, testing_features):
    scaler = MinMaxScaler()
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    testing_features = scaler.transform(testing_features)
    return training_features, testing_features


def robust_scaler(training_features, testing_features):
    scaler = RobustScaler()
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    testing_features = scaler.transform(testing_features)
    return training_features, testing_features


def std_scaler(training_features, testing_features):
    scaler = StandardScaler()
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    testing_features = scaler.transform(testing_features)
    return training_features, testing_features


def max_abs_scaler(training_features, testing_features):
    scaler = MaxAbsScaler()
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    testing_features = scaler.transform(testing_features)
    return training_features, testing_features


# Kernel PCA - not helpful
def kernel_pca(training_features, testing_features, n_components=15):
    pca = KernelPCA(n_components=15)
    pca.fit(training_features)
    training_features = pca.transform(training_features)
    testing_features = pca.transform(testing_features)
    return training_features, testing_features


def pca(training_features, testing_features, n_components=15):
    pca = PCA(n_components=15)
    pca.fit(training_features)
    training_features = pca.transform(training_features)
    testing_features = pca.transform(testing_features)
    return training_features, testing_features


def pool_cat(training_features, testing_features, training_target, y_test):

    training_features = training_features.replace(np.nan, '', regex=True)
    testing_features = testing_features.replace(np.nan, '', regex=True)
    train = Pool(data=training_features, cat_features=[1, 2, 4, 13, 32, 33, 34, 38, 41, 47, 48], label=training_target)
    test = Pool(data=testing_features, cat_features=[1, 2, 4, 13, 32, 33, 34, 38, 41, 47, 48])
    model = CatBoostClassifier(iterations=1000, loss_function='Logloss')
    # Fit model
    model.fit(train)
    preds_proba = model.predict_proba(test)[:, 1]
    pred = pd.DataFrame(columns=['Id', 'Class']).reset_index(drop=True)
    pred.Id = np.arange(1, len(preds_proba) + 1)
    pred.Class = preds_proba
    pred.to_csv('cat.csv', index=False)
    # fpr, tpr, thresholds = roc_curve(y_test, preds_proba, pos_label=1)
    # print(auc(fpr, tpr))
    # test_score = auc(fpr, tpr)
    # # cv_score = cv_evaluation(model, training_features, training_target, testing_features, y_test)
    # cv_score = 0
    # with open(r'datasets/results.csv', 'a', newline='') as csvfile:
    #     fieldnames = ['test_score', 'cv_score', 'cls', 'preprocess']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writerow({'test_score': str(test_score), 'cv_score': str(cv_score), 'cls': str(cls), 'preprocess': str(preproc_comp)})
    return train, test


def remove_by_nans(training_features, testing_features, prec=0.6):
    unified_ds = training_features.append(testing_features).reset_index(drop=True)
    dum_unified = unified_ds.dropna(axis=1, thresh=prec*unified_ds.shape[0])
    training_features = dum_unified[:training_features.shape[0]].reset_index(drop=True)
    testing_features = dum_unified[training_features.shape[0]:].reset_index(drop=True)
    return training_features, testing_features


def rt_embedding(training_features, testing_features):
    rt = RandomTreesEmbedding()
    rt.fit(training_features)
    testing_features = rt.transform(testing_features)
    training_features = rt.transform(training_features)
    return training_features, testing_features


def rfe(training_features, training_target, model):
    selector = RFE(model)
    return selector


def cv_evaluation(model, training_features, training_target, testing_features, y_test, cv=5):
    training_features = np.append(training_features, testing_features, axis=0)
    training_target = np.append(training_target, y_test, axis=0)
    score = cross_val_score(model, training_features, training_target, cv=cv, scoring='roc_auc')
    return np.mean(score)


def train_test_evaluation(model, training_features, training_target, testing_features, y_test):
    # pipe = make_pipeline(RandomOverSampler(ratio=0.8), model)
    model.fit(training_features, training_target)
    # results = model.predict_proba(testing_features)[:, 1]
    results = model.predict(testing_features)
    fpr, tpr, thresholds = roc_curve(y_test, results, pos_label=1)
    return auc(fpr, tpr)


def produce_submission(model, training_features, training_target, testing_features, name='datasets/6latest.csv'):
    model.fit(training_features, training_target)
    # results = model.predict_proba(testing_features)[:, 1]
    results = model.predict(testing_features)
    pred = pd.DataFrame(columns=['Id', 'Class']).reset_index(drop=True)
    pred.Id = np.arange(1, len(results) + 1)
    pred.Class = results
    pred.to_csv(name, index=False)


classifier_components = {
    1: xgboost.XGBClassifier,
    2: KNeighborsClassifier,
    3: RandomForestClassifier,
    4: ExtraTreesClassifier,
    5: GradientBoostingClassifier,
    6: StackingClassifier,
    7: GaussianNB,
    # 8: RandomForestClassifierNB,
    10: SVC,
    11: CatBoostClassifier,
    12: lgbm.LGBMClassifier,
    13: AdaBoostClassifier,
    14: BalancedRandomForestClassifier,
    15: RUSBoostClassifier,
    16: BalancedBaggingClassifier,
    # 17: RotationForestClassifier,
    18: xgboost.XGBRegressor,
    19: AdaBoostRegressor
}

preprocess_components = {
    2: one_hot_encoder,
    3: missing_imputer,
    4: kernel_pca,
    5: min_max_scaler,
    6: pca,
    7: rt_embedding,
    8: robust_scaler,
    9: std_scaler,
    10: max_abs_scaler,
    11: remove_by_nans,
    12: pool_cat
}

feat_selec_components = {
    1: RFE,
    2: RFECV
}


def preprocess(components_list, training_features, testing_features):
    for component in components_list:
        training_features, testing_features = preprocess_components[component](training_features, testing_features)
    return training_features, testing_features


def NN(training_features, training_target, testing_features, y_test):
    input_dim = training_features.shape[1]
    model = Sequential()
    class_weight = {0: 1.,
                    1: 10.}
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
    model.add(Dense(12, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
    # The Hidden Layers :
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))

    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mae'])
    model.fit(training_features, training_target, epochs=200, batch_size=2, validation_split=0.2, class_weight=class_weight, callbacks=[early])
    results = model.predict(testing_features)
    fpr, tpr, thresholds = roc_curve(y_test, results, pos_label=1)
    print(auc(fpr, tpr))
    return model

training_features, testing_features, training_target, y_test = load_train_ds()
preproc_comp = [3, 5]
# training_features, training_target = balance_ds(training_features, training_target, amount=10)
training_features, testing_features = remove_by_nans(training_features, testing_features)

training_features, testing_features = get_dummies(training_features, testing_features)
training_features, testing_features = preprocess(preproc_comp, training_features, testing_features)
NN(training_features, training_target, testing_features, y_test)



# train_df = pd.read_csv('../input/saftey_efficay_myopiaTrain.csv').iloc[:30451]
x_train, x_test, y_train, y_test = load_train_ds()

x_train, x_test = get_dummies(x_train, x_test)
x_train, x_test = remove_by_nans(x_train, x_test)
x_train, x_test = missing_imputer(x_train, x_test)
# x_train, x_test = pol_features(x_train, x_test)

# x_train, y_train = random_over_sampling(x_train, y_train)

#Create a FeatureImportances figure
#FeatureImportances_analysis(x_test, y_test)
Stacking=True

if Stacking==True:
    np.random.seed(5)
    model = StackingCVRegressor(regressors=(RandomForestRegressor(random_state=5), ExtraTreesRegressor(random_state=5), GradientBoostingRegressor(random_state=5), xgboost.XGBRegressor(learning_rate=0.01, max_depth=6, colsample_bytree=1, subsample=0.8, gamma=0, random_seed=5)),
                                meta_regressor=xgboost.XGBRegressor(learning_rate=0.01, max_depth=7, colsample_bytree=1, subsample=0.8, gamma=0, random_seed=5))
else:
    model = xgboost.XGBRegressor(learning_rate=0.01, max_depth=7, colsample_bytree=1, subsample=0.8, gamma=0, random_seed=5)

test_score = train_test_evaluation(model, x_train, y_train, x_test, y_test)
# cv_score = cv_evaluation(model, x_train, y_train, x_test, y_test)
print('Test Set Score: '+str(test_score))
# print('CV Score: '+str(cv_score))


print('hi')

if __name__ == '__main__':

    # Choose pre-processing components from preprocess_components dict
    preproc_comp = [11, 3]
    feat_selec_comp = 2
    # Choose a classifier
    cls = 18
    dummies = True

    ######################## TRAINING #######################
    print('starting training')
    # Necessary pre-processing, do not change
    training_features, testing_features, training_target, y_test = load_train_ds()
    if dummies:
        training_features, testing_features = get_dummies(training_features, testing_features)
    if cls == 11:
        training_features, testing_features = pool_cat(training_features, testing_features, training_target, y_test)
        sys.exit()
    training_features, training_target = balance_ds(training_features, training_target, amount=6)

    # Perform additional pre-process
    training_features, testing_features = preprocess(preproc_comp, training_features, testing_features)

    # Fit the chosen classifier

    model = StackingClassifier(classifiers=[xgboost.sklearn.XGBClassifier(n_estimators=100),
                                            # classifier_components[5](n_estimators=30, min_samples_leaf=15),
                                            GaussianNB()],
                                            meta_classifier=xgboost.XGBRegressor(n_estimators=30), use_probas=True,
                                            average_probas=False)

    # Evaluate
    test_score = train_test_evaluation(model, training_features, training_target, testing_features, y_test)
    # cv_score = cv_evaluation(model, training_features, training_target, testing_features, y_test)
    cv_score = 0

    ################ Produce submission file ##################
    training_features_t, testing_features_t, training_target_t = load__competition_ds()
    training_features_t, testing_features_t = remove_by_nans(training_features_t, testing_features_t)
    if dummies:
        training_features_t, testing_features_t = get_dummies(training_features_t, testing_features_t)
    if cls == 11:
        training_features_t, testing_features_t = pool_cat(training_features_t, testing_features_t, training_target_t, None)
        sys.exit()

    training_features_t, training_target_t = balance_ds(training_features_t, training_target_t, amount=6)

    training_features_t, testing_features_t = preprocess(preproc_comp, training_features_t, testing_features_t)

    model_t = StackingClassifier(classifiers=[xgboost.sklearn.XGBClassifier(n_estimators=40),
                                            # classifier_components[5](n_estimators=30, min_samples_leaf=15),
                                            GaussianNB()],
                               meta_classifier=xgboost.XGBRegressor(n_estimators=30), use_probas=True,
                               average_probas=False)

    produce_submission(model_t, training_features_t, training_target_t, testing_features_t, 'datasets/17_12.csv')

train_df = pd.read_csv('../input/saftey_efficay_myopiaTrain.csv').iloc[:30451]
x_train, x_test, y_train, y_test = load_train_ds(train_df)
