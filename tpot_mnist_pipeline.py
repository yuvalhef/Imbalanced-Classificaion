import sys
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from RotationAlgorithms.rotation_forest import RotationForestClassifier
from RandomForestClassifierGaussianNB_leaves import RandomForestClassifierNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, AdaBoostClassifier, AdaBoostRegressor
from sklearn.preprocessing import Imputer, OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import make_pipeline
# from tpot.builtins import ZeroCount
from sklearn.feature_selection import RFE, RFECV, univariate_selection
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
from catboost import CatBoostClassifier, Pool
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
from tpot.builtins import ZeroCount

# NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)

ds = pd.read_csv('datasets/saftey_efficay_myopiaTrain.csv').iloc[:30451]


attributes = ds.iloc[:, :-1]
targets = ds.iloc[:, -1]
train, testing_features, training_target, y_test = train_test_split(attributes, targets,
                                                    train_size=0.8, test_size=0.2)

unified_ds = train.append(testing_features).reset_index(drop=True)
dum_unified = pd.get_dummies(unified_ds)
training_features = dum_unified[:train.shape[0]].reset_index(drop=True)
testing_features = dum_unified[train.shape[0]:].reset_index(drop=True)


# training_features['Class'] = training_target.reset_index(drop=True)
# df_majority = training_features[training_features['Class'] == 0].reset_index(drop=True)
# df_minority = training_features[training_features['Class'] == 1].reset_index(drop=True)
#
# df_majority = df_majority.sample(df_minority.shape[0]*5, replace=False)
# training_features = df_majority.append(df_minority).reset_index(drop=True)
#

# training_target = training_features['Class']
# training_features = training_features.iloc[:, :-1]



imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

scaler = MinMaxScaler()
scaler.fit(training_features)
training_features = scaler.transform(training_features)
testing_features = scaler.transform(testing_features)

# pca = KernelPCA(n_components=15)
# pca.fit(training_features)
# training_features = pca.transform(training_features)
# testing_features = pca.transform(testing_features)

# model = xgboost.sklearn.XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=8, min_child_weight=13, nthread=1, subsample=0.8)
model = StackingClassifier(classifiers=[xgboost.sklearn.XGBClassifier(n_estimators=40),
                                       # classifier_components[5](n_estimators=30, min_samples_leaf=15),
                                       GaussianNB()],
                                       meta_classifier=xgboost.XGBRegressor(n_estimators=50), use_probas=True,
                                       average_probas=False)

# exported_pipeline = make_pipeline(
#
#     RFE(estimator=GradientBoostingClassifier(max_features=0.3, n_estimators=40), step=0.3),
#     xgboost.XGBClassifier(learning_rate=0.01, max_depth=8, max_features=0.5, min_samples_leaf=19, min_samples_split=4, n_estimators=40, subsample=0.8)
# )

model.fit(training_features, training_target)
# results = model.predict_proba(testing_features)[:, 1]
results = model.predict(testing_features)
fpr, tpr, thresholds = roc_curve(y_test, results, pos_label=1)
print(auc(fpr, tpr))
# print(auc(results, y_test))

print('bi')