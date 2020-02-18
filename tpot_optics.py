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
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was:0.6512769669063172
exported_pipeline = make_pipeline(
    ZeroCount(),
    RFE(estimator=RandomForestClassifier(max_features=0.45, n_estimators=20), step=0.3),
    xgboost.XGBClassifier(learning_rate=0.001, max_depth=4, max_features=0.5, min_samples_leaf=19, min_samples_split=4, n_estimators=20, subsample=0.8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
