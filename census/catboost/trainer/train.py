#Google Cloud Libraries
from google.cloud import storage


#System Libraries
import datetime
import subprocess

#Data Libraries
import pandas as pd
import numpy as np

#ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv
from catboost import CatBoost, Pool

from catboost.utils import get_gpu_device_count
print('I see %i GPU devices' % get_gpu_device_count())


# Fill in your Cloud Storage bucket name
BUCKET_ID = "mchrestkha-demo-env-ml-examples"

census_data_filename = 'adult.data.csv'

# Public bucket holding the census data
bucket = storage.Client().bucket('cloud-samples-data')

# Path to the data inside the public bucket
data_dir = 'ai-platform/census/data/'

# Download the data
blob = bucket.blob(''.join([data_dir, census_data_filename]))
blob.download_to_filename(census_data_filename)

# these are the column labels from the census data files
COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income-level'
)
# categorical columns contain data that need to be turned into numerical values before being used by XGBoost
CATEGORICAL_COLUMNS = (
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
)

# Load the training census dataset
with open(census_data_filename, 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)
# remove column we are trying to predict ('income-level') from features list
X = raw_training_data.drop('income-level', axis=1)
# create training labels list
#train_labels = (raw_training_data['income-level'] == ' >50K')
y = raw_training_data['income-level']

# Since the census data set has categorical features, we need to convert
# them to numerical values.
# convert data in categorical columns to numerical values
X_enc=X
encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
for col in CATEGORICAL_COLUMNS:
    X_enc[col] = encoders[col].fit_transform(X[col])
        
        
y_enc=LabelEncoder().fit_transform(y)

X_train, X_validation, y_train, y_validation = train_test_split(X_enc, y_enc, train_size=0.75, random_state=42)


#model = CatBoost({'iterations':50})
model=CatBoostClassifier(
        od_type='Iter'
#iterations=5000,
#custom_loss=['Accuracy']
)
model.fit(
    X_train,y_train,eval_set=(X_validation, y_validation),

    verbose=50)

# # load data into DMatrix object
# dtrain = xgb.DMatrix(train_features, train_labels)
# # train model
# bst = xgb.train({}, dtrain, 20)


# Export the model to a file
fname = 'catboost_census_model.onnx'
model.save_model(fname, format='onnx')

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('census/catboost_model_dir/catboost_census_%Y%m%d_%H%M%S'),
    fname))
blob.upload_from_filename(fname)