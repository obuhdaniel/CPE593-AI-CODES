'''
Import Libraries
'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'Age': [25, 30, np.nan, 22, 35],
    'Income': [50000, 60000, 75000, np.nan, 80000],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
})

numeric_features = ['Age', 'Income']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Gender']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

transformed_data = preprocessor.fit_transform(data)

print("UnTransformed Data:", data)


print("Transformed Data:", transformed_data)
