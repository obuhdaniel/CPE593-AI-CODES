'''
Step 1: Import Libraries
'''

#this libraries are for the basic data processing and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#this libraries are for the advanced data visualization and Outlier Handling
import seaborn as sns
from scipy import stats
import sklearn
from sklearn.datasets import load_diabetes #This is the sklearn library data set for outliers detection

#this libraries are for the Normalizationa nd Standardization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#this libraries are for the PCA and data transformation
from sklearn.decomposition import PCA

'''
Step 2: Load diabetes dataset
'''

df = pd.read_csv('ANN\Data Processing\diabetes.csv')
# Display the first few rows of the dataset
print(df.head())


# Load the second dataset from sklearn
diabetics = load_diabetes()\

column_name = diabetics.feature_names
# Create a DataFrame from the diabetes dataset
df_diabetics = pd.DataFrame(data=diabetics.data, columns=column_name)
print(df_diabetics.head())

'''
Step 3: Check Data Info
'''
df.info()


'''
Step 4: Check Null values
'''

df.isnull().sum()

'''
Step 5: Statistical Analysis
'''
df.describe()
# Display the shape of the dataset
print("Shape of the dataset:", df.shape)
# Display the summary statistics of the dataset
print("Summary statistics:\n", df.describe())
# Display the number of missing values in each column
print("Missing values:\n", df.isnull().sum())


'''
Step 6: Check for Outliers
'''

# Create a boxplot to visualize the distribution of each feature
sns.boxplot(df_diabetics['bmi'])

#create a function to remove the box_plot
def removal_box_plot(df, column, threshold): 
    sns.boxplot(df[column]) 
    plt.title(f'Original Box Plot of {column}') 
    plt.show() 
  
    removed_outliers = df[df[column] <= threshold] 
  
    sns.boxplot(removed_outliers[column]) 
    plt.title(f'Box Plot without Outliers of {column}') 
    plt.show() 
    return removed_outliers 
  
threshold_value = 0.12 
removed_df = removal_box_plot(df_diabetics, 'bmi', threshold_value)

'''
Step 7: Drop Outliers in colums comnbined
'''

outlier_indices = np.where((df_diabetics['bmi'] > 0.12) & (df_diabetics['bp'] < 
0.8)) 
no_outliers = df_diabetics.drop(outlier_indices[0]) 
# Scatter plot without outliers 
fig, ax_no_outliers = plt.subplots(figsize=(6, 4)) 
ax_no_outliers.scatter(no_outliers['bmi'], no_outliers['bp']) 
ax_no_outliers.set_xlabel('(body mass index of people)') 
ax_no_outliers.set_ylabel('(bp of the people )') 
plt.show() 


'''
Step 8: Check for Z-Score
'''
z = np.abs(stats.zscore(df_diabetics['age']))
print("Z-scores:\n", z)



'''
Step 9: Check for Correlation
'''

corr = df.corr()
plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()

'''
Step 10: Check Outcomes Proportionality
'''
plt.pie(df['Outcome'].value_counts(), labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%')
plt.title('Proportionality of Outcomes')
plt.show()


#Now lets seperate independent featurea and Target variables
#This means lets get our X and Y values to train on

X = df.drop(columns=['Outcome'])
Y = df['Outcome']
'''
Step 11: Normalization/ Standardization
'''

#init MINMAX scaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)
print("Rescaled data:\n", rescaled_X[:5])

#init Standard Scaler
scaler = StandardScaler().fit(X)
rescaled_X = scaler.transform(X)
print("Standardized data:\n", rescaled_X[:5])

#Standardization is the process of rescaling the features so that they have a mean of 0 and a standard deviation of 1.

'''
Step 12: Data Transformation and Reduction

PCA stands for Principal Component Analysis. It is a technique used to reduce the dimensionality of a dataset while preserving as much variance as possible. PCA transforms the original features into a new set of uncorrelated features called principal components.
'''

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
print("Reduced data:\n", reduced_data)

