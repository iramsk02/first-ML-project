import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

housing = pd.read_csv("dataset.csv")
# Remove leading/trailing spaces from column names
housing.columns = housing.columns.str.strip()

#housing.head()
#housing.info()
#housing['CHAS'].value_counts()
#housing.describe()

# For plotting histogram
housing.hist(bins=50, figsize=(20, 15))
plt.show()  # <--- Add this line to display the histogram

# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:] 
#     return data.iloc[train_indices], data.iloc[test_indices]
#train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):  # Remove leading space in 'CHAS'
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# strat_test_set['CHAS'].value_counts()
# strat_train_set['CHAS'].value_counts()


housing = strat_train_set.copy()
#looking for correlations
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
attributes = ["MEDV", "RM", "ZN", "LSTAT"]

scatter_matrix(housing[attributes], figsize = (12,8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)
plt.show()

#adding attribute
# housing["TAXRM"] = housing['TAX']/housing[RM']
# housing.head()
# corr_matrix = housing.corr()
# corr_matrix['MEDV'].sort_values(ascending=False)
#housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Missing Attributes

# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)

# a = housing.dropna(subset=["RM"]) #Option 1
# a.shape
# Note that the original housing dataframe will remain unchanged

# housing.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged

median = housing["RM"].median() # Compute median for Option 3
housing["RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged
housing.shape
housing.describe() # before we started filling missing attributes

imputer = SimpleImputer(strategy="median")
imputer.fit(housing)
imputer.statistics_
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()

# Scikit-learn Design
# Primarily, three types of objects
# 1. Estimators - It estimates some parameter based on a dataset.
#        Eg. imputer. It has a fit method and transform method.
#        Fit method - Fits the dataset and calculates internal parameters

# 2. Transformers - transform method takes input and returns output based on the learnings from fit().
#        It also has a convenience function called fit_transform() which fits and then transforms.

# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. 
#          It also gives score() function which will evaluate the predictions.

# # Feature Scaling
# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     (value - min)/(max - min)
#     Sklearn provides a class called MinMaxScaler for this
    
# 2. Standardization
#     (value - mean)/std
#     Sklearn provides a class called StandardScaler for this

## Creating a Pipeline

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr.shape

## Selecting a desired model for Dragon Real Estates
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)

# Evaluating the model

housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print(rmse)

# Using better evaluation technique - Cross Validation

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)

# def print_scores(scores):
#     print("Scores:", scores)
#     print("Mean: ", scores.mean())
#     print("Standard deviation: ", scores.std())
def print_scores(scores, model_name=model):
    # Check current directory to ensure the file is being created where expected
    print(f"Current Directory: {os.getcwd()}")

    file_path = "modeloutput.txt"
    try:
        # Open file in write mode to save scores and include model name at the top
        with open(file_path, "w") as file:
            print("File opened successfully for writing.")

            # Write the model name at the top of the file
            file.write(f"Model: {model_name}\n")
            file.write("="*50 + "\n")  # Add separator for clarity

            # Print and write scores to the file
            print(f"Model: {model_name}")
            file.write(f"Scores: {scores}\n")
            print("Scores:", scores)

            print("Mean: ", scores.mean())
            file.write(f"Mean: {scores.mean()}\n")

            print("Standard deviation: ", scores.std())
            file.write(f"Standard deviation: {scores.std()}\n")

            file.write("="*50 + "\n")  # End separator for clarity

        print(f"Output successfully written to {file_path}.")
        
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
print_scores(rmse_scores)


#testing on test data

X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))

print(final_rmse)
prepared_data[0]

## Using the model

from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)
# model.predict(features)
predicted_price = model.predict(features)
print(predicted_price)





# Unprocessed features and actual prices from the test set
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"]


best_match_idx = None
best_diff = np.inf

for idx, row in X_test.iterrows():
    diff = np.linalg.norm(row.values - features)
    if diff < best_diff:
        best_diff = diff
        best_match_idx = idx

# Now, we can compare the actual price with the predicted price
actual_price = Y_test.loc[best_match_idx]
print(f"Actual price: {actual_price}")
print(f"Difference: {abs(predicted_price[0] - actual_price)}")
