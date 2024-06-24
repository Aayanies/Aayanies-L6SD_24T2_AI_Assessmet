import numpy as pd
import matplotlib as mp
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression as lr, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
import joblib



# dataframe = pd.read_excel('Net_Worth_Data.xlsx')
df = pd.read_excel('Net_Worth_Data.xlsx')  # Importing dataset

print("Printing First 5 rows")
print(df.head(5)) #Giving first 5 of data

print("Printing Last 5 rows ")
print(df.tail(5)) #Giving last 5 of data

print('Rows and colums', df.shape)
print('rows', df.shape[0])
print('colums', df.shape[1])

# Displaying summary of dataset

print("Printing a concise Summary")
print(df.info()) # Printing the information

# Dropping irrelevant colums that aren't needed

x = df.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country'], axis=1)
print(x)

# Transform the input dataset into percentagrs based wieghed  between 0-1
scX= MinMaxScaler()
X_Scaled= scX.fit_transform(x)
print(X_Scaled)
# Creating the output dateset from the orginal dataset

# it store output data into Y varibale

y = df['Net Worth']
print(y)
print(f" Printing input values\n{y}")

# Transform the input dataset into percentagrs based wieghed  between 0-1
scY= MinMaxScaler()
Y_Reshape = y.values.reshape(-1, 1)
Y_Scaled= scY.fit_transform(Y_Reshape)
print(Y_Scaled)


# Print a few rows of the scaled into dataset (x)

print(f"Printing first 10 data base into scale:\n {X_Scaled[:10]}")

# Print a few rows of the scaled into dataset (Y)

print(f"Printing last 10 data base into scale:\n {Y_Scaled[:10]}")



# Printing the few rows into scale input data

print(X_Scaled.shape)
print(Y_Scaled.shape)

# Printing the shape of test and training data
sns.scatterplot(X_Scaled)
mp.show()

sns.scatterplot(Y_Scaled)
mp.show()

# Imports and initialize models

lr = lr()  
ridge = Ridge()
lasso = Lasso()
elasticNet = ElasticNet()
svr = SVR()
dtree = DecisionTreeRegressor()
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
knr = KNeighborsRegressor()
mlp = MLPRegressor(random_state=42, max_iter=1000)

# Spliting the data into training & testing sets
# X_Train = x[:80]
# Y_Train = x[:80]
# X_Test = x[:20]
# Y_Test = x[:20]
 
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Training models using training data
lr.fit(X_Train, Y_Train)
ridge.fit(X_Train, Y_Train)
lasso.fit(X_Train, Y_Train)
elasticNet.fit(X_Train, Y_Train)
svr.fit(X_Train, Y_Train)
dtree.fit(X_Train, Y_Train)
rfr.fit(X_Train, Y_Train)
gbr.fit(X_Train, Y_Train)
knr.fit(X_Train, Y_Train)
mlp.fit(X_Train, Y_Train)

# Predicting on test data
lr_preds = lr.predict(X_Test)
ridge_preds = ridge.predict(X_Test)
lasso_preds = lasso.predict(X_Test)
elasticNet_preds = elasticNet.predict(X_Test)
svr_preds = svr.predict(X_Test)
dtree_preds = dtree.predict(X_Test)
rfr_preds = rfr.predict(X_Test)
gbr_preds = gbr.predict(X_Test)
knr_preds = knr.predict(X_Test)
mlp_preds = mlp.predict(X_Test)

# Evaluating model performance
lr_rmse = mean_squared_error(Y_Test, lr_preds, squared=False)
ridge_rmse = mean_squared_error(Y_Test, ridge_preds, squared=False)
lasso_rmse = mean_squared_error(Y_Test, lasso_preds, squared=False)
elasticNet_rmse = mean_squared_error(Y_Test, elasticNet_preds, squared=False)
svr_rmse = mean_squared_error(Y_Test, svr_preds, squared=False)
dtree_rmse = mean_squared_error(Y_Test, dtree_preds, squared=False)
rfr_rmse = mean_squared_error(Y_Test, rfr_preds, squared=False)
gbr_rmse = mean_squared_error(Y_Test, gbr_preds, squared=False)
knr_rmse = mean_squared_error(Y_Test, knr_preds, squared=False)
mlp_rmse = mean_squared_error(Y_Test, mlp_preds, squared=False)

# Displaying evaluation results
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Ridge Regression RMSE: {ridge_rmse}")
print(f"Lasso Regression RMSE: {lasso_rmse}")
print(f"Elastic Net Regression RMSE: {elasticNet_rmse}")
print(f"SVR Regression RMSE: {svr_rmse}")
print(f"Decision Tree Regression RMSE: {dtree_rmse}")
print(f"Random Forest Regression RMSE: {rfr_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"K-Neighbors Regressor RMSE: {knr_rmse}")
print(f"MLP Regressor RMSE: {mlp_rmse}")
 
# Plotting RMSE values for each model
models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression', 'SVR Regression',
          'Decision Tree Regression', 'Random Forest Regression', 'Gradient Boosting Regressor', 'K-Neighbors Regressor', 'MLP Regressor']
rmse_values = [lr_rmse, ridge_rmse, lasso_rmse, elasticNet_rmse, svr_rmse, dtree_rmse, rfr_rmse, gbr_rmse, knr_rmse, mlp_rmse]
 
mp.figure(figsize=(10, 6))
mp.barh(models, rmse_values, color='Pink')
mp.xlabel('RMSE')
mp.title('RMSE of Different Regression Models')
mp.gca().invert_yaxis()  # To display the highest RMSE on top
mp.show()

# saving the model
filename = 'Gbr_model.sav'
joblib.dump(gbr, filename)

# loading the model 
loaded_model = joblib.load(filename)

