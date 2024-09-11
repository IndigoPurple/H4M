# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import csv
import sys

from sklearn.datasets import load_boston

# Initializing the data
csv_file = open('./data/dsaa_dataset_order_rename.csv')
rows = csv.reader(csv_file)
rows = list(rows)
data = pd.DataFrame(rows[1:]).astype(float)
pd.set_option('display.max_columns', None)

# __console = sys.stdout
# log_location = 'all_in.log'
# log = open(log_location,'a+')
# sys.stdout = log

# preprocessing and check the data
print(data.shape)
data.columns = rows[0]
data = data.drop('id', axis=1)
data = data.drop('TtlPrc', axis=1)
# data = data.rename(columns={"TtlPrc": "TotalPrice"})
data = data.rename(columns={"UntPrc": "Price"})

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

plt.clf()
# sns.distplot(data['TotalPrice'])
# plt.title("Histogram of Total Price")
# plt.xlabel("Total Price")
# plt.ylabel("Frequency")
# plt.show()
# exit()
#
# # the histogram of the data
# n, bins, patches = plt.hist(data['TotalPrice'])
#
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# # plt.xlim(40, 160)
# # plt.ylim(0, 0.03)
# plt.grid(True)
# plt.show()
#
# exit()
# print(data.head())
# print(data.dtypes)
# print(data.isnull().sum())

# Viewing the data statistics
print(data.describe(include='all'))
lat_min, lat_max = data['Lat'].min(), data['Lat'].max()
lng_min, lng_max = data['Lng'].min(), data['Lng'].max()
print(data.sort_values(by='Lat', ascending=True))
print(data.sort_values(by='Lng', ascending=True))
# exit()

'''scatter plot'''
data = data.loc[data['Lat']>36]
data = data.loc[data['Lat']<40.5]
data = data.loc[data['Lng']>115]
data = data.loc[data['Lng']<117.25]
######
data = data.loc[data['Lat']<40.3]
data = data.loc[data['Lng']>116]
data = data.loc[data['Lng']<116.8]
#####
lat_min, lat_max = 39.7145817, 40.1626081
lng_min, lng_max = 116.1186218, 116.6802978
data = data.loc[data['Lat']>lat_min]
data = data.loc[data['Lat']<lat_max]
data = data.loc[data['Lng']>lng_min]
data = data.loc[data['Lng']<lng_max]
price_max, price_min = data['Price'].max(), data['Price'].min()
# T = (data['Price']-price_min)/(price_max-price_min)

# T = data['Price']
# plt.scatter(data['Lng'], data['Lat'], marker='.', s=1, vmin=price_min, vmax=price_max,c=T, cmap='plasma')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# # plt.colorbar()
# cbar = plt.colorbar()
# cbar.ax.set_ylabel('Unit Price')
# plt.tight_layout()
# plt.show()

# exit()

##########
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

T = data['Price']

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.15, hspace=0.1)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

# use the previously defined function
# ax.scatter(data['Lng'], data['Lat'], marker='.', s=1, vmin=price_min, vmax=price_max,c=T, cmap='plasma')
ax.scatter(data['Lng'], data['Lat'], marker='.', s=1, color='orange')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
binwidth = 0.008
x, y = data['Lng'], data['Lat']
ax_histx.hist(data['Lng'], bins=np.arange(min(x), max(x)+binwidth, binwidth))
ax_histy.hist(data['Lat'], orientation='horizontal', bins=np.arange(min(y), max(y)+binwidth, binwidth))

plt.tight_layout()
plt.show()
exit()

# Finding out the correlation between the features
corr = data.corr()
print(corr.shape)

# Plotting the heatmap of correlation between features
# plt.figure(figsize=(20,20))
# sns.set(font_scale=1.5)
# sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# plt.show()

# Spliting target variable and independent variables
X = data.drop(['Price'], axis = 1)
y = data['Price']

# Splitting to training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

print('------------Linear Regression-----------------')
# Import library for Linear Regression
from sklearn.linear_model import LinearRegression

# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)

# Value of y intercept
print(lm.intercept_)

#Converting the coefficient values to a dataframe
coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T
coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
print(coeffcients)

# Model prediction on train data
y_pred = lm.predict(X_train)
# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# # Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()
#
# # Checking residuals
# plt.clf()
# plt.scatter(y_pred,y_train-y_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.show()
#
# # Checking Normality of errors
# plt.clf()
# sns.distplot(y_train-y_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

# Predicting Test data with the model
y_test_pred = lm.predict(X_test)
# Model Evaluation
acc_linreg = metrics.r2_score(y_test, y_test_pred)
mae_linreg = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_linreg = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_linreg)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_linreg)

# Visualizing the differences between actual prices and predicted values
plt.clf()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices", labelpad=1.5)
plt.title("Prices vs Predicted prices")
plt.show()

# Checking residuals
plt.clf()
plt.scatter(y_test_pred,y_test-y_test_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals", labelpad=1.5)
plt.show()

# Checking Normality of errors
plt.clf()
sns.distplot(y_test-y_test_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

'''
print('------------Random Forest Regressor-----------------')
# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()

# Checking residuals
# plt.clf()
# plt.scatter(y_pred,y_train-y_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.show()

# Checking Normality of errors
# plt.clf()
# sns.distplot(y_train-y_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

# Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_rf = metrics.r2_score(y_test, y_test_pred)
mae_rf = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_rf)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_rf)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_rf)
'''

print('------------XGBoost Regressor-----------------')
# Import XGBoost Regressor
from xgboost import XGBRegressor

#Create a XGBoost Regressor
reg = XGBRegressor()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()

# Checking residuals
# plt.scatter(y_pred,y_train-y_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.show()

# Checking Normality of errors
# plt.clf()
# sns.distplot(y_train-y_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

#Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_test_pred)
mae_xgb = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_xgb)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_xgb)

# Visualizing the differences between actual prices and predicted values
plt.clf()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices", labelpad=1.5)
plt.title("Prices vs Predicted prices")
plt.show()

# Checking residuals
plt.clf()
plt.scatter(y_test_pred,y_test-y_test_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals", labelpad=1.5)
plt.show()

# Checking Normality of errors
plt.clf()
sns.distplot(y_test-y_test_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

from xgboost import plot_importance
plt.clf()
fig,ax = plt.subplots(figsize=(10,10))
plot_importance(reg,height=0.5,max_num_features=64,ax=ax)
plt.show()


'''
print('------------SVM Regressor-----------------')
# Creating scaled set to be used in model to improve our results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Import SVM Regressor
from sklearn import svm

# Create a SVM Regressor
reg = svm.SVR()
# Train the model using the training sets
reg.fit(X_train, y_train)

# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()

# Checking residuals
# plt.scatter(y_pred,y_train-y_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.show()

# Checking Normality of errors
# plt.clf()
# sns.distplot(y_train-y_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

# Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_svm = metrics.r2_score(y_test, y_test_pred)
mae_svm = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_svm = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_svm)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_svm)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_svm)
'''

# print('------------Evaluation and comparision of all the models-----------------')
# models = pd.DataFrame({
#     'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines'],
#     'R-squared Score': [acc_linreg*100, acc_rf*100, acc_xgb*100, acc_svm*100],
#     'MAE': [mae_linreg, mae_rf, mae_xgb, mae_svm],
#     'RMSE': [rmse_linreg, rmse_rf, rmse_xgb, rmse_svm]})
# print(models.sort_values(by='R-squared Score', ascending=False))
# # print(models.sort_values(by='MAE', ascending=True))
# # print(models.sort_values(by='RMSE', ascending=True))

sys.stdout = __console