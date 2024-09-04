# prodigy-task_1-linear-reggression-
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
Importing Necessary Libraries:
pandas: Used for data manipulation and analysis, especially for handling tabular data.
numpy: A library for numerical computations.
LinearRegression: The linear regression model from scikit-learn.
mean_squared_error, r2_score: Functions to evaluate the performance of the model.
matplotlib.pyplot: A library for plotting data.
2. Loading the Data:
train_data = pd.read_csv('train.csv') loads the training data from a CSV file into a pandas DataFrame.
test_data = pd.read_csv('test.csv') loads the test data similarly.
3. Exploring the Data:
train_data.head() and test_data.head() display the first few rows of the training and test datasets.
train_data.shape and test_data.shape show the dimensions of the datasets.
train_data.info() and test_data.info() provide information about the data types and missing values in the datasets.
train_data.isnull() and test_data.isnull() check for missing values.
train_data.describe() and test_data.describe() provide summary statistics for the datasets.
4. Selecting Features and Target Variable:
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]: Selects the features GrLivArea, BedroomAbvGr, and FullBath from the training dataset.
y_train = train_data['SalePrice']: Sets the target variable (house prices) for the training dataset.
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]: Selects the same features from the test dataset.
5. Handling Missing Values:
X_train = X_train.fillna(X_train.mean()): Fills any missing values in the training features with the mean value of the respective column.
X_test = X_test.fillna(X_test.mean()): Does the same for the test features.
6. Training the Model:
model = LinearRegression(): Initializes the linear regression model.
model.fit(X_train, y_train): Trains the model on the training data.
7. Evaluating the Model:
y_train_pred = model.predict(X_train): Predicts the house prices for the training data.
mse_train = mean_squared_error(y_train, y_train_pred): Calculates the mean squared error (MSE) for the training predictions.
r2_train = r2_score(y_train, y_train_pred): Calculates the R-squared score, which measures the proportion of variance in the target variable that is explained by the model.
The MSE and R-squared values are printed to assess the performance of the model on the training data.

8. Making Predictions on Test Data:
y_test_pred = model.predict(X_test): Predicts the house prices for the test data using the trained model.
9. Visualizing the Results:
plt.scatter(y_train, y_train_pred, label='Predicted vs Actual Prices (Training)', alpha=0.6): Creates a scatter plot comparing the actual house prices (y_train) with the predicted prices (y_train_pred) for the training data.
plt.title('Actual vs Predicted House Prices for Training Data'): Adds a title to the plot.
plt.xlabel('Actual Prices') and plt.ylabel('Predicted Prices'): Label the axes.
plt.legend() and plt.show(): Display the legend and the plot.
Summary:
The code trains a linear regression model on a dataset to predict house prices based on living area, number of bedrooms, and number of bathrooms. It evaluates the model’s performance on the training data using MSE and R-squared metrics, and visualizes the relationship between actual and predicted prices. Finally, it applies the model to a test dataset to predict house prices.
