import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.utils import check_array


# Define the load_data function to read the data from a CSV file
def load_data(file_path):
    # Use Pandas' read_csv to read the data from the CSV file
    data = pd.read_csv(file_path)
    return data


# Assuming 'data' is your dataframe containing independent variables and 'MntWines'
# is your target variable
file_path = "D:\\DS - Projects\\Final project\\No - 2\\Marketing data.csv"
data = load_data(file_path)

X = data.drop('MntWines', axis=1)
y = data['MntWines']

# Identify categorical and numerical columns
cat_columns = [col for col in X.columns if X[col].dtype == 'object']
num_columns = [col for col in X.columns if col not in cat_columns]

# Preprocessing for numerical data
num_transformer = StandardScaler()

# Preprocessing for categorical data
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_columns),
        ('cat', cat_transformer, cat_columns)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Convert sparse matrix to dense matrix
X_dense = check_array(X_processed, accept_sparse='csr').toarray()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dense, y, test_size=0.2, random_state=42)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train models
# Multiple Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_pca, y_train)

# SVM
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_pca, y_train)

# Decision Trees
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train_pca, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_pca)
y_pred_svm = svm_model.predict(X_test_pca)
y_pred_dt = dt_model.predict(X_test_pca)

# Evaluate models
metrics_lr = {
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'R^2': r2_score(y_test, y_pred_lr),
    'Explained Variance': explained_variance_score(y_test, y_pred_lr)
}

metrics_svm = {
    'MSE': mean_squared_error(y_test, y_pred_svm),
    'MAE': mean_absolute_error(y_test, y_pred_svm),
    'R^2': r2_score(y_test, y_pred_svm),
    'Explained Variance': explained_variance_score(y_test, y_pred_svm)
}

metrics_dt = {
    'MSE': mean_squared_error(y_test, y_pred_dt),
    'MAE': mean_absolute_error(y_test, y_pred_dt),
    'R^2': r2_score(y_test, y_pred_dt),
    'Explained Variance': explained_variance_score(y_test, y_pred_dt)
}

print("Multiple Linear Regression Metrics:")
for metric, value in metrics_lr.items():
    print(metric + ": ", value)

print("\nSVM Metrics:")
for metric, value in metrics_svm.items():
    print(metric + ": ", value)

print("\nDecision Tree Metrics:")
for metric, value in metrics_dt.items():
    print(metric + ": ", value)