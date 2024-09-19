import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("Automobile_data.csv")

# Display basic information
print(data.info())
print(data.describe())

# Handling missing values
data = data.replace("?", pd.NA)  # Replace '?' with NA
data.dropna(inplace=True)  # Drop rows with missing values

# Feature and target variables
X = data.drop(['price','make'], axis=1)  # Features
y = data['price']  # Target variable

# Preprocessing for numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
data.replace({'fuel-type':{'gas':0,"diesel":1}},inplace=True)

# Preprocessing for categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import matplotlib.pyplot as plt
import seaborn as sns

# Summary statistics
print(data.describe())

# Visualizations
sns.histplot(data['price'], bins=30, kde=True)
plt.title('Distribution of Car Prices')
plt.show()

# Correlation analysis
correlation_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                         ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R-squared: {r2_score(y_test, y_pred):.2f}')
