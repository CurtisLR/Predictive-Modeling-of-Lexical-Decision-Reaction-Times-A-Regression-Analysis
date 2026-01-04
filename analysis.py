import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# For a cleaner output
import warnings
warnings.filterwarnings('ignore')


# Helper Functions
def calculate_bic(X, y, degree, model):
    """Calculates Bayesian Information Criterion (BIC) for model selection."""
    n = X.shape[0]
    k = degree + 1
    RSS = mean_squared_error(y, model.predict(X)) * n
    return n * np.log(RSS / n) + k * np.log(n)


# Data Preprocessing
# Assuming data is in a 'data' subdirectory
english = pd.read_csv('data/english_a4.csv')


# Filter for young subjects and relevant columns
print("Preprocessing: Filtering for young subjects and encoding categorical variables...")
english_young = english[english['AgeSubject'] == 'young'].copy()
english_young = english_young.drop(columns=['AgeSubject', 'RTnaming'])

# Map categorical predictors to numeric
category_map = {'N': 0, 'V': 1, 'voiced': 0, 'voiceless': 1}
english_young['WordCategory'] = english_young['WordCategory'].map(category_map)
english_young['Voice'] = english_young['Voice'].map(category_map)

# Select final features
features = ['RTlexdec', 'Word', 'Familiarity', 'WordCategory', 'WrittenFrequency',
            'WrittenSpokenFrequencyRatio', 'FamilySize', 'InflectionalEntropy',
            'LengthInLetters', 'Voice']
english_young = english_young[features]

print(f"Data ready: {english_young.shape[0]} rows, {english_young.shape[1]} columns.")

# Exploratory Visualization
print("Generating Exploratory Plots...")
sns.set(style="whitegrid")

# Comparison: Linear Fit vs LOWESS
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: Linear Trend
sns.regplot(x='WrittenFrequency', y='RTlexdec', data=english_young, ax=axes[0],
            scatter_kws={'s': 18, 'alpha': 0.6, 'color': 'blue'},
            line_kws={'color': 'red', 'lw': 3})
axes[0].set_title('Linear Fit: Frequency vs. RT', fontsize=14)

# Plot B: LOWESS
sns.regplot(x='WrittenFrequency', y='RTlexdec', data=english_young, ax=axes[1],
            scatter_kws={'s': 18, 'alpha': 0.6, 'color': 'green'},
            line_kws={'color': 'orange', 'lw': 3}, lowess=True)
axes[1].set_title('LOWESS Fit: Frequency vs. RT', fontsize=14)

plt.tight_layout()
plt.show()
print(
    "Observation: The LOWESS curve suggests a non-linear relationship. Reaction time levels off as frequency increases.")

# Polynomial Regression Analysis
print("\n--- Model Selection: Polynomial Regression ---")

X = english_young[['WrittenFrequency']]
y = english_young['RTlexdec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualization setup
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.1, label='Training Data')
colors = ['red', 'black', 'limegreen', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'gold']

# Loop through degrees to find best fit
for i, degree in enumerate([1, 2, 3, 4, 5, 6, 7, 10, 25]):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    model_bic = calculate_bic(X_test, y_test, degree, model)

    if degree <= 10:  # Only print reasonable degrees
        print(f"Degree {degree:2d} | Train R^2: {train_r2:.4f} | Test R^2: {test_r2:.4f} | BIC: {model_bic:.2f}")

    # Plotting specific lines for visual comparison
    if degree in [1, 2, 3, 4, 10, 25]:
        X_test_sorted = np.sort(X_test, axis=0)
        y_pred = model.predict(X_test_sorted)
        c_idx = i % len(colors)
        plt.plot(X_test_sorted, y_pred, label=f"Degree {degree}", color=colors[c_idx], linewidth=2)

plt.legend()
plt.xlabel("Written Frequency")
plt.ylabel("Lexical Decision RT")
plt.title("Polynomial Regression Fits")
plt.show()

# Multivariate LASSO Regression
print("\n--- Feature Selection: LASSO Regression ---")

# Prepare full feature set with polynomial expansion for Frequency
X_full = english_young[['Familiarity', 'WordCategory', 'WrittenFrequency',
                        'WrittenSpokenFrequencyRatio', 'FamilySize',
                        'InflectionalEntropy', 'LengthInLetters', 'Voice']].copy()

# Adding polynomial terms for WrittenFrequency (based on previous analysis favoring non-linear fit)
X_full['WrittenFrequency2'] = X_full['WrittenFrequency'] ** 2
X_full['WrittenFrequency3'] = X_full['WrittenFrequency'] ** 3
X_full['WrittenFrequency4'] = X_full['WrittenFrequency'] ** 4

# Standardization
scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns)
y_std = (y - y.mean()) / y.std()

# Train/Test Split
X_std_train, X_std_test, y_std_train, y_std_test = train_test_split(
    X_std, y_std, test_size=0.2, random_state=42
)

# Fit LASSO
lasso = Lasso(alpha=0.02)
lasso.fit(X_std_train, y_std_train)

print(f"LASSO Train R^2: {lasso.score(X_std_train, y_std_train):.4f}")
print(f"LASSO Test R^2:  {lasso.score(X_std_test, y_std_test):.4f}")

# Extract Coefficients
coef_df = pd.DataFrame({'Predictor': X_std.columns, 'Coefficient': lasso.coef_})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\nTop Predictors of Reaction Time:")
print(coef_df[['Predictor', 'Coefficient']].head(5))

print(
    "\nFinding: 'WrittenFrequency' and 'Familiarity' are highly correlated. LASSO handles this collinearity by arbitrarily selecting one (or dampening both) to optimize the penalized loss function.")
