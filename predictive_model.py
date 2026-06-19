import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ===== SAMPLE DATA CREATE PANNROM =====
# (Dataset illa na, itha use pannunga)

np.random.seed(42)
months = np.arange(1, 49)  # 48 months data
sales = 200 + (months * 15) + np.random.randint(-30, 30, 48)

df = pd.DataFrame({'Month': months, 'Sales': sales})

print("Data Ready!")
print(df.head(10))

# ===== CLEAN PANNROM =====
df.dropna(inplace=True)
print("\nMissing values:", df.isnull().sum().sum())

# ===== FEATURES =====
X = df[['Month']]
y = df['Sales']

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MODEL TRAIN =====
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Trained!")

# ===== PREDICT =====
predictions = model.predict(X_test)

# ===== ACCURACY =====
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"\nR2 Score: {r2:.2f}")
print(f"MSE: {mse:.2f}")

if r2 >= 0.8:
    print("Excellent Model!")
else:
    print("Need Improvement")

# ===== GRAPH =====
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', color='blue', linewidth=2)
plt.plot(predictions, label='Predicted', color='red',
         linewidth=2, linestyle='--')
plt.title('Predictive Analytics - Actual vs Predicted')
plt.xlabel('Data Points')
plt.ylabel('Sales Value')
plt.legend()
plt.grid(True)
plt.savefig('prediction_graph.png')
plt.show()
print("\nGraph saved!")
