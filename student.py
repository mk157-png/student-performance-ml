import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

# Load dataset
csv_file = "StudentPerformance.csv"
df = pd.read_csv(csv_file)
df = df.dropna()

# Features and target
study_hours = df['Hours_Studied']
exam_scores = df['Exam_Score']

# Scatter plot + regression line
slope, intercept, r, p, std_err = stats.linregress(study_hours, exam_scores)

def myfunc(x):
    return slope * x + intercept

regression_line = list(map(myfunc, study_hours))
print(f"Correlation coefficient: {r:.2f}")

plt.scatter(study_hours, exam_scores, color='blue', label='Data')
plt.plot(study_hours, regression_line, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.legend()
plt.show()

# Prepare data for TensorFlow model
X = study_hours.values.reshape(-1, 1)
y = exam_scores.values.reshape(-1, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build TensorFlow model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=50, verbose=1)

# Predict on test set
y_pred = model.predict(X_test)

# Optional: visualize predicted vs actual
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Predicted vs Actual Exam Scores")
plt.legend()
plt.show()
