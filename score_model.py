import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Load the trained model
model_path = './models/geyser_model.pkl'
pipeline = joblib.load(model_path)

# Load the dataset
df = pd.read_csv('./data/faithful.csv')

# Score the dataset
predictions = pipeline.predict(df[['eruptions']])

# Add predictions as new column
df['predicted_waiting'] = predictions

# Create output directory
os.makedirs('./data/scored', exist_ok=True)

# Export scored dataset (3 columns: eruptions, waiting, predicted_waiting)
df.to_csv('./data/scored/scored_geyser.csv', index=False)

print(f"✓ Scored dataset saved to ./data/scored/scored_geyser.csv")
print(f"✓ Shape: {df.shape}")
print(f"✓ Columns: {list(df.columns)}")

# Create plot: Actual vs Predicted waiting times
plt.figure(figsize=(10, 6))
plt.scatter(df['waiting'], df['predicted_waiting'], alpha=0.5, label='Predictions')
plt.plot([df['waiting'].min(), df['waiting'].max()], 
         [df['waiting'].min(), df['waiting'].max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Waiting Time')
plt.ylabel('Predicted Waiting Time')
plt.title('Actual vs Predicted Waiting Times')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
os.makedirs('./plots', exist_ok=True)
plt.savefig('./plots/geyser_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to ./plots/geyser_predictions.png")
