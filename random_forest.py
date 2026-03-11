import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load the dataset we built with build_dataset.py
print("Loading dataset...")
df = pd.read_csv("final_ml_dataset.csv")
print(f"Loaded {len(df)} total pockets.")

# 2. Separate the data
# The "answer key" to predict
y = df['is_binding_site']

# The features the model has to study
# Drop the IDs and the answer key itself (the ones the model doesn't have to study)
X = df.drop(columns=['protein_id', 'pocket_id', 'is_binding_site'])

# Keep the IDs handy in a separate variable to use them later for ranking
identifiers = df[['protein_id', 'pocket_id']]

# 3. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, identifiers, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} pockets...")
print(f"Testing on {len(X_test)} pockets...")

# 4. Create the model
rf_model = RandomForestClassifier(
    n_estimators=100,      # Build 100 decision trees
    class_weight="balanced", # Help the model handle the imbalanced true/false pockets
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# 5. Make predictions on the test set
# Predict the probability of being a binding site (class 1)
probabilities = rf_model.predict_proba(X_test)[:, 1] 

# Get the strict 0 or 1 predictions just for the performance report
strict_predictions = rf_model.predict(X_test)

# Print a standard ML performance report
print("\n--- Model Performance ---")
print(classification_report(y_test, strict_predictions))

# 6. Rank the binding sites
# Create a new DataFrame with the test IDs
results_df = ids_test.copy()
results_df['True_Label'] = y_test
results_df['Predicted_Probability'] = probabilities

# Sort them from highest probability to lowest
ranked_sites = results_df.sort_values(by='Predicted_Probability', ascending=False)

print("\n--- Top 5 Most Likely Binding Sites in Test Set ---")
print(ranked_sites.head(5))