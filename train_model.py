import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and train model
data = pd.read_csv("Crop_recommendation.csv")
X = data.drop("label", axis=1)
y = data["label"]

rf = RandomForestClassifier()
rf.fit(X, y)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)
