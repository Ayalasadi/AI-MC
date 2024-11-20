import joblib
from sklearn.dummy import DummyClassifier
import numpy as np

#create a dummy classifier to simulate emotion prediction
model = DummyClassifier(strategy='most_frequent')
model.fit(np.random.rand(10, 3), ['calm', 'excited', 'sad', 'calm', 'excited', 'sad', 'calm', 'excited', 'sad', 'calm'])

#save the model
joblib.dump(model, 'emotion_classification_model.pkl')

print("Dummy model has been generated and saved as 'emotion_classification_model.pkl'")
