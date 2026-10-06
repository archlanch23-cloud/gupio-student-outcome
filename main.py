import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# Load
df = pd.read_csv("dataset.csv", sep=";")
X = df.drop("Target", axis=1)
y = df["Target"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Train
model = XGBClassifier(random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, pred, target_names=le.classes_))

# --- GRAPH FOR TEACHER ---
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title(f'Confusion Matrix - Accuracy {acc*100:.2f}%')
plt.colorbar()
plt.xticks([0,1,2], le.classes_, rotation=20)
plt.yticks([0,1,2], le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')

# Add numbers inside boxes
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("\nGraph saved as confusion_matrix.png - SUCCESS!")