# supervised_project3
The Breast Cancer dataset provides information about breast tumors, helping to classify them as cancerous or non-cancerous. This dataset is widely used in machine learning and can be easily accessed using the Scikit-learn library in Python.
code:
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(data.feature_names)
print(data.target_names)
