import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

true_values = [
    'db', 'sm', 'bk', 'db', 'db', 'db', 'bk', 'md', 'bo', 'bk',
    'bk', 'md', 'bk', 'sm', 'sm', 'md', 'db', 'bo', 'md', 'md',
    'sm', 'sm', 'sm', 'sm', 'bo', 'bk', 'bk', 'md', 'md', 'db',
    'md', 'md', 'db', 'bo', 'db', 'db', 'db', 'db', 'bk', 'bo',
    'sm', 'bo', 'md', 'bo', 'bo', 'bo', 'bo', 'bk', 'sm', 'sm',
    'sm', 'bo', 'bo', 'bo', 'sm', 'sm', 'sm', 'bk', 'sm', 'md',
    'bo', 'sm', 'bo', 'bo', 'sm', 'sm', 'md', 'md', 'md', 'md',
    'bk', 'bo', 'md', 'sm', 'bk', 'bk', 'md', 'md', 'md', 'db',
    'db', 'db', 'db', 'db', 'db', 'md', 'bk', 'db', 'bo', 'db',
    'md', 'md', 'db', 'bk', 'bo', 'bo', 'db', 'db', 'db', 'sm',
    'sm', 'md', 'sm', 'md', 'sm', 'bk', 'sm', 'bo', 'bk', 'bo',
    'bk', 'md', 'bo', 'md', 'db', 'db', 'bo', 'bo', 'bk', 'sm',
    'bo', 'bk', 'db', 'bk', 'sm', 'bk', 'db', 'bo', 'bk', 'md',
    'sm'
]

predicted_values =[
    'db', 'sm', 'bk', 'db', 'db', 'db', 'bk', 'md', 'bo', 'bk', 
    'bk', 'md', 'bk', 'sm', 'sm', 'md', 'db', 'bo', 'md', 'md', 
    'sm', 'sm', 'sm', 'sm', 'bo', 'bk', 'bk', 'md', 'md', 'db', 
    'md', 'md', 'db', 'bo', 'db', 'db', 'db', 'db', 'bk', 'bo', 
    'sm', 'bo', 'md', 'bo', 'bo', 'bo', 'bo', 'bk', 'sm', 'sm', 
    'sm', 'bo', 'bo', 'bo', 'sm', 'sm', 'sm', 'bk', 'sm', 'md', 
    'bo', 'sm', 'bo', 'bo', 'sm', 'sm', 'md', 'md', 'md', 'md', 
    'bk', 'bo', 'md', 'sm', 'bk', 'bk', 'md', 'md', 'md', 'db', 
    'db', 'db', 'db', 'db', 'db', 'md', 'bk', 'db', 'bo', 'db', 
    'md', 'md', 'db', 'bk', 'md', 'db', 'db', 'bo', 'db', 'sm', 
    'sm', 'md', 'sm', 'sm', 'sm', 'sm', 'sm', 'bo', 'bk', 'bo', 
    'sm', 'db', 'bo', 'md', 'db', 'db', 'db', 'bo', 'bk', 'sm', 
    'sm', 'bk', 'db', 'bk', 'db', 'db', 'bk', 'md', 'db', 'bo', 
    'sm'
]

cm = confusion_matrix(true_values, predicted_values, labels=classes)

PA = np.diag(cm) / np.sum(cm, axis=0)

UA = np.diag(cm) / np.sum(cm, axis=1)

OA = np.sum(np.diag(cm)) / np.sum(cm)

kappa = cohen_kappa_score(true_values, predicted_values)

print("Producer's Accuracy:", PA)
print("User's Accuracy:", UA)
print("Overall Accuracy:", OA)
print("Kappa Index:", kappa)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
