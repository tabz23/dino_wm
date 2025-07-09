from deslib.dcs.ola import OLA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# More samples to avoid warnings
X, y = make_classification(n_samples=100, n_features=20)
X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=0.3)

# Create independent classifiers
clf1 = RandomForestClassifier().fit(X_train, y_train)
clf2 = RandomForestClassifier().fit(X_train, y_train)

# Fit DESLIB model
ola = OLA(pool_classifiers=[clf1, clf2])
ola.fit(X_dsel, y_dsel)

# Now check internal KNN
print("KNN algorithm used:", ola.knn_.algorithm)
