# Method lifted from https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# Here we convert our Excel sheet training set into a GraphViz decision tree

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

name = 'mclaren'

col_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'Cluster']
# load dataset
dataset = pd.read_excel("output"+name+".xlsx", header=0, names=col_names)
print(dataset.head())

feature_cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
X = dataset[feature_cols] # Features
y = dataset.Cluster # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(name+'tree.png')
Image(graph.create_png())
