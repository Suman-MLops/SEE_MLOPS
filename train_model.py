from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
iris= load_iris()
x=iris.data
y=iris.target
model=LogisticRegression(max_iter=200).fit(x,y)
print("model_trained")