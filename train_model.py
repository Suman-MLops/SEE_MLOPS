from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
x,y= load_iris(return_x_y=True)
model=LogisticRegression(max_iter=200).fit(x,y)
print("model_trained")