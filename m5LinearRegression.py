'''
実際はjupyterhubで動かす。
'''
​
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
​
#sklearn.linear_model.LinearRegressionクラスを読み込み
from sklearn import linear_model
​
​
​
sns.set()
​
#ファイルの読み込み。補正なしの重さ(raw_data)と実際の重さ(weight)が設定されている
m5 = pd.read_csv("m5WeightData.csv", sep=",")
​
#ファイルの中身の表示
m5.head
​
​
clf = linear_model.LinearRegression()
​
#説明変数に"raw_data(補正なしの値)"を利用
X = m5.loc[:, ['raw_data']].values
​
#目的変数に"weight(実際にのっけた重さ)"を利用
Y = m5['weight'].values
​
plt.plot(X,Y)
​
#予測モデルを作成
clf.fit(X,Y)
​
#回帰係数
print("回帰係数",clf.coef_)
​
#切片(誤差)
print("切片(誤差)",clf.intercept_)
​
#決定関数
print("決定関数",clf.score(X,Y))
​
print("回帰式:[weight]=",clf.coef_," * [raw_data] + ",clf.intercept_)
