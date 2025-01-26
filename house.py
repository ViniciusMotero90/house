import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

pd.set_option('display.max_columns',21)
data = pd.read_csv('kc_house_data.csv')

data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)
data.drop('zipcode',axis=1,inplace=True)
data.drop('lat',axis=1,inplace=True)
data.drop('long',axis=1,inplace=True)

modelo = LinearRegression()

y = data['price']
x = data.drop('price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=14)

modelo.fit(x_train,y_train)
resultado = modelo.score(x_test,y_test)
print(resultado)

modeloRidge = Ridge(alpha=10)
modeloRidge.fit(x_train,y_train)
resultadoRidge = modeloRidge.score(x_test,y_test)
print(resultadoRidge)

modeloLasso = Lasso(alpha=1000,tol=0.1,max_iter=1000)
modeloLasso.fit(x_train,y_train)
resultadoLasso = modeloLasso.score(x_test,y_test)
print(resultadoLasso)


modeloElasticNet = ElasticNet(alpha=1,tol=0.2,l1_ratio=0.9,max_iter=5000)
modeloElasticNet.fit(x_train,y_train)
modeloElasticNet = modeloElasticNet.score(x_test,y_test)
print(modeloElasticNet)