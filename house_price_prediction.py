import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet      #works with combination of ridge and lasso
from sklearn.metrics import(r2_score,mean_absolute_error,mean_squared_error)

df=pd.read_csv("data/train.csv")

#filing nan missing values with none 
no_feature=[
    "PoolQC",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "MasVnrType"
]
for col in no_feature:
    df[col]=df[col].fillna("none")

#filling the numerical missing values

df["LotFrontage"]=df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"]=df["MasVnrArea"].fillna(0)
df["GarageYrBlt"]=df["GarageYrBlt"].fillna(0)

#one hot encoding 
df=pd.get_dummies(df,drop_first=True,dtype=int)

#now splting training and scaling
x=df.drop(["SalePrice","Id"],axis=1)
y=df["SalePrice"]
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=42,
    test_size=0.2
)

#scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#now implementing elastic net as it uses both ridge and lasso by tuning and eliminating the coefficient
enet=ElasticNet(random_state=42,max_iter=10000)
param_grid={
    "alpha":[0.001,0.01,0.1,1,10,100],
    "l1_ratio": [0.2, 0.4, 0.6, 0.8, 1.0]
}
grid=GridSearchCV(
    estimator=enet,
    param_grid=param_grid,
    cv=5,
    scoring="r2"
)
grid.fit(x_train_scaled,y_train)
best_enet=grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

train_pred = best_enet.predict(x_train_scaled)
test_pred = best_enet.predict(x_test_scaled)

print("Train R²:", r2_score(y_train, train_pred))
print("Test R²:", r2_score(y_test, test_pred))

#checking number of removed features
coef = pd.Series(best_enet.coef_, index=x.columns)
print("Features removed:", (coef == 0).sum())