import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# 读取文件数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
print(train_data.shape)
print(test_data.shape)

# 查看分布
y_train = train_data.pop('SalePrice')
plt.hist(y_train, bins=20)
plt.show()

# 处理数据使其平滑
y_train = np.log1p(y_train)
plt.hist(y_train, bins=20)
plt.show()

# 合并训练集和测试集方便处理
data = pd.concat((train_data, test_data), axis=0)
print(data.shape)

# 年份相关的属性处理
data.eval('Built2Sold = YrSold-YearBuilt', inplace=True)
data.eval('Add2Sold = YrSold-YearRemodAdd', inplace=True)
data.eval('GarageBlt = YrSold-GarageYrBlt', inplace=True)
data.drop(['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

# 类别型属性数字格式的值转换成字符串
data['OverallQual'] = data['OverallQual'].astype(str)
data['OverallCond'] = data['OverallCond'].astype(str)
data['MSSubClass'] = data['MSSubClass'].astype(str)

# one-hot处理
dummied_data = pd.get_dummies(data)

# 查看numerical类型属性的缺失并填充
print(dummied_data.isnull().sum().sort_values(ascending=False).head())
mean_cols = dummied_data.mean()
dummied_data = dummied_data.fillna(mean_cols)

# 标准差标准化
numerical_cols = data.columns[data.dtypes != 'object']  # 数据为数值型的列名
num_cols_mean = dummied_data.loc[:, numerical_cols].mean()
num_cols_std = dummied_data.loc[:, numerical_cols].std()
dummied_data.loc[:, numerical_cols] = (dummied_data.loc[:, numerical_cols] - num_cols_mean) / num_cols_std
print(dummied_data.shape)

# 将处理后的数据分割成训练集和测试集
X_train = dummied_data[:1460]
X_test = dummied_data[1460:]

# 寻找最佳参数
params = [3,4,5,6,7,8]
# 深度参数列表
scores = []
# 存储分数的列表
for param in params:
    model = XGBRegressor(max_depth=param)
    score = np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    scores.append(np.mean(score))
plt.plot(params, scores)
plt.show()
# 可视化

# 利用最佳参数构造回归器并进行预测
xgbr = XGBRegressor(max_depth=4)
xgbr.fit(X_train, y_train)
y_prediction = np.expm1(xgbr.predict(X_test))

# 将预测结果按照要求格式写入csv文件
submitted_data = pd.DataFrame(data= {'Id' : test_data.index+1461, 'SalePrice': y_prediction})
submitted_data.to_csv('submission.csv', index=False)
