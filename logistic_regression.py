import pandas as pda
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

# 1.读取数据
fname="A:/Python_16/luqu.csv"
dataf=pda.read_csv(fname)

# 2.将特征（影响因素）和结果变成矩阵的形式。
x=dataf.iloc[:,1:4].as_matrix()
y=dataf.iloc[:,0:1].as_matrix()

# 3.导入模块sklearn.linear_model 下RandomizedLogisticRegression，进行实例化。
r1=RLR()

# 4.通过fit()进行训练模型
r1.fit(x, y)

# ***5.通过get_support()筛选有效特征，也是降维的过程，"rank"属性被去除
r1.get_support()
t=dataf[dataf.columns[r1.get_support(indices=True)]].as_matrix()

# 6.导入模块sklearn.linear_model 下LogisticRegression，进行实例化。
r2=LR()

# 7.通过fit()训练简化后的模型
r2.fit(t, y)

# 8.输出LR模型正确率
print("训练结束")
print("模型正确率为:"+str(r2.score(x, y)))








