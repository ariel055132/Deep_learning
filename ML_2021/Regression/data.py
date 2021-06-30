import pandas as pd
import pprint as pp

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

tr_path = 'covid.train.csv' # training dataset file path
tt_path = 'covid.test.csv'  # testing dataset file path

data_tr = pd.read_csv(tr_path)
data_tt = pd.read_csv(tt_path)

#print(data_tr.head())  # Return the first n(5) rows.
#print(data_tr.columns) # Return The column labels of the DataFrame.
data_tr.drop(['id'], axis=1, inplace=True) # Drop specified labels from rows or columns.
cols = list(data_tr.columns)

#pp.pprint(data_tr.columns)
#pp.pprint(data_tr.info()) # see the data type and size of each column

WI_index = cols.index('WI')
#print(WI_index)

#print(data_tt.iloc[:, 40:].describe()) #查看测试集数据分布

# check whether cli and ili (Covid-like illness) affect tested positive
# I do not know cli stands for what.
plt.scatter(data_tr.loc[:, 'cli'], data_tr.loc[:, 'tested_positive.2'])
plt.title("cli")
plt.xlabel("cli")
plt.ylabel("tested positive")
plt.show()

# ili stands for influenza-like illness
plt.scatter(data_tr.loc[:, 'ili'], data_tr.loc[:, 'tested_positive.2'])
plt.title("ili")
plt.xlabel("ili")
plt.ylabel("tested positive")
plt.show()

plt.scatter(data_tr.loc[:, 'ili'], data_tr.loc[:, 'cli'])
plt.title("ili vs cli")
plt.xlabel("ili")
plt.ylabel("cli")
plt.show()

# relationship between tested positive in day 1 and day 3
# linear relationship
plt.scatter(data_tr.loc[:, 'tested_positive'], data_tr.loc[:, 'tested_positive.2'])
plt.title("day1 and day3")
plt.xlabel("day1")
plt.ylabel("day3")
plt.show()

plt.scatter(data_tr.loc[:, 'tested_positive.1'], data_tr.loc[:, 'tested_positive.2'])
plt.title("day2 and day3")
plt.xlabel("day2")
plt.ylabel("day3")
plt.show()

# 还是利用corr方法自动分析
data_corr = data_tr.iloc[:, 40:].corr()
target_col = data_corr['tested_positive.2']
#print(target_col)

# 相关性数据中选择大于0.8的行，这个0.8是自己设的超参，大家可以根据实际情况调节
feature = target_col[target_col > 0.8]
print(feature)

# add target_col > 0.8 into the list
feature_cols = feature.index.tolist()
feature_cols.pop() # pop out test_positive
pp.pprint(feature_cols)

feats_selected = [cols.index(col) for col in feature_cols]  #获取该特征对应列索引编号，后续就可以用feats + feats_selected作为特征值
print(feats_selected)