import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# virgülden sonra iki basamak göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
# sabit (b - bias)
X = pd.DataFrame([[5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1]])
y = pd.DataFrame([[600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]])
b = 275
w = 90

# maaş tahmini
y_ = b + w * X

# hata
y - y_

# hata kareleri
(y - y_) ** 2

# mutlak hata
abs(y - y_)

# m = y.shape[1]


mean_squared_error(y, y_)
# 4438.333333333333

# RMSE
np.sqrt(mean_squared_error(y, y_))
# 66.62081756728398

# MAE
mean_absolute_error(y, y_)
# 54.333333333333336






