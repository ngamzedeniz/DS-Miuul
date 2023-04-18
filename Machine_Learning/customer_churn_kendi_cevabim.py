import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
df = pd.read_csv("Telco-Customer-Churn.csv")

df.head()
df.shape
df["Churn"].value_counts()

#Check the datatypes and columns that are part of the data set
df.info()




df.TotalCharges.values

# infoya bakınca total charge değişkeninin O olduğunu görünce dtypeının deiştirilmesi laızm
pd.to_numeric(df.TotalCharges)

# ValueError: Unable to parse string " " hatası verdiği için içinde boşluk olan değerleri uçur


# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# total charge değişkeninin boşluk olduğu değerler
df[pd.to_numeric(df.TotalCharges, errors = "coerce").isnull()]


# total charge ın olmadığı 11 değişken var
pd.to_numeric(df.TotalCharges, errors = "coerce").isnull().sum()

# total charge da boşluk olan satır
df.iloc[753]['TotalCharges']

df.loc[df["TotalCharges"] != ' ']

# total charge ı boşluk olanları çıkardım
df = df.loc[df["TotalCharges"] != ' ']

# sonra total charge ın tipini değiştirdim
df['TotalCharges'] = df['TotalCharges'].astype(float)

# bir de customer id değişkenini işime yaramayacağı için siliyorum
df.drop("customerID", axis = 'columns', inplace = True)


# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz

cat_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() < 5]
cat_but_num = [col for col in df.columns if df[col].dtypes != "O" and df[col].nunique() < 5]
cat_cols += cat_but_num
num_cols = [col for col in df.columns if df[col].dtypes != "O"]


cat_cols

# cat değişkenlerin ne old hatırlamak için
# ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', /
# 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', /
# 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn', 'SeniorCitizen']

df["gender"].value_counts()


# kategorik dğeişken: sutün grafik, countplot bar
# num değişkenler: hist, boxplot

# kategorik değişkenler için
for col in cat_cols:
    print(df[col].value_counts())

## kategorik değişkenlerin yüzde ve dağılımlarının fonksiyonlaştırılması
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

for col in cat_cols:
    cat_summary(df, col)

## grafikleştirme için
import seaborn as sns
import matplotlib.pyplot as plt

for col in cat_cols:
    df[col].value_counts().plot(kind="bar")
    plt.show(block=True)

## cat değişkenlerin grafikleştirilmesinin fonksiyonlaştırılması

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)

# sayısal değişkenler için
# num değişkenler: hist, boxplot
for col in num_cols:
    plt.hist(df[col])
    plt.show(block=True)

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def target_sum_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"xx": dataframe.groupby(categorical_col)[target].count()}))

target_sum_with_cat(df, "Churn", 'tenure')



cat_cols

# Adım 5: Aykırı gözlem var mı inceleyiniz.


df["TotalCharges"].describe().T

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

num_cols
# ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
outlier_thresholds(df,"SeniorCitizen")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


num_cols
# ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

check_outlier(df, "SeniorCitizen")
check_outlier(df, "tenure")
check_outlier(df, "MonthlyCharges")
check_outlier(df, "TotalCharges")

# hepsi false döndü

# 6: Eksik gözlem var mı inceleyiniz.

#############################################
# Eksik Değerlerin Yakalanması
#############################################


df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)


# Adım 2: Yeni değişkenler oluşturunuz.

def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column} : {df[column].unique()}')

print_unique_col_values(df)

# gender : ['Female' 'Male']
# Partner : ['Yes' 'No']
# Dependents : ['No' 'Yes']
# PhoneService : ['No' 'Yes']
# MultipleLines : ['No phone service' 'No' 'Yes']
# InternetService : ['DSL' 'Fiber optic' 'No']
# OnlineSecurity : ['No' 'Yes' 'No internet service']
# OnlineBackup : ['Yes' 'No' 'No internet service']
# DeviceProtection : ['No' 'Yes' 'No internet service']
# TechSupport : ['No' 'Yes' 'No internet service']
# StreamingTV : ['No' 'Yes' 'No internet service']
# StreamingMovies : ['No' 'Yes' 'No internet service']
# Contract : ['Month-to-month' 'One year' 'Two year']
# PaperlessBilling : ['Yes' 'No']
# PaymentMethod : ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
#  'Credit card (automatic)']
# Churn : ['No' 'Yes']

df.replace('No phone service', 'No', inplace = True)
df.replace('No internet service', 'No', inplace = True)

print_unique_col_values(df)

# gender : ['Female' 'Male']
# Partner : ['Yes' 'No']
# Dependents : ['No' 'Yes']
# PhoneService : ['No' 'Yes']
# MultipleLines : ['No' 'Yes']
# InternetService : ['DSL' 'Fiber optic' 'No']
# OnlineSecurity : ['No' 'Yes']
# OnlineBackup : ['Yes' 'No']
# DeviceProtection : ['No' 'Yes']
# TechSupport : ['No' 'Yes']
# StreamingTV : ['No' 'Yes']
# StreamingMovies : ['No' 'Yes']
# Contract : ['Month-to-month' 'One year' 'Two year']
# PaperlessBilling : ['Yes' 'No']
# PaymentMethod : ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
#  'Credit card (automatic)']
# TotalCharges : ['29.85' '1889.5' '108.15' ... '346.45' '306.6' '6844.5']
# Churn : ['No' 'Yes']

yes_no_columns = ["Partner","Dependents","PhoneService", "MultipleLines","OnlineSecurity", "OnlineBackup", \
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling","Churn"]

for col in yes_no_columns:
    df[col].replace({"Yes": 1, "No": 0}, inplace = True)

df.head()

for col in df:
    print(f'{col} : {df[col].unique()}')

# gender : ['Female' 'Male']
# SeniorCitizen : [0 1]
# Partner : [1 0]
# Dependents : [0 1]
# tenure : [ 1 34  2 45  8 22 10 28 62 13 16 58 49 25 69 52 71 21 12 30 47 72 17 27
#   5 46 11 70 63 43 15 60 18 66  9  3 31 50 64 56  7 42 35 48 29 65 38 68
#  32 55 37 36 41  6  4 33 67 23 57 61 14 20 53 40 59 24 44 19 54 51 26  0
#  39]
# PhoneService : [0 1]
# MultipleLines : [0 1]
# InternetService : ['DSL' 'Fiber optic' 'No']
# OnlineSecurity : [0 1]
# OnlineBackup : [1 0]
# DeviceProtection : [0 1]
# TechSupport : [0 1]
# StreamingTV : [0 1]
# StreamingMovies : [0 1]
# Contract : ['Month-to-month' 'One year' 'Two year']
# PaperlessBilling : [1 0]
# PaymentMethod : ['Electronic check' 'Mailed check' 'Bank transfer (automatic)'
#  'Credit card (automatic)']
# MonthlyCharges : [29.85 56.95 53.85 ... 63.1  44.2  78.7 ]
# TotalCharges : ['29.85' '1889.5' '108.15' ... '346.45' '306.6' '6844.5']
# Churn : [0 1]


df['gender'].replace({'Female' : 1, 'Male' : 0}, inplace = True)

df_d = pd.get_dummies(data = df, columns = ['InternetService', 'Contract', 'PaymentMethod'], dtype=float, drop_first=True)
df = pd.get_dummies(data = df, columns = ['InternetService', 'Contract', 'PaymentMethod'], dtype=float)
df_d.head()

# tenure, MonthlyCharges, TotalCharges eğişkenleri one-hot encode edilmedi, kategorik olmadıkları için
# bu değişkenleri scale edicez

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_d[cols_to_scale]= scaler.fit_transform(df_d[cols_to_scale])

for col in df_d:
    print(f'{col}: {df_d[col].unique()}')

# GÖREV 3: MODELLEME

# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracyskorlarını inceleyip. En iyi 4 modeli seçiniz.

y = df_d["Churn"]
X = df_d.drop(["Churn"], axis=1)

#####################################
# KNN
#####################################

knn_model = KNeighborsClassifier().fit(X, y)

# Confusion Matrix için y_pred
y_pred = knn_model.predict(X)

# AUC için y_prob
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))

#               precision    recall  f1-score   support
#            0       0.87      0.91      0.89      5163
#            1       0.72      0.63      0.67      1869
#     accuracy                           0.84      7032
#    macro avg       0.80      0.77      0.78      7032
# weighted avg       0.83      0.84      0.83      7032


# AUC
roc_auc_score(y, y_pred)
# 0.77

cv_results = cross_validate(knn_model, X, y, cv=5, scoring= ["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.76
cv_results["test_f1"].mean()
# 0.54
cv_results["test_roc_auc"].mean()
# 0.77

###################################
# CART
###################################
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# train hatalarında modelin sonucu hep 1, recall, precision vs. 1

# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))

#   precision    recall  f1-score   support
#            0       0.80      0.80      0.80      1552
#            1       0.44      0.45      0.45       558
#     accuracy                           0.71      2110
#    macro avg       0.62      0.62      0.62      2110
# weighted avg       0.71      0.71      0.71      2110

roc_auc_score(y_test, y_prob)
# 0.62

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 30)

import warnings
warnings.simplefilter(action='ignore', category=Warning)


################################################
# Random Forests
################################################

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.79
cv_results['test_f1'].mean()
# 0.55
cv_results['test_roc_auc'].mean()
# 0.82

################################################
# GBM
################################################

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.80
cv_results['test_f1'].mean()
# 0.58
cv_results['test_roc_auc'].mean()
# 0.845

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.785
cv_results['test_f1'].mean()
# 0.56
cv_results['test_roc_auc'].mean()
# 0.823




# Adım 2: Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve bulduğunuz hiparparametrelerile modeli tekrar kurunuz.

# hiperparametre optimizasyonu KNN
knn_model.get_params()
# n_neighbors: 5 tanımlı

knn_params = {"n_neighbors": range(1,40)}

# n_jobs = -1 işlemciyi en hızlı şekilde kullan
# verbose=1 işlem sonucunda rapor istiyor musun? evet
## gridsearch hiperparametler birden fazlaysa bütün kombinasyonlarına bakar
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_
# {'n_neighbors': 34}

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.79
cv_results['test_f1'].mean()
# 0.58
cv_results['test_roc_auc'].mean()
# 0.83

# RF
rf_model.get_params()

# hiperpametre optimizasyonu RANDOM FOREST

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "sqrt"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.80
cv_results['test_f1'].mean()
# 0.57
cv_results['test_roc_auc'].mean()
# 0.84

# hiperparametre optimiasyonu GBM

gbm_model.get_params()
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
# {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 1}

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# önceki değerler, 80,58,84
cv_results['test_accuracy'].mean()
# 0.80
cv_results['test_f1'].mean()
# 0.59
cv_results['test_roc_auc'].mean()
# 0.845

# hp opt XGB

xgboost_model.get_params()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# önceki değerler 78, 56, 82
cv_results['test_accuracy'].mean()
# 0.80
cv_results['test_f1'].mean()
# 0.58
cv_results['test_roc_auc'].mean()
# 0.84

# GBM en iyi model, 80, 59, 8425

# hızlı old iiçin lightgbm i de deneyelim

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.79
cv_results['test_f1'].mean()
# 0.57
cv_results['test_roc_auc'].mean()
# 0.83
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.80
cv_results['test_f1'].mean()
# 0.58
cv_results['test_roc_auc'].mean()
# 0.844
# Hiperparametre yeni değerlerle
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.80
cv_results['test_f1'].mean()
# 0.585
cv_results['test_roc_auc'].mean()
# 0.843

# SONUÇ: GBM, KNN, CART, RF, XGB VE LIGHTGBM ARASINDAN EN İYİ MODEL GBM, AMA YAVAŞ, EN İYİ İKİNCİ LIGHTGBM VE GBM'E ÇOK YAKIN VE HIZLI

