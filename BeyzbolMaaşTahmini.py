
# Salary Prediction with Hitters Data Set


# Dataset Story

# The Hitters is a data set which contains certain statistics and salaries of Major league baseball players for the years 1986–87.
# The dataset is part of the data used in the 1988 ASA Graphics Section Poster Session.
# Salary data originally taken from Sports Illustrated, April 20, 1987.
# 1986 and career statistics are from the 1987 Baseball Encyclopedia Update, published by Collier Books, Macmillan Publishing Company, NewYork.

# # Features
# Dataset consist of 20 variables and 322 observations, only the salary variable has missing observations. The definitions of the variables of the dataset are as follows:
#
# * AtBat: Number of shots made with a baseball bat during the 1986–1987 season
# * Hits: Number of hits made in the 1986–1987 season
# * HmRun: Most valuable hits in the 1986–1987 season
# * Runs: The points he earned for his team in the 1986–1987 season
# * RBI: Number of players a batsman had jogged when he hit in the season
# * Walks: Number of mistakes made by the opposing player
# * Years: Player’s playing time in major league (in year)
# * CAtBat: Number of shots made with a baseball bat in career
# * CHits: Number of hits made in the career
# * CHmRun: Most valuable hits in the career
# * CRuns: The points he earned for his team in his career
# * CRBI: Number of players a batsman had jogged when he hit in the career
# * CWalks: Number of mistakes made by the opposing player in career
# * League: A factor with A and N levels showing the league in which the player played until the end of the season
# * Division: A factor with levels E and W indicating the position played by the player at the end of 1986
# * PutOuts: Helping your teammate in-game
# * Assists: Number of assists made by the player in the 1986–1987 season
# * Errors: Player’s errors in the 1986–1987 season
# * Salary: The salary of the player in the 1986–1987 season (in thousand)
# * NewLeague: A factor with A and N levels showing the player’s league at the start of the 1987 season

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import math as mt
import missingno as msno
from sklearn.impute import KNNImputer

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score,  cross_validate, GridSearchCV , validation_curve, RandomizedSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import warnings
from warnings import filterwarnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
filterwarnings("ignore",category=DeprecationWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_rows', 50)

##############################################################
# Import Data
##############################################################
df_ = pd.read_csv("AllMiul/W7_8_9/W7_HW/hitters.csv")
df = df_.copy()
df.head()

def upper_col_name(dataframe):
    upper_cols = [col.upper() for col in dataframe.columns]
    dataframe.columns = upper_cols
    return dataframe

df = upper_col_name(df)
df.head()

##############################################################
# Data Understanding
##############################################################
# Genel Exploration for Dataset
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Categorical & Numerical Variables:
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and "ID" not in col]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(cat_cols)
print(num_cols)

# Categorical Variables:
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

#  Numerical Variables:

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

############## Other Visualizations ##############
# Salary distribution:
sns.distplot(df["SALARY"]);

# Salary by League & New League:
g = sns.catplot(x="LEAGUE",
                y="SALARY",
                hue="NEWLEAGUE", 
                data=df, kind="bar",
                height=4, aspect=1,palette="deep");
plt.show()

# Salary by Division: 
sns.barplot(x="DIVISION",
            y = "SALARY",
            data=df,
            hue="NEWLEAGUE",
            palette="deep");
plt.show()

##############################################################
# Data Preprocessing & Feature Engineering
##############################################################
# This part consists of 4 steps which are below:
# 1. Missing Values
# 2. Outliers
# 3. Rare Encoding, Label Encoding, One-Hot Encoding
# 4. Feature Scaling

# 1. Missing Values

def missing_values_df(dataframe, na_name=False):
    na_column = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] # missing değer iceren kolon adı
    n_miss = dataframe[na_column].isnull().sum().sort_values(ascending=False) # boş gözlem sayısı
    ratio = (dataframe[na_column].isnull().sum() * 100/ dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.DataFrame({"n_miss":n_miss, "n_miss_ratio":ratio})
    # missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_column
    
missing_values_df(df)

#         n_miss  n_miss_ratio
# SALARY      59     18.322981

# Bağımlı değişkenimizde NA görüldüğü için silme yöntemini tercih edebiliriz, ancak veri setinin yaklaşık
# %20 sinde NA bulunduğu için alternatif olarak median ile doldurmayı deneyelim
# Bunun için öncelikle SALARY değişkenini en çok açıklayan değişkenleri bulalım:

# Veri setinde SALARY değişkeni dolu olan gözlemleri ele alalım:

hitters_df = df.copy()
hitters_df.dropna(inplace=True)
hitters_df.head()
df_.head()


# One Hot encoding:
dms = pd.get_dummies(hitters_df[cat_cols], drop_first=True, dtype="int64")
df_ = hitters_df.drop(columns=cat_cols, axis=1)
hitters_df_ = pd.concat([df_, dms],axis=1)
hitters_df_.head()


# Base Model : LightGBM modeli ile salary değişkenini tahminleyelim, Feature Imp metodunu kullanarak
# açıklayıcılığı yüksek değişkenleri bulalım:

X = hitters_df_.drop(["SALARY"], axis=1)
y = hitters_df_["SALARY"]

lgb_model = LGBMRegressor().fit(X, y)
y_pred = lgb_model.predict(X)   

# r2 Score:
r2_scr= r2_score(y,y_pred)
print("R2 Score:", r2_scr)

#  5-Fold Cross-Validation Score?
print("5 Fold CV Score:", np.mean(np.sqrt(-cross_val_score(lgb_model, X, y, cv=5, scoring="neg_mean_squared_error"))))

# Feature Selection Metodu:
# lgb_model.feature_importances_


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num]) # num argümanı ile barplot üzerinde kaç değişkeni göstereceğimizi seçebiliriz.
    plt.title("Features")
    plt.title(f"Features for {type(model).__name__}") # modelin __name__ metodu ile adını döndürür
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgb_model, X, 5)


# Salary değişkenini en iyi açıklayan 5 değişken:
# CRBI - WALKS - CWALKS - PUTOUTS - YEARS

# Bu değişkenlerden kategorik değişken türeterek, türetilen yeni değişkenlere göre doldurma işlemi gerçekleştirelim:



# CRBI_Cat 
df['CRBI_CAT'] = pd.qcut(x=df['CRBI'], q=3, labels = ["low", "medium", "high"])

# Walks_Cat
df['WALKS_CAT'] = pd.qcut(x=df['WALKS'], q=3, labels = ["low", "medium", "high"])

# CWalks_Cat
df['CWALKS_CAT'] = pd.qcut(x=df['CWALKS'], q=3, labels = ["low", "medium", "high"]) 

# PutOuts_Cat
df['PUTOUTS_CAT'] = pd.qcut(x=df['PUTOUTS'], q=2, labels = ["low", "high"]) 

# Years_Cat
years_bins = [0, 3, 5, 10,  int(df["YEARS"].max())]
df["YEARS_CAT"] = pd.cut(df["YEARS"],
                         years_bins,
                         labels= ["new", "medium_experienced", "experienced", "high_experienced"])

# NA Değerler:
# df.isnull().sum()

# Yukarıdaki kategorik değişkenlere göre gruplama yaparak NaN değer içeren gözlemleri dolduralım

df["SALARY"] = df["SALARY"]. \
                fillna(df.groupby(["YEARS_CAT", "PUTOUTS_CAT", "CWALKS_CAT", "WALKS_CAT", "CRBI_CAT"])["SALARY"]. \
                transform("median"))

df.isnull().sum().sum()

# NaN değer içeren 3 kayıt kaldı, çünkü bu case'e denk gelen kayıtların tamamında zaten Salary değişkeni NaN olduğu için
# doldurma işlemi gerçekleştirilmedi, silebiliriz.

df[df[["SALARY"]].isnull().any(axis=1)][["YEARS_CAT","PUTOUTS_CAT", "CWALKS_CAT", "WALKS_CAT", "CRBI_CAT","SALARY"]]
df.dropna(inplace=True)


##############################################################
# FEATURE ENGINEERING:
##############################################################

 # Baseball oyunu 9 inning'ten (atış sırası) oluşmaktadır:
df['NEW_HITS_PER_GAME'] = df["HITS"] / 9
df['NEW_CHITS_PER_GAME'] = df["CHITS"] / 9
df["NEW_CWALKS_PER_GAME"] = df["CWALKS"] / 9
df["NEW_WALKS_PER_GAME"] = df["WALKS"] / 9
df["NEW_CHMRUN_PER_GAME"] = df["CHMRUN"] / 9
df["NEW_HMRUN_PER_GAME"] = df["HMRUN"] / 9
 df["NEW_ERA"] = df["RUNS"] /9

# Hits Rate:
df["NEW_HITS_RATE"] = (df["HITS"] / df["ATBAT"]) 

# Number of missed hits :
df["NEW_NUM_OF_MISSING"] =  (df["ATBAT"]) - ( df["HITS"] )
df["NEW_NUM_OF_MISSING_RATE"] = df["NEW_NUM_OF_MISSING"] / df["ATBAT"] 
df["NEW_HITS_NUM_OF_MISSING_RATE"] = (df["HITS"] / df["NEW_NUM_OF_MISSING"] )

# WALKS and HITS per inning pitched:
df["NEW_WHIP"] = df["WALKS"] + df["WALKS"]
 
# Hits rate missed hits and rate over career:  
df["NEW_HITS_RATE_CAREER"] = (df["CHITS"] / df["CATBAT"]) 
df["NEW_NUM_OF_MISSING_CAREER"] = df["CATBAT"] - df["CHITS"] 
df["NEW_NUM_OF_MISSING_RATE_CAREER"] = df["CHITS"] / df["CATBAT"] 
df["NEW_HITS_NUM_OF_MISSING_CAREER_RATE"] = (df["CHITS"] / df["NEW_NUM_OF_MISSING_CAREER"] )
df["NEW_NUM_OF_MISSING_RATE"] = df["HITS"] / df["ATBAT"] 

# HITS_Missing_Rate_Delta
df["NEW_HITS_MISSING_RATE_DELTA"] = (df["NEW_HITS_NUM_OF_MISSING_RATE"] / df["NEW_HITS_NUM_OF_MISSING_CAREER_RATE"] )

df["NEW_HITS/CHITS"] = (df["HITS"] / df["CHITS"]) 
df["NEW_HMRUN/ATBAT"] = (df["HMRUN"] / df["ATBAT"])
df["NEW_HMRUN/CHMRUN"] = (df["HMRUN"] / df["CHMRUN"])
df["NEW_RBI/ATBAT"] = (df["RBI"] / df["ATBAT"])
df["NEW_RUNS/ATBAT"] = (df["RUNS"] / df["ATBAT"]) 
df["NEW_CHMRUN/ATBAT"] = (df["CHMRUN"] / df["ATBAT"]) 
df["NEW_CHMRUN/CATBAT"] = (df["CHMRUN"] / df["CATBAT"] )
df["NEW_ATBAT/CATBAT"] = (df["ATBAT"] / df["CATBAT"])
df["NEW_HMRUN/HITS"] = (df["HMRUN"] / df["HITS"])
df["NEW_HITS/CHITS"] = (df["HITS"] / df["CHITS"])
df["NEW_CHITS/CRBI"] = (df["CHITS"] / df["CRBI"])
df["NEW_CHMRUN/CRUNS"] = (df["CHMRUN"] / df["CRUNS"])
df["NEW_RUNS/CRUNS"] = (df["RUNS"] / df["CRUNS"])
df['NEW_HMRUN/RUNS'] =  (df['HMRUN'] / df['RUNS']) 
df['NEW_CHITS/CATBAT'] = (df['CHITS'] / df['CATBAT'])
df["NEW_WALKS/RBI"] = (df["WALKS"] / df["RBI"])
df["NEW_WALKS/CWALKS"] = (df["WALKS"] / df["CWALKS"])
df["NEW_RBI/CRBI"] = (df["RBI"] / df["CRBI"])
df["NEW_CRBI/RBI"] = (df["CRBI"] / df["RBI"]) 
df["NEW_Total_RBI"] = df["RBI"] * df["WALKS"]
df["NEW_CRUNS/CHITS"] = (df["CRUNS"] / df["CHITS"])
df["NEW_CRUNS/CATBAT"] = df["CRUNS"]/df["CATBAT"]
df["NEW_CRBI/RBI"] = (df["CRBI"] / df["RBI"])
df["NEW_PUTOUTS/ATBAT"] = (df["PUTOUTS"] / df["ATBAT"])
df["NEW_ASSISTS/PUTOUTS"] = (df["ASSISTS"] / df["PUTOUTS"])
df["NEW_ERRORS/ASSISTS"] = (df["ERRORS"] / df["ASSISTS"] ) 
df["NEW_ERRORS/CWALKS"] = (df["ERRORS"] / df["CWALKS"] ) 
df["NEW_ASSISTS/ERRORS"] = (df["ASSISTS"] / df["ERRORS"] )
df["NEW_CATBAT_PER_YEAR"] = (df["CATBAT"] / df["YEARS"] )
df["NEW_CHITS_PER_YEAR"] = (df["CHITS"] / df["YEARS"] )  
df["NEW_CRUNS_PER_YEAR"] = (df["CRUNS"] / df["YEARS"] )  
df["NEW_CHMRUN_PER_YEAR"] = (df["CHMRUN"] / df["YEARS"] ) 
df["NEW_WALKS_PER_YEAR"] = (df["WALKS"] / df["YEARS"] ) 
df["NEW_CWALKS_PER_YEAR"] = (df["CWALKS"] / df["YEARS"] ) 
df["NEW_PUTOUTS_PER_YEAR"] = (df["PUTOUTS"] / df["YEARS"] ) 
df["NEW_CRBI_PER_YEAR"] = (df["CRBI"] / df["YEARS"])
df["NEW_EQA"] = (df["ATBAT"] + df["HITS"] + (1.5 * df["WALKS"]) ) / (df["ATBAT"] + df["WALKS"] )

df['NEW_HITS_RATE_CAT'] = pd.qcut(x=(df['NEW_HITS_RATE']), q=3, labels= ["low", "medium", "high"] )
df['NEW_NUM_OF_MISSING_RATE_CAT'] = pd.qcut(x=(df['NEW_NUM_OF_MISSING_RATE']), q=3,  labels= ["low", "medium", "high"] )
df["NEW_LEAGUE_BEST_PLAYER"] = np.where( ( (df["LEAGUE"] =="A") & (df["NEWLEAGUE"] =="N") &  (df["DIVISION"] =="E")), 1 , 0)
df["NEW_LEAGUE_BEST_PLAYER"] = np.where(((df["LEAGUE"] =="N") & (df["NEWLEAGUE"] =="N") & (df["DIVISION"] =="E") ), 1, df["NEW_LEAGUE_BEST_PLAYER"] )


# If Hits_Missing_Rate_Delta is greater than 1, this may represent that the player has been playing more successfully lately:

df["NEW_PLAYER_SUCCESS_INCREASED"] = np.where(df['NEW_HITS_MISSING_RATE_DELTA'] >1, "YES", "NO")
df["NEW_IS_YOUNG_TALENTED"] =  np.where(( (df['YEARS'] <= 5) & (df["NEW_HITS_RATE"] >=0.30)),"YES", "NO")

df["NEW_HITS_RATE/CHITS_RATE"] = (df['NEW_HITS_RATE'] / df['NEW_HITS_RATE_CAREER'] )
df["NEW_IS_PLAYER_UPGRADED"] =  np.where(( (df['YEARS'] >=5) & (df["NEW_HITS_RATE/CHITS_RATE"] >1.1 )) , "YES", "NO")

df.head()

# Yeni türettiğimiz değişkenlerden kaynaklı NaN Değerler oluşmuş olabilir yeniden kontrol edelim:
missing_values_df(df)

missing_values_df(df)
#                      n_miss  n_miss_ratio
# NEW_ERRORS/ASSISTS       17      5.329154
# NEW_ASSISTS/ERRORS       17      5.329154
# NEW_ASSISTS/PUTOUTS      15      4.702194
# NEW_WALKS/RBI             2      0.626959
# NEW_HMRUN/RUNS            1      0.313480
# NEW_WALKS/CWALKS          1      0.313480
# NEW_RBI/CRBI              1      0.313480
# NEW_CRBI/RBI              1      0.313480

# Bu değişkenleri ve türetirken kullandığımız değişkenleri inceleyelim,
# bölme işlemi kaynaklı olabilir?

df[["HMRUN", "CHMRUN", "NEW_HMRUN/CHMRUN"]][df[["HMRUN", "CHMRUN", "NEW_HMRUN/CHMRUN"]].isnull().any(axis=1)].head(3)
df[["HMRUN", "RUNS", "NEW_HMRUN/RUNS"]][df[["HMRUN", "RUNS", "NEW_HMRUN/RUNS"]].isnull().any(axis=1)].head(3)
df[["CWALKS", "WALKS", "NEW_WALKS/CWALKS"]][df[["CWALKS", "WALKS", "NEW_WALKS/CWALKS"]].isnull().any(axis=1)].head(3)
df[["RBI", "WALKS", "NEW_WALKS/RBI"]][df[["RBI", "CRBI", "NEW_WALKS/RBI"]].isnull().any(axis=1)].head(3)
df[["RBI", "WALKS", "NEW_WALKS/RBI"]][df[["RBI", "CRBI", "NEW_WALKS/RBI"]].isnull().any(axis=1)].head(3)
df[["RBI", "CRBI", "NEW_RBI/CRBI"]][df[["RBI", "CRBI", "NEW_CRBI/RBI"]].isnull().any(axis=1)].head(3)
df[["RBI", "CRBI", "NEW_RBI/CRBI"]][df[["RBI", "CRBI", "NEW_RBI/CRBI"]].isnull().any(axis=1)].head(3)
df[["ASSISTS", "PUTOUTS", "NEW_ASSISTS/PUTOUTS"]][df[["ASSISTS", "ERRORS",  "NEW_ASSISTS/PUTOUTS"]].isnull().any(axis=1)].head(3)
df[["ASSISTS", "ERRORS", "NEW_ASSISTS/ERRORS"]][df[["ASSISTS", "ERRORS", "NEW_ASSISTS/ERRORS"]].isnull().any(axis=1)].head(3)
df[["ASSISTS", "ERRORS", "NEW_ASSISTS/ERRORS"]][df[["ASSISTS", "ERRORS", "NEW_ERRORS/ASSISTS"]].isnull().any(axis=1)].head(3)

#  Tamamında bölme işlemi kaynaklı olduğunu gördük, bu nedenle 0 ile doldurabiliriz:
# Peki bu değişkenleri dinamik olarak nasıl seçeriz?

na_cols = [var for var in df.columns if df[var].isnull().sum() > 0]
df[na_cols] = df[na_cols].fillna(0)
df.isnull().sum().sum()


# NaN değer dışında bölme işlemi kaynaklı inf ya da  -inf değerler oluşmuş olabilir, 0 ile replace edelim:
df[df.isin([np.nan, np.inf, -np.inf]).any(axis=1)].shape[0]

# Replace [inf, -inf] values with zero:
df.replace([np.inf, -np.inf], 0, inplace=True)
df.shape

#  Yeni değişkenler türettik, şimdi numerik değişkenleri yeniden atayarak outlier değerlere odaklanalım:
num_cols = grab_col_names(df)[1]
num_cols

##############################################################
# 2. Outliers
##############################################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    q1 = dataframe[col_name].quantile(q1)  # 1.Çeyrek
    q3 = dataframe[col_name].quantile(q3)  # 3.Çeyrek
    interquantile_range = q3 - q1  # range'i hesaplayalım
    low_limit = q1 - 1.5 * interquantile_range # low & up limit:
    up_limit = q3 + 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
    
for col in num_cols:
    print(col, ":", check_outlier(df, col))


# Outlier değerler olduğunu gözlemledik şimdi LOF yöntemi ile ele alalım:

#LOF(Local Outliers Factor) method:

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df_ = df[num_cols]

# LOF Scores:
clf = LocalOutlierFactor(n_neighbors=5)
clf.fit_predict(df_)
df_scores = clf.negative_outlier_factor_

# LOF Visualization: 
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 10], style='.-')
plt.show()

# Görseldeki ilk 10 gözlemin skorlarını inceleyelim:
np.sort(df_scores)[0:10]

# Elbow yöntemi ile ele alındığında 6.indexi threshold belirleyebiliriz:

th = np.sort(df_scores)[0:10][6]
th #  -1.5849

# Bu thresholdun altında skora sahip olan gözlem sayısı:
df[df_scores < th].shape #6

df.drop(df[df_scores < th].index, inplace=True)
df.shape #  (313, 83)

##############################################################
# 3. Rare Encoding
##############################################################

cat_cols = grab_col_names(df)[0]

 # Her bir kategorik değişkenin sınıf frekanslarına ve target değişken ortalamasına bakalım:
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SALARY", cat_cols)


# Sınıf sayısının tek olması ya da 2 sınıflı değişkenden sınıflardan birinin frekanslarının %1 in altında olması
# bu değişkenin açıklayıcılığının olmadığını gösterir, bu değişkenleri silebiliriz:

useless_cols = [col for col in df.columns if ( ( df[col].nunique() == 2 
                                            and (df[col].value_counts() / len(df) < 0.01).any(axis=None))
                                            | df[col].nunique() == 1 )]
print(useless_cols)
df.drop(useless_cols, axis = 1,inplace=True)


# Rare Encoding: Sınıf frekansı çok düşük olan (örneğin %1, 2 den fazla sınıfı bulunan değişkenlerde düşük frekanslıları
# rare kategorisi altında birleştirerek, operasyonal maliyeti düşürebiliriz)

def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) <= rare_perc).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe) # th altında kalan sınıfı olan değişkenlerin sınıf frekanslarından olusan df yarat
        rare_labels = tmp[tmp <= rare_perc].index # sınıf frekansı < th olanların indexlerini bul
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col]) # th altında kalan değerleri Rare olarak grupla

    return dataframe

cat_cols = grab_col_names(df)[0]
df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "SALARY", cat_cols)

# Bu veri setinde böyle bir case yer almamaktadır.

##############################################################
#  One-Hot Encoding:
##############################################################

cat_cols = grab_col_names(df)[0]
dms = pd.get_dummies(df[cat_cols], drop_first=True, dtype="int64")
df_ = df.drop(columns=cat_cols, axis=1)
df = pd.concat([df_, dms],axis=1)
df.head(2)

###############################################################
# 4. Feature Scaling:
###############################################################

num_cols = grab_col_names(df)[1]
num_cols = [col for col in num_cols if "SALARY" not in col]
num_cols
df.head()

# Scaling:

# Target değişken dışındakileri scale etmek model sonucunun yorumlanabilir olmasını sağlar.
# Bu yüzden diğer numerik değişkenlere odaklanalım:

def StandartScaling(dataframe, col_name):
    ss = StandardScaler()
    dataframe[col_name] = ss.fit_transform(dataframe[col_name])
    return dataframe

def MinMaxScaling(dataframe, col_name):
    mms = MinMaxScaler()
    dataframe[col_name] = mms.fit_transform(dataframe[col_name])
    return dataframe

def RobustScaling(dataframe, col_name):
    rs = RobustScaler()
    dataframe[col_name] = rs.fit_transform(dataframe[col_name])
    return dataframe



def Scaling(dataframe, target, method="RobustScaling"):
    numerical_cols = grab_col_names(dataframe)[1]
    numerical_cols = [col for col in numerical_cols if target not in col.upper()]
    if method == "StandartScaling":
        StandartScaling(dataframe, numerical_cols)
    elif method == "MinMaxScaling":
        MinMaxScaling(dataframe, numerical_cols)
    else:
        RobustScaling(dataframe, numerical_cols)
    return dataframe


df[num_cols] = (Scaling(df[num_cols], "SALARY"))
df.head()

# Correlation matrix

def high_correlated_cols(dataframe, plot=False, corr_th=0.95):
    num_cols = grab_col_names(df)[1]
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True, corr_th=0.95)

high_correlated_col_df = high_correlated_cols(df, corr_th=0.95)

# df.drop(columns=high_correlated_col_df, axis=1, inplace=True)
# İlk olarak modeli kurup Importance'ı yüksek olan değişkenleri tutacak şekilde yüksek korelasyonlu değişkenlerden birini silebiliriz:


###############################################################
# Modelling**
###############################################################

# Bağımlı - Bağımsız değişkenler:
X = df.drop("SALARY", axis=1)
y = df[["SALARY"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=112)

# Model:

lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Model Performanse (RMSE)
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred) )) # 338.99

# Model Validation
np.sqrt(np.mean(-cross_val_score(lr_model,X_train,y_train, cv = 5, scoring="neg_mean_squared_error"))) # 250.074

# df["SALARY"].head(20)
# df["SALARY"].mean()
# df["SALARY"].median()

##############################################################################

# Base Models: Şimdi diğer modellerin performanslarını gözlemleyelim:
# Bunun için model adı ve objelerinin tuple olarak tutulduğu liste oluşturup, for döngüsü ile tümünü çalıştıralım

models = [('LR', LinearRegression()),
          ('Lasso', Lasso()),
          ('KNN', KNeighborsRegressor()),
          ('CRT',DecisionTreeRegressor()),
          ('RF' , RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGB", XGBRegressor(objective='reg:squarederror')),
          ("LGBM",LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
]

for name,  regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

#
# RMSE: 271.0296 (LR)
# RMSE: 251.2683 (Lasso)
# RMSE: 264.7055 (KNN)
# RMSE: 315.0568 (CRT)
# RMSE: 222.8122 (RF)
# RMSE: 418.1515 (SVR)
# RMSE: 220.2016 (GBM)
# RMSE: 235.1871 (XGB)
# RMSE: 230.9111 (LGBM)
####################################################################################

# Lasso Regression : Feature Importance için tercih edilebilir, coefficient değerleri 0 olanlar model
# açıklayılığı düşük değişkenler olduğunu gösterir:

# Fit Model:

Lasso_model = Lasso().fit(X,y)
# print(Lasso_model.coef_[0:10])
# print(Lasso_model.intercept_)

# Lasso Regression - Hyperparameter Optimization:
# Optimum alpha değerini bulup modeli kurarak değişkenlerin önem düzeylerini inceleyelim:

coefs = []   
alphas= 10**np.linspace(10,-2,100)*0.5
lasso_cv = LassoCV(alphas = alphas, cv = 10, max_iter=10000)
lasso_cv_model = lasso_cv.fit(X,y)

#Optimum alpha: Yukarıda tanımlanan alpha değerlerinden en iyi model çıktısını veren optimum değeri bulalım
lasso_cv_model.alpha_

# Lasso Regression - Tuned Model:
ls = Lasso(lasso_cv_model.alpha_)
lasso_tuned_model = ls.fit(X,y)
y_pred = lasso_tuned_model.predict(X)
print("RMSE:" , np.sqrt(mean_squared_error(y, y_pred))) # 202.661
print("r2 Score:", r2_score(y,y_pred)) #  0.769


# 10-Fold Cross-Validation:  247.7213
print("10 - Fold CV Score:", np.mean(np.sqrt(-cross_val_score(lasso_tuned_model, X, y, cv=10, scoring="neg_mean_squared_error"))))


# Lasso Regression Coefficients: Katsayılardan önem düzeyi düşük değişkenleri tespit edelim
Importance = pd.DataFrame({"Feature": X.columns, 
                           "Coefs" : lasso_tuned_model.coef_ })

Importance.sort_values("Coefs").sort_values("Coefs", ascending=True)["Coefs"]

# Katsayısı >0 olanları seçelim:
selected_features = list(Importance[Importance["Coefs"] > 0]["Feature"])
selected_features

# Şimdi önem düzeyi yüksek olan bu değişkenleri baz alarak (X setini bu değişkenler oluşturacak) yeniden modeli kuralım:
X = df[selected_features]
# X.columns
y = df[["SALARY"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=112)


lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Model Performanse (RMSE)
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred) )) # 321.739

# Model Validation
np.sqrt(np.mean(-cross_val_score(lr_model,X_train,y_train, cv = 5, scoring="neg_mean_squared_error"))) # 251.21
 