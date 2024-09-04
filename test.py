import numpy as np
import pandas as pd
import polars as pl
import duckdb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings('ignore')


# reading the data
df1 = pl.read_excel('~/Downloads/case_study1.xlsx')
df2 = pl.read_excel('~/Downloads/case_study2.xlsx')
print(df1.describe())
print(df2.describe())


# Storing the orginal data
Originaldf1 = df1.clone()
Originaldf2 = df2.clone()


# removing '-99999' from  oldest TL
# print(df1.shape)
df1 = df1.filter(pl.col('Age_Oldest_TL') != -99999)
# print(df1.shape)


# removing columns if null values (-99999) have occurred more than 10000 times
# print(df2.shape)
ColsToRmv = []
for i in df2.select([pl.col(pl.Int64), pl.col(pl.Float64)]).columns:
    if df2.filter(pl.col(i) == -99999).shape[0] > 10000:
        ColsToRmv.append(i)
# print(ColsToRmv)
df2 = df2.drop(ColsToRmv)
# print(df2.shape)
for i in df2.select([pl.col(pl.Int64), pl.col(pl.Float64)]).columns:
    df2 = df2.filter(pl.col(i) != -99999)
print(df2.shape)


# Joining both dataframes
df = df1.join(df2, how='inner', left_on='PROSPECTID', right_on='PROSPECTID')
print(df[:5])
print(df.shape)


# Checking Null values
# print(df.null_count().sum_horizontal())


# checking contingency of categorical variable with our target variable
CheckCols = df2.select([pl.col(pl.String)]).columns[:-1]
def pivot_creator(Index: str,
                  DataFrame: pl.DataFrame = df) -> list:
    # Pivot the dataframe to show it in cross tabulation form
    CrossTab = DataFrame.pivot(on="Approved_Flag", index=Index, values="Approved_Flag", aggregate_function='count')
    # Remove index and convert it into list
    CrossTab = CrossTab.to_numpy()[:, 1:].tolist()
    return CrossTab

# print(CheckCols)
for i in CheckCols:
    chi2, pval, _, _ = chi2_contingency(pivot_creator(i))
    print(f'{i}: {pval}')


# VIF sequential check
VifData = df.select([pl.col(pl.Int64), pl.col(pl.Float64)])
ColKept = []
ColIndex = 0

for i in VifData.columns[1:]:

    VifValue = variance_inflation_factor(VifData, ColIndex)
    print(f'{ColIndex}: {VifValue}')


    if VifValue <= 6.:
        ColKept.append(i)
        ColIndex = ColIndex + 1

    else:
        VifData = VifData.drop(i)

print(ColKept)
