import polars as pl
import polars.selectors as cs
from tqdm.auto import tqdm


def filter_nan(df1,
               df2):

    # removing '-99999' from  oldest TL
    # print(df1.shape)
    df1 = df1.filter(pl.col('Age_Oldest_TL') != -99999)
    # print(df1.shape)

    '''
    removing columns if null values (-99999) have occurred
    more than 10000 times.
    '''
    # print(df2.shape)
    ColsToRmv = []
    for i in tqdm(df2.select([cs.integer(), cs.float()]).columns):
        if df2.filter(pl.col(i) == -99999).shape[0] > 10000:
            ColsToRmv.append(i)
    # print(ColsToRmv)
    df2 = df2.drop(ColsToRmv)
    # print(df2.shape)
    for i in tqdm(df2.select([cs.integer(), cs.float()]).columns):
        df2 = df2.filter(pl.col(i) != -99999)

    return df1, df2


def join_df(df1: pl.DataFrame,
            df2: pl.DataFrame):

    # Joining both dataframes
    df = df1.join(df2, how='inner',
                  left_on='PROSPECTID',
                  right_on='PROSPECTID')

    return df
