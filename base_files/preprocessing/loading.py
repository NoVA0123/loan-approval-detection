import os
import psutil
import polars as pl
import polars.selectors as cs
from base_files.utils import errors
from tqdm.auto import tqdm


# Reading the data
def read_file(FilePath1: str,
              FilePath2: str) -> tuple:

    Threshold1 = os.path.getsize(FilePath1)
    Threshold2 = os.path.getsize(FilePath2)
    TotalThreshold = Threshold1 + Threshold2

    Available = psutil.virtual_memory().available()

    try:
        if TotalThreshold * 2 > Available:
            raise errors.notEnoughMemory
    except errors.notEnoughMemory:
        print("Not enough memory to work on project")
        print(f"Available Memory: {Available/2**30} GiB")
        print(f"Memory Needed: {(TotalThreshold*2)/2**30} GiB")
        print("Currently working on chunking the data.")

    ExtName1 = FilePath1.split(sep='.')
    ExtName2 = FilePath1.split(sep='.')

    Extensions = {
            'csv': pl.read_csv,
            'xlsx': pl.read_excel,
            'ods': pl.read_ods,
            'parquet': pl.read_parquet
            }

    for x in tqdm(Extensions.keys()):
        if ExtName1 == x:
            df1 = Extensions[x](FilePath1)
        if ExtName2 == x:
            df2 = Extensions[x](FilePath2)

    return df1, df2


def uint_casting(dataframe: pl.DataFrame,
                 cols: list) -> dict:
    tmp = {}
    for x in cols:
        if dataframe[:, x].max < 2**8:
            tmp[x] = pl.UInt8
        elif dataframe[:, x].max < 2**16:
            tmp[x] = pl.UInt16
        elif dataframe[:, x].max < 2**32:
            tmp[x] = pl.UInt32
        else:
            tmp[x] = pl.UInt64

        return tmp


def int_casting(dataframe: pl.DataFrame,
                cols: list) -> dict:
    tmp = {}
    for x in cols:
        if dataframe[:, x].max < 2**7 and dataframe[:, x].min > -(2**7 + 1):
            tmp[x] = pl.Int8
        if dataframe[:, x].max < 2**15 and dataframe[:, x].min > -(2**15 + 1):
            tmp[x] = pl.Int16
        if dataframe[:, x].max < 2**31 and dataframe[:, x].min > -(2**31 + 1):
            tmp[x] = pl.Int32
        else:
            tmp[x] = pl.Int64

        return tmp


def float_casting(cols: list) -> dict:
    tmp = {}
    for x in cols:
        tmp[x] = pl.Float32
    return tmp


def digit_casting(df: pl.DataFrame,
                  UintCols: dict,
                  IntCols: dict,
                  FloatCols: dict):
    UintCasted = uint_casting(df,
                              UintCols)
    IntCasted = int_casting(df,
                            IntCols)
    FloatCasted = float_casting(FloatCols)

    FinalMap = {**UintCasted,
                **IntCasted,
                **FloatCasted}

    return FinalMap


def col_casting(df1: pl.DataFrame,
                df2: pl.DataFrame) -> tuple:

    df1UintCols = df1.select([cs.unsigned_integer()]).columns
    df1IntCols = df1.select([cs.signed_integer()]).columns
    df1FloatCols = df1.select([cs.float()]).columns
    df2UintCols = df2.select([cs.unsigned_integer()]).columns
    df2IntCols = df2.select([cs.signed_integer()]).columns
    df2FloatCols = df2.select([cs.float()]).columns

    # Type casting df1
    Df1Mapping = digit_casting(df1,
                               df1UintCols,
                               df1IntCols,
                               df1FloatCols)
    df1 = df1.cast(Df1Mapping)

    # Type casting df2
    Df2Mapping = digit_casting(df2,
                               df2UintCols,
                               df2IntCols,
                               df2FloatCols)
    df2 = df2.cast(Df2Mapping)

    return df1, df2
