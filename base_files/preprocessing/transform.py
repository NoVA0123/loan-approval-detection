import polars as pl
import polars.selectors as cs
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm


def pivot_creator(Index: str,
                  DataFrame: pl.DataFrame) -> list:
    # Pivot the dataframe to show it in cross tabulation form
    CrossTab = DataFrame.pivot(on="Approved_Flag",
                               index=Index,
                               values="Approved_Flag",
                               aggregate_function='len')
    # Remove index and convert it into list
    CrossTab = CrossTab.to_numpy()[:, 1:].tolist()
    return CrossTab


def suitable_cat_col(df: pl.DataFrame):

    # checking contingency of categorical variable with our target variable
    CheckCols = df.select([pl.col(pl.String)]).columns[:-1]
    ColsToRmv = []

    for i in tqdm(CheckCols):
        chi2, pval, _, _ = chi2_contingency(pivot_creator(i, df))
        if pval <= 0.05:
            ColsToRmv.append(i)

    df = df.drop(ColsToRmv)
    return df


def suitable_num_cols(df: pl.DataFrame):

    # VIF sequential check
    VifData = df.select([cs.integer(), cs.float()])
    ColKept = []
    ColIndex = 0

    for i in VifData.columns[1:]:

        VifValue = variance_inflation_factor(VifData, ColIndex)

        if VifValue <= 6.:
            ColKept.append(i)
            ColIndex = ColIndex + 1

        else:
            VifData = VifData.drop(i)

    # Checking Anova for columns to be kept
    GrpDf = df.group_by('Approved_Flag').all()
    UniqueItems = GrpDf[:, 'Approved_Flag'].to_list()
    FinalColsNum = []
    print(UniqueItems)
    for i in ColKept:
        GrpPs = {}
        for NumFlag in range(len(UniqueItems)):
            GrpPs[UniqueItems[NumFlag]] = GrpDf[NumFlag, i].to_list()

        FStatistic, Pval = f_oneway(*GrpPs.values())
        if Pval <= 0.5:
            FinalColsNum.append(i)

    return FinalColsNum


def treating_every_col(df: pl.DataFrame):
    FinalColsNum = suitable_num_cols(df)

    # Treating categorical variables
    Mapper = {'SSC': 1,
              '12TH': 2,
              'GRADUATE': 3,
              'UNDER GRADUATE': 3,
              'POST-GRADUATE': 4,
              'PROFESSIONAL': 3,
              'OTHERS': 1}

    df = df.with_columns(pl.col('EDUCATION').replace(Mapper).cast(pl.UInt8))

    # Creating final list
    FinalFeatures = FinalColsNum + ['MARITALSTATUS',
                                    'EDUCATION',
                                    'GENDER',
                                    'last_prod_enq2',
                                    'first_prod_enq2',
                                    'Approved_Flag']
    df = df[:, FinalFeatures]

    # One hot encoding
    OneHotCols = ['MARITALSTATUS',
                  'GENDER',
                  'last_prod_enq2',
                  'first_prod_enq2']
    df = df.to_dummies(columns=OneHotCols)

    return df
