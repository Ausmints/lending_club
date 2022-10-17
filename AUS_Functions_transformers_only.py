import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ZipStateToCoordinates(BaseEstimator, TransformerMixin):
    """
    Transformer that converts state and zip_code columns into latitude and longitude.

    Parameters:
        state_coord_means(pd.DataFrame): A dataframe containing mean latitude and longitude values for each state
        bin_coord_means(str): A dataframe containing mean latitude and longitude values for each partial zip-code and state combination.
        state(bool): A toggle for whether the state mean latidude and longitude values are returned or zip code specific values.

    Returns:
        Dataframe with the latitude and longitude added and zip-code and state dropped.
    """

    def __init__(
        self, state_coord_means: pd.DataFrame, bin_coord_means: pd.DataFrame, state=True
    ):
        self.state_coord_means = state_coord_means
        self.bin_coord_means = bin_coord_means
        self.state = state

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        if self.state:
            X1 = (
                X1.reset_index()
                .merge(self.state_coord_means.reset_index(), on="state", how="left")
                .set_index("index")
            )
            X1 = X1.drop(["zip_code", "state"], axis=1)
            X1.rename(
                columns={"state_latitude": "latitude", "state_longitude": "longitude"}
            )

        else:
            X1["bin"] = X1["zip_code"].str.strip("x") + X1["state"]
            X1 = X1.merge(self.bin_coord_means.reset_index(), on="bin", how="left")
            X1 = X1.merge(self.state_coord_means.reset_index(), on="state", how="left")
            X1["latitude"] = X1["latitude"].fillna(X1["state_latitude"])
            X1["longitude"] = X1["longitude"].fillna(X1["state_longitude"])
            X1 = X1.drop(
                ["zip_code", "bin", "state", "state_latitude", "state_longitude"],
                axis=1,
            )

        return X1


class ExtractWords(BaseEstimator, TransformerMixin):
    """
    A one-hot-encoding transformer for the relevant words in the loan_title column.

    Parameters:
        relevant_words(list): A list containing the relevant words
    """

    def __init__(self, relevant_words: list):
        self.relevant_words = relevant_words

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for word in self.relevant_words:
            current_word = f"word_{word}"
            X1.loc[
                X1["loan_title"].str.contains(rf"(?:^|\W){word}(?:$|\W)").fillna(False),
                current_word,
            ] = 1
            X1.loc[:, current_word] = X1.loc[:, current_word].fillna(0).astype(int)
        X1 = X1.drop("loan_title", axis=1)
        return X1


class AddMissingColumn(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a "missing_data" column to the dataset, populated with binary values.
    Positive value is given if any of the features has a missing value in the row.

    Parameters:
        missing_list(list): A list containing the features with missing data
        one_column(bool): A toggle to whether make a single column showing the missing data, or a seperate column for each feature
    """

    def __init__(self, missing_list: list, one_column=True):
        self.missing_list = missing_list
        self.one_column = one_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        if self.one_column:
            for elem in self.missing_list:
                X1.loc[X1[elem].isna(), "missing_data"] = 1
            X1["missing_data"] = X1["missing_data"].fillna(0)

        else:
            for elem in self.missing_list:
                X1.loc[X1[elem].isna(), f"{elem}_missing"] = 1
            missing_columns = X1.columns[X1.columns.str.contains("_missing")]
            X1.loc[:, missing_columns] = X1.loc[:, missing_columns].fillna(0)

        return X1


class UnknownToNan(BaseEstimator, TransformerMixin):
    """
    Transformer that changes a differently marked missing value to np.nan.

    Parameters:
        missing_dict(dict): A dictonary containing {feature:value} pairs to transform into np.nan values.
    """

    def __init__(self, missing_dict):
        self.missing_dict = missing_dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.missing_dict:
            if self.missing_dict[key] == "nan":
                continue
            else:
                X1.loc[X1[key] == self.missing_dict[key], key] = np.nan

        return X1


class FillMissingData(BaseEstimator, TransformerMixin):
    """
    Transformer that fills the mising data for columns used in the first task -
    Missing data in risk_score column is filled using mean risk score values for each value in employment_length_years.
    Missing data in the debt_to_income_ratio is filled using median value.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        self.risk_score_employement_means = X.groupby("employment_length_years")[
            "risk_score"
        ].mean()
        self.column_median = X.median()
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        X1 = X1.join(
            self.risk_score_employement_means.rename("risk_score_mean"),
            on="employment_length_years",
        )
        X1["risk_score"] = X1["risk_score"].fillna(X1["risk_score_mean"])
        X1 = X1.drop("risk_score_mean", axis=1)

        X1["debt_to_income_ratio"] = X1["debt_to_income_ratio"].fillna(
            self.column_median["debt_to_income_ratio"]
        )
        return X1


class TransformToLog(BaseEstimator, TransformerMixin):
    """
    Transformer that transforms heavily skewed columns into log scale.

    Parameters:
        log_list(list): A list containing features to transform.
        drop_original(bool): A toggle whther to drop the original columns. Default value is "True".
    """

    def __init__(self, log_list, drop_original=True):
        self.log_list = log_list
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for column in self.log_list:
            if column in X1.columns:
                X1.loc[X1[column] == 0, column] = (
                    X1.loc[X1[column] > 0, column].min() / 2
                )
                X1[f"{column}_log"] = np.log(X1[column])
                if self.drop_original:
                    X1 = X1.drop(column, axis=1)
        return X1


class FillMissingDataAccepted(BaseEstimator, TransformerMixin):
    """
    Transformer that fills the missing data for columns used in the second task and also drops columns with too high multicollinarity -
    Columns in columns_to_drop are dropped
    Columns in months_columns either get binarized between missing and present values or fill the missing values with max_value*2
    Columns in max_corr_pairs.index fill the missing values with mean values for binned highest correlated features for each column
    The rest of the missing values get filled with median column value.

    Parameters:
        best_corrs_means_dict({str:pd.Series}): A dictionary containing series with feature_1 mean values for each feature_2 interval for each feature pair in max_corr_pairs.
        max_corr_pairs(pd.Series): A series containing missing_feature and corr_feature pairs.
        months_columns(list): A list containing features that contain "number_of_months_since_X" data
        columns_to_drop(list): A list containing features to drop to avoid multicollinearity
        binarize(str): Whether to binarize features in months_columns list. Values are ["all", "some", "none"] with "all" binarizing all of the features in the list that have more than 50% data missing. Default is "some".
    """

    def __init__(
        self,
        best_corrs_means_dict,
        max_corr_pairs,
        months_columns,
        columns_to_drop,
        binarize="some",
    ):
        self.best_corrs_means_dict = best_corrs_means_dict
        self.max_corr_pairs = max_corr_pairs
        self.months_columns = months_columns
        self.columns_to_drop = columns_to_drop
        self.binarize = binarize

    def fit(self, X: pd.DataFrame, y=None):
        X1 = X.copy()
        self.months_impute = X1.loc[:, self.months_columns].max() * 2
        self.median_impute = X1.loc[
            :,
            X1.columns[
                ((~X1.columns.isin(self.columns_to_drop)) & (X1.dtypes == "float"))
            ],
        ].median()
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()

        #Fixing problems in live model - not actually part of transformer
        X1.index.name  = "id"
        X1 = X1.reset_index()
        X1 = X1.rename(columns = {"level_0":"index"})
        X1 = X1.drop("id", axis=1)
        #Until here

        bin_columns = []
        if self.binarize == "some":
            bin_columns = X1[self.months_columns].columns[
                X1[self.months_columns].isna().sum() * 100 / X1.shape[0] > 70
            ]
        elif self.binarize == "all":
            bin_columns = X1[self.months_columns].columns[
                X1[self.months_columns].isna().sum() * 100 / X1.shape[0] > 50
            ]

        X1.loc[:, bin_columns] = ~X1.loc[:, bin_columns].isna()
        X1.loc[:, self.months_columns] = X1.loc[:, self.months_columns].fillna(
            self.months_impute
        )

        for column in self.max_corr_pairs.index:
            column_index = (
                self.best_corrs_means_dict[column].loc[:, "bin_column"].dtype.categories
            )
            X1.loc[:, "bin_column"] = pd.cut(
                X1[self.max_corr_pairs[column]], column_index
            )
            X1.loc[
                X1[column] > column_index.max().right, "bin_column"
            ] = column_index.max()
            X1.loc[
                X1[column] < column_index.min().left, "bin_column"
            ] = column_index.min()
            mean_column_values = (
                X1.reset_index()
                .loc[:, ["index", "bin_column"]]
                .merge(self.best_corrs_means_dict[column], on="bin_column")
                .set_index("index")
                .drop("bin_column", axis=1)
            )
            X1 = X1.merge(
                mean_column_values, how="left", left_index=True, right_on="index"
            )
            X1[column] = X1[column].fillna(X1["medians"])
            X1 = X1.drop(["medians", "bin_column"], axis=1)

        X1 = X1.loc[:, X1.columns[(~X1.columns.isin(self.columns_to_drop))]]

        X1.loc[:, X1.columns[X1.dtypes == "float"]] = X1.loc[
            :, X1.columns[X1.dtypes == "float"]
        ].fillna(self.median_impute)
        X1 = X1.fillna(0)

        self.column_names = X1.columns
        return X1


class SmallValuesToOther(BaseEstimator, TransformerMixin):
    """
    Transformer that transforms values in categorical features that encompass less than cutoff_num percent of data to "other".

    Parameters:
        column_names(list): List containing column names to transform
        cutoff_num(int): Percent cutoff number for filtering categories
    """

    def __init__(self, column_names: list, cutoff_num=5):
        self.cutoff_num = cutoff_num
        self.column_names = column_names

    def fit(self, X: pd.DataFrame, y=None):
        X1 = X.copy()
        filter_values_dict = {}
        for column in self.column_names:
            filter_values_dict[column] = (
                X1[column].value_counts()[
                    X1[column].value_counts() * 100 / X1.shape[0] < self.cutoff_num
                ]
            ).index.tolist()
        self.filter_values_dict = filter_values_dict
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.filter_values_dict:
            X1.loc[X1[key].isin(self.filter_values_dict[key]), key] = "other"
        return X1


class FilterCareers(BaseEstimator, TransformerMixin):
    """
    Transformer that adds "negative_corr_careers" and "positive_corr_careers" features by checking whether words in column_name are in "negative_corr_careers" and "positive_corr_careers" lists.

    Parameters:
        negative_corr_careers(list): List containing careers that negatively correlate with target column
        positive_corr_careers(list): List containing careers that positively correlate with target column
        column_name(str): Name of the column containing the carrer names
    """

    def __init__(self, negative_corr_careers, positive_corr_careers, column_name):
        self.negative_corr_careers = negative_corr_careers
        self.positive_corr_careers = positive_corr_careers
        self.column_name = column_name

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        split_str = X1[self.column_name].str.strip().str.split(" ", expand=True)
        split_str = split_str.loc[
            :, (~split_str.isna()).sum() * 100 / split_str.shape[0] > 1
        ]
        for column in split_str.columns:
            X1.loc[
                split_str[column].isin(self.negative_corr_careers),
                "negative_corr_careers",
            ] = 1
            X1.loc[
                split_str[column].isin(self.positive_corr_careers),
                "positive_corr_careers",
            ] = 1
        X1 = X1.drop(self.column_name, axis=1)
        X1.loc[:, ["negative_corr_careers", "positive_corr_careers"]] = X1.loc[
            :, ["negative_corr_careers", "positive_corr_careers"]
        ].fillna(0)

        return X1
