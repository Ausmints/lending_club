import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

sns.set()
colors = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "grey": "#8C8C8C",
}

###


def draw_simple_barplot(
    x_column: pd.Series,
    y_column: pd.Series,
    metadata: list,
    rotate_x=False,
    largefig=True,
):
    """
    Draws a simple barplot with title and labels set

      Parameters:
        x_column(pd.Series): Column containing data for x axis
        y_column(pd.Series): Column containing data for y axis
        metadata(list): Contains a list of 3 elements : title, label for x axis, label for y axis
        rotate_x(Bool): Sets whether rotate x labels for better readability. Default is "False".
        largefig(Bool): Sets whether to create large or small plot. Default is "True".

      Returns:
        Nothing
    """
    if largefig:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10, 5))

    ax = sns.barplot(x=x_column, y=y_column)

    if rotate_x:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, horizontalalignment="right"
        )
    ax.set_title(metadata[0])
    ax.set(xlabel=metadata[1], ylabel=metadata[2])

    plt.show()


###


def draw_color_barplot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    sort_column: str,
    metadata: list,
    category_list_1: list,
    category_list_2=[],
    rotate_x=False,
    largefig=True,
):
    """
    Draws a barplot from the input features with possibility to color specific features orange, blue and grey

      Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        y_column(str): Name of the column containing data for y axis
        sort_column(str): Name of the column used for sorting the x column
        metadata(list): Contains a list of 3 elements : title, label for x axis, label for y axis
        category_list(list): Contains the categories from x column to be colored blue
        category_list(list): Contains the categories from x column to be colored orange. Default is empty list.
        rotate_x(Bool): Sets whether rotate x labels for better readability. Default is "False".
        largefig(Bool): Sets whether to create large or small plot. Default is "True".

      Returns:
        Nothing
    """
    if not sort_column:
        sort_column = df.columns[0]
    temp_df = (
        df.sort_values(sort_column, ascending=True)
        .reset_index(drop=True)
        .loc[:10, [x_column, y_column]]
    )
    temp_df[x_column] = pd.Categorical(
        temp_df[x_column], categories=list(temp_df.loc[:10, x_column])
    )
    if not category_list_2:
        colors_graph = [
            colors["blue"] if (x in category_list_1) else colors["grey"]
            for x in temp_df[x_column]
        ]
    else:
        colors_graph = [
            colors["blue"]
            if (x in category_list_1)
            else colors["orange"]
            if (x in category_list_2)
            else colors["grey"]
            for x in temp_df[x_column]
        ]

    if largefig:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10, 5))

    ax = sns.barplot(data=temp_df, x=x_column, y=y_column, palette=colors_graph)

    if rotate_x:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, horizontalalignment="right"
        )
    ax.set_title(metadata[0])
    ax.set(xlabel=metadata[1], ylabel=metadata[2])

    plt.show()


###


def draw_boxplot(df: pd.DataFrame, x_column: str, metadata: list):
    """
    Draws a simple boxplot with title and labels set

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        metadata(list): Contains a list of 3 elements : title, label for x axis, label for y axis

      Returns:
        Nothing
    """
    plt.figure(figsize=(20, 3))
    ax = sns.boxplot(data=df, x=x_column)
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###


def draw_histplot(df: pd.DataFrame, x_column: str, metadata: list, hue_param=""):
    """
    Draws either a uniform histplot or a comparison histplot with the x_column values split by a binary column.

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        metadata(list): Contains a list of 4 elements : title, label for x axis, label for y axis, and two labels for legend if the x_column is split (if histplot is drawn uniformly, no need to include last two).
        hue_param(str): Name of the binary column that is used to split the x_column. Default is empty string and the histplot is drawn uniformly

      Returns:
        Nothing
    """
    plt.figure(figsize=(20, 10))
    if hue_param:
        ax = sns.histplot(
            data=df,
            x=x_column,
            hue=hue_param,
            palette=[colors["grey"], colors["orange"]],
        )
        plt.legend(labels=[metadata[3], metadata[4]], title="")
    else:
        ax = sns.histplot(data=df, x=x_column)
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###


def draw_kdeplot(df: pd.DataFrame, x_column: str, hue_param: str, metadata: list):
    """
    Draws a comparison kdeplot with the x_column values split by a binary column. The kde lines each sum up to 1, not together.

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        x_column(str): Name of the column containing data for x axis
        hue_param(str): Name of the binary column that is used to split the x_column.
        metadata(list): Contains a list of 4 elements : title, label for x axis, label for y axis, and two labels for legend.


      Returns:
        Nothing
    """
    plt.figure(figsize=(20, 10))
    ax = sns.kdeplot(
        data=df,
        x=x_column,
        hue=hue_param,
        palette=[colors["grey"], colors["orange"]],
        common_norm=False,
    )
    plt.legend(labels=[metadata[3], metadata[4]], title="")
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###


def draw_proportion_barplot(data: pd.Series, metadata: list):
    """
    Draws a horizontal barplot that is split between orange and blue colors to show proportions of a binary value count. The input values are transformed into percentages.

      Parameters:
        data(pd.Series): Contains the value counts and names as indexes of the binary data.
        metadata(list): Contains a list of 4 elements : title, label for x axis, two labels for y axis - for the binary class names. If the labels for y axis are given as empty strings, the labels are taken from the index of data parameter.

      Returns:
        Nothing
    """
    percentages = [
        float(data.iloc[0] * 100 / (data.iloc[0] + data.iloc[1])),
        100 - float(data.iloc[0] * 100 / (data.iloc[0] + data.iloc[1])),
    ]

    plt.figure(figsize=(20, 3))
    ax = sns.barplot(x=[100], color=colors["orange"])
    ax = sns.barplot(x=[percentages[1]], color=colors["blue"])
    if metadata[2]:
        ax.set(xlabel=metadata[1], ylabel=metadata[2])
    else:
        ax.set(xlabel=metadata[1], ylabel=data.index[1])
    ax.set_title(metadata[0])
    plt.xlim(0, 100)

    patches = ax.patches
    for i in range(len(patches)):
        if i == 0:
            x = (
                patches[i].get_x()
                + patches[i].get_width()
                - (patches[i].get_width() - patches[1].get_width()) / 2
                - 3
            )
        else:
            x = patches[i].get_x() + patches[i].get_width() / 2 - 3
        y = patches[i].get_y() + 0.5
        ax.annotate(
            "{:.2f}%".format(percentages[i]),
            (x, y),
            size=20,
            xytext=(5, 10),
            textcoords="offset points",
            color="white",
        )

    ax2 = ax.twinx()
    if metadata[3]:
        ax2.set(yticklabels=[], ylabel=metadata[3], yticks=[0.5])
    else:
        ax2.set(yticklabels=[], ylabel=data.index[0], yticks=[0.5])
    ax2.grid(False)

    plt.show()


###


def draw_comparison_barplot(
    df: pd.DataFrame,
    binary_column: str,
    operating_column: str,
    category_list_1: list,
    metadata: list,
    rotate_x=False,
    largefig=True,
    mode="count",
):
    """
    Draws a double barplot for comparing a multiple category column's value counts split by a different binary column. If mode_count is False, then just takes the values of "count" column in the provided dataframe. It's possible to color some columns in orange/blue or grey.

    Parameters:
      df(pd.DataFrame): A dataframe containing both the binary and the operating column
      binary_column(str): Name of the column containing the binary data
      operating_column(str): Name of the column containing multiple category data - plotted on the x axis
      category_list_1(list): A list of categories from the operating column to color orange/blue. If empty, all categories will be orange/blue
      metadata(list): A list containing 5 values - Title, label for x axis, label for y axis, and two labels for the categories in binary column. If the last two are given as empty string, automatically sets the labels as the values from the binary column.
      rotate_x(Bool): Sets whether rotate x labels for better readability. Default is "False".
      largefig(Bool): Sets whether to create large or small plot. Default is "True".
      mode(str): Sets whether to count the amount of values ("count") or the proportions of the values ("proportion") in the binary column or use the "count" column in the provided dataframe. Default is "count".
    """
    temp_df = df
    if (mode == "count") or (mode == "proportion"):
        id_column = temp_df.index.name
        temp_df = (
            temp_df.reset_index()
            .loc[:, [operating_column, binary_column, id_column]]
            .groupby([operating_column, binary_column])
            .count()
            .reset_index()
            .rename(columns={id_column: "count"})
        )
        if mode == "proportion":
            if len(df[operating_column].unique()) < 3:
                temp_df = temp_df.merge(
                    temp_df.groupby(operating_column)["count"]
                    .sum()
                    .rename("count_total"),
                    left_on=operating_column,
                    right_index=True,
                )
            else:
                temp_df = temp_df.merge(
                    temp_df.groupby(binary_column)["count"].sum().rename("count_total"),
                    left_on=binary_column,
                    right_index=True,
                )
            temp_df["count"] = temp_df["count"] / temp_df["count_total"]
            temp_df.drop("count_total", inplace=True, axis=1)
    temp_df = temp_df.set_index([operating_column, binary_column])["count"].unstack()
    temp_df = temp_df.sort_values(operating_column, ascending=True).reset_index()
    temp_df[operating_column] = pd.Categorical(
        temp_df[operating_column], categories=list(temp_df.loc[:, operating_column])
    )

    if category_list_1:
        category_list_2 = [
            x for x in df[operating_column] if (x not in category_list_1)
        ]
    else:
        category_list_1 = [x for x in df[operating_column]]
        category_list_2 = []
    binary_values = [temp_df.columns[1], temp_df.columns[2]]
    colors_list_1 = {
        binary_values[1]: colors["blue"],
        binary_values[0]: colors["orange"],
    }
    colors_list_2 = [colors["grey"] for x in df[operating_column]]

    temp_df = temp_df.melt(id_vars=[operating_column])

    if largefig:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=temp_df[(temp_df[operating_column].isin(category_list_1))],
        x=operating_column,
        y="value",
        hue=binary_column,
        palette=colors_list_1,
    )
    if category_list_2:
        sns.barplot(
            data=temp_df[(temp_df[operating_column].isin(category_list_2))],
            x=operating_column,
            y="value",
            hue=binary_column,
            palette=colors_list_2,
            ax=ax,
        )
    ax.set(xlabel=metadata[1], ylabel=metadata[2])
    if rotate_x:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, horizontalalignment="right"
        )
    ax.set_title(metadata[0])

    if metadata[4] and metadata[3]:
        blue_patch = mpatches.Patch(color=colors["blue"], label=metadata[4])
        orange_patch = mpatches.Patch(color=colors["orange"], label=metadata[3])
    else:
        blue_patch = mpatches.Patch(color=colors["blue"], label=binary_values[1])
        orange_patch = mpatches.Patch(color=colors["orange"], label=binary_values[0])
    plt.legend(handles=[orange_patch, blue_patch])

    plt.show()


def draw_confusion_heatmap(matrix: np.ndarray, metadata: list):
    """
    Draws a simple heatmap from the given confusion matrix

    Parameters:
      matrix(np.ndarray): An array containing the confusion matrix
      metadata(list): Contains 3 strings and a list - The title, labels for X and Y axis and a list containing the labels for categories in the confusion matrix

    Returns:
      Nothing
    """
    plt.figure(figsize=(20, 8))
    ax = sns.heatmap(
        matrix, xticklabels=metadata[3], yticklabels=metadata[3], annot=True, fmt="g"
    )
    ax.set_xlabel(metadata[1])
    ax.set_ylabel(metadata[2])
    ax.set_title(metadata[0])

    plt.show()


###

###


def grid_search_results_to_df(results: dict, model_name: str):
    """
    Transforms parameter results from GridSearchCV from dict to df, combines with mean test and train score and also gets rid of "<model_name>__" prefix added for parameter use in pipeline

    Parameters:
      results(dict): cv_results dict given by GridSearchCV
      mosel_name(str): name of the model used in the pipeline fed into the GridSearchCV
    """
    grid_df = (
        pd.DataFrame(results)
        .loc[:, ["mean_test_score", "mean_train_score", "params"]]
        .sort_values("mean_test_score", ascending=False)
        .iloc[:10]
    )
    grid_df = (
        grid_df.reset_index()
        .loc[:, ["mean_test_score", "mean_train_score"]]
        .join(pd.DataFrame(grid_df["params"].tolist()))
    )
    grid_df.columns = grid_df.columns.str.replace(model_name + "__", "")
    return grid_df


###

###


class UnknownToNan(BaseEstimator, TransformerMixin):
    """
    Transformer that changes a differently marked missing value to np.nan.
    """
    def __init__(self, value):
        self.value = value

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X[X == self.value] = np.nan
        return X


class Reshape_1_2_D(BaseEstimator, TransformerMixin):
    """
    Transformer that changes the Series to DataFrame object, and the opposite.
    """
    def __init__(self, mode):
        self.mode = mode

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if self.mode == "toDf":
            return X.to_frame()
        else:
            return pd.Series(X.reshape(-1))


class AddMissingColumn(BaseEstimator, TransformerMixin):
    """
    Transformer that adds a "missing_data" column to the dataset, populated with binary values.
    Positive value is given if any of the features has a missing value in the row, with a
    dictionary passed as an argument with {feature:value} pairs that show the missing
    value for the feature.
    """
    def __init__(self, missing_dict):
        self.missing_dict = missing_dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.missing_dict:
            if self.missing_dict[key] == "nan":
                X1.loc[X1[key].isna(), "missing_data"] = 1
            else:
                X1.loc[X1[key] == self.missing_dict[key], "missing_data"] = 1

        X1["missing_data"].fillna(0, inplace=True)
        return X1
    
    
    
class StringToDatetime(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        return pd.to_datetime(X1)

    
class DatetimeToMonth(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for column in X1.columns:
            X1[column+"_month"] = X1[column].dt.month
        return X1
    
    


    
class ReplaceString(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, replace_dict):
        self.replace_dict = replace_dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.replace_dict:
            X1[key] = X1[key].replace(self.replace_dict[key])
 
        return X1


class ReplaceStringRegex(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, replace_dict):
        self.replace_dict = replace_dict

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        for key in self.replace_dict:
            X1[key] = X1[key].replace(self.replace_dict[key])
 
        return X1
    
    
class CleanData(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self):
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        X1["application_date"] = pd.to_datetime(X1["application_date"])
        X1["employment_length"] = X1["employment_length"].replace({"10+ years":"15", "< 1 year":"0"})
        X1["employment_length"] = X1["employment_length"].str.extract(r"([0-9]+)").astype(float)
        X1.rename(columns={"employment_length" : "employment_length_years"})
        X1["debt_to_income_ratio"] = X1["debt_to_income_ratio"].replace("%", "", regex=True).astype(float)
        X1["month"] = X1["application_date"].dt.month
        return X1
    
    
    
    
    

class AgeGroupsMeanBmi(BaseEstimator, TransformerMixin):
    """
    Transformer that bins age feature, takes the mean value of bmi for each age bin,
    creates a "age_mean_bmi" column, then imputes the missing bmi values with
    the mean value for the age bin.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        X1 = X.copy()
        X1.loc[:, "age_bins"] = pd.cut(X1["age"], 10)
        self.age_bins = X1["age_bins"].dtype.categories
        self.mean_bmi = (
            X1.groupby("age_bins")["bmi"].mean().rename("age_mean_bmi").round(1)
        )

        return self

    def transform(self, X: pd.DataFrame):
        X1 = X.copy()
        X1.loc[:, "age_bins"] = pd.cut(X1["age"], self.age_bins)
        X1.loc[X1["age"] > self.age_bins.max().right, "age_bins"] = self.age_bins.max()
        mean_age_bmi = (
            X1.reset_index()
            .loc[:, ["id", "age_bins"]]
            .merge(self.mean_bmi, on="age_bins")
            .set_index("id")
            .drop("age_bins", axis=1)
        )
        X1 = X1.merge(mean_age_bmi, left_index=True, right_on="id").drop(
            "age_bins", axis=1
        )
        X1["bmi"].fillna(X1["age_mean_bmi"], inplace=True)
        return X1
