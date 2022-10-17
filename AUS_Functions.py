import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import geopandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
import xgboost as xg
from sklearn.pipeline import Pipeline


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
    y_labels=True,
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

    if not y_labels:
        ax.set_yticks([])

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


def draw_usa(df, column, title, legend=True) -> None:
    """
    Draws the map of USA, with the states colored in different intensity given by the "column" attribute in the "df".

    Parameters:
        df(pd.DataFrame): A dataframe containing data for plotting
        column(str): Name of the column containing data for state color intensity
        title(str): Contains the title for the graph
        legend(bool): Toggle for the legend explaining the colors. Default is "True"

      Returns:
        Nothing

    """
    ax = df.plot(figsize=(30, 10), column=column, legend=legend)
    ax.plot(
        [-14200000, -14200000], [7000000, 5000000], linewidth=2, color=colors["black"]
    )
    ax.plot(
        [-14200000, -16500000], [5000000, 5000000], linewidth=2, color=colors["black"]
    )
    ax.plot(
        [-14200000, -13500000], [5000000, 3000000], linewidth=2, color=colors["black"]
    )
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(title)
    plt.show()


###

###


def df2_to_df(df2: pd.DataFrame) -> pd.DataFrame:
    """
    Function that transforms data from accepted application dataset into similar format to the data in the rejected application dataset.
    """
    df2 = df2.loc[
        :,
        [
            "loan_amnt",
            "title",
            "fico_range_low",
            "fico_range_high",
            "dti",
            "zip_code",
            "addr_state",
            "emp_length",
            "loan_status",
        ],
    ]
    df2["risk_score"] = (df2["fico_range_low"] + df2["fico_range_high"]) / 2
    df2["accepted"] = 1
    df2.loc[df2["loan_status"] == "Charged Off", "accepted"] = 2
    df2 = df2.drop(["fico_range_low", "fico_range_high", "loan_status"], axis=1)

    df2_name_dict = {
        "loan_amnt": "amount_requested",
        "title": "loan_title",
        "dti": "debt_to_income_ratio",
        "addr_state": "state",
        "emp_length": "employment_length",
    }
    df2 = df2.rename(columns=df2_name_dict)
    return df2


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that cleans up data for the first task for EDA.
    """
    df["employment_length"] = df["employment_length"].replace(
        {"10+ years": "15", "< 1 year": "0"}
    )
    df["employment_length"] = (
        df["employment_length"].str.extract(r"([0-9]+)").astype("float")
    )
    df = df.rename(columns={"employment_length": "employment_length_years"})
    df["debt_to_income_ratio"] = (
        df["debt_to_income_ratio"].replace("%", "", regex=True).astype(float)
    )
    df["amount_requested"] = df["amount_requested"].astype(int)

    df.loc[:, "loan_title"] = (
        df["loan_title"].str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)
    )

    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    return df


def zip_state_to_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that converts state and zip_code columns into latitude and longitude.
    """
    df = df.drop(df[df["zip_code"].isnull()].index.tolist(), axis=0)
    df["bin"] = df["zip_code"].str.strip("x") + df["state"]
    df = df.merge(bin_coord_means.reset_index(), on="bin", how="left")
    df = df.merge(state_coord_means.reset_index(), on="state", how="left")
    df["latitude"] = df["latitude"].fillna(df["state_latitude"])
    df["longitude"] = df["longitude"].fillna(df["state_longitude"])
    df.drop(["zip_code", "bin", "state"], axis=1, inplace=True)

    return df


def customize_us_shapefile(states: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """
    Function that customizes us shapefile changing Alaska and Hawaii location for better visualization.

    Returns:
        Transformed states shapefile
    """
    states = states.loc[states["STUSPS"].isin(df["state"].unique())]
    states.loc[states["NAME"] == "Alaska", "geometry"] = states.loc[
        states["NAME"] == "Alaska", "geometry"
    ].scale(0.3, 0.3)
    states.loc[states["NAME"] == "Alaska", "geometry"] = states.loc[
        states["NAME"] == "Alaska", "geometry"
    ].translate(-10000000, -3000000)
    states.loc[states["NAME"] == "Hawaii", "geometry"] = states.loc[
        states["NAME"] == "Hawaii", "geometry"
    ].translate(3000000, 1500000)

    alaska = states.loc[states["NAME"] == "Alaska", "geometry"]
    alaska_geom = alaska[47]
    alaska_bool_list = []
    for P in alaska_geom.geoms:
        alaska_bool_list.append(P.bounds[0] < -10000000)
    alaska = alaska.explode(index_parts=True)
    alaska = alaska.reset_index().set_index("level_1")
    alaska = MultiPolygon(list(alaska[alaska_bool_list].geometry))

    states.loc[states["NAME"] == "Alaska", "geometry"] = pd.Series([alaska]).values

    return states


def get_relevant_words(df: pd.DataFrame, target: pd.Series) -> list:
    """
    Function that filters out the words in the "loan_title" whose usage frequency differs between the accepted and rejected applications by at least 1% of total words used count.
    Also, to prevent multicollinearity in modelling, if words have a correlation higher than 0.7, then only the word with a higher correlation with target column is kept.

    Parameters:
        df(pd.DataFrame): A dataframe containing "loan_title" column
        target(pd.Series): A series containing the target data.

      Returns:
        A list containing the filtered words
    """
    word_df = (
        pd.Series(
            [y for x in df["loan_title"].dropna().values.flatten() for y in x.split()]
        )
        .value_counts()
        .to_frame("total")
    )
    word_df["total"] = (word_df["total"] / word_df["total"].sum()).round(3)
    for index, value in enumerate(["rejected", "accepted"]):
        word_df[value] = pd.Series(
            [
                y
                for x in df.loc[target == index, "loan_title"].dropna().values.flatten()
                for y in x.split()
            ]
        ).value_counts()
        word_df[value] = (word_df[value] / word_df[value].sum()).round(3)

    difference_words = (
        word_df[np.abs(word_df["rejected"] - word_df["accepted"]) > 0.01]
    ).index.to_list()

    word_df = pd.DataFrame(index=df.index)
    for word in difference_words:
        word_df.loc[
            df["loan_title"].str.contains(rf"(?:^|\W){word}(?:$|\W)").fillna(False),
            word,
        ] = 1
        word_df.loc[:, word] = word_df.loc[:, word].fillna(0).astype(int)

        word_corr = word_df.corrwith(word_df[word])
        multicoll_list = word_corr[
            (word_corr > 0.7) & (word_corr.index != word)
        ].index.tolist()
        if multicoll_list:
            for multicoll_name in multicoll_list:
                if np.abs(word_df[multicoll_name].corr(target)) < np.abs(
                    word_df[word].corr(target)
                ):
                    word_df = word_df.drop(multicoll_name, axis=1)
                else:
                    word_df = word_df.drop(word, axis=1)
                    break
    return word_df.columns.to_list()


def clean_data_2_1(df):
    """
    Function 1 of 2 for cleaning data for second task - transforming data into correct format, dropping columns with too little variability, etc.
    """
    df = df.drop(df[df.isna().all(axis=1)].index, axis=0)

    df["risk_score"] = (df["fico_range_high"] + df["fico_range_low"]) / 2
    df = df.drop(
        [
            "fico_range_high",
            "fico_range_low",
            "sec_app_fico_range_high",
            "sec_app_fico_range_low",
            "title",
            "url",
            "funded_amnt",
        ],
        axis=1,
    )

    df.loc[:, ["earliest_cr_line", "issue_d"]] = df.loc[
        :, ["earliest_cr_line", "issue_d"]
    ].astype("datetime64[D]")

    df = df.drop(
        df.columns[df.columns.str.contains("joint") | df.columns.str.contains("sec_")],
        axis=1,
    )

    top_value_percentage = (
        df.loc[:, df.columns[~(df.dtypes == "string")]].apply(
            lambda x: x.value_counts().max(), axis=0
        )
        * 100
        / df.shape[0]
    )
    df = df.drop(top_value_percentage[top_value_percentage > 98].index, axis=1)

    df["emp_length"] = df["emp_length"].replace({"10+ years": "15", "< 1 year": "0"})
    df["emp_length"] = df["emp_length"].str.extract(r"([0-9]+)").astype("float")
    df = df.rename(columns={"emp_length": "emp_length_years"})

    df["home_ownership"] = df["home_ownership"].str.lower()
    df = df.rename(columns={"addr_state": "state"})

    df["mths_since_earliest_cr_line"] = (
        (df["issue_d"] - df["earliest_cr_line"]) / np.timedelta64(1, "M")
    ).round()
    df = df.drop("earliest_cr_line", axis=1)

    sub_grade = (
        pd.Series(df["sub_grade"].dropna().unique())
        .sort_values(ascending=False)
        .tolist()
    )
    sub_grade_dict = {}
    grade_dict = {}
    num = 0
    grade_num = 7
    for grade in sub_grade:
        sub_grade_dict[grade] = num
        if (num == 5) | (num % 5 == 0):
            grade_dict[grade[0]] = grade_num
            grade_num -= 1
        num += 1

    df["sub_grade"] = df["sub_grade"].replace(sub_grade_dict)
    df["grade"] = df["grade"].replace(grade_dict)

    numeric_cols = df.columns[
        ((df.dtypes == "float") | (df.dtypes == "int"))
        & (~df.columns.isin(["sub_grade", "int_rate", "grade"]))
    ]
    for column in numeric_cols:
        if column in df.columns:
            col_corr = df.corrwith(df[column])
            multicoll_list = col_corr[
                (col_corr > 0.7) & (col_corr.index != column)
            ].index.tolist()
            if multicoll_list:
                for corr in multicoll_list:
                    if np.abs(df[corr].corr(df["sub_grade"])) < np.abs(
                        df[column].corr(df["sub_grade"])
                    ):
                        df = df.drop(corr, axis=1)
                    else:
                        df = df.drop(column, axis=1)
                        break

    df = df.convert_dtypes()

    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    return df


def clean_data_2_2(df):
    """
    Function 2 of 2 for cleaning data for second task - transforming data into correct format, dropping columns with too little variability, etc.
    """
    df = df.drop(df[df.isna().all(axis=1)].index, axis=0)

    df["risk_score"] = (df["fico_range_high"] + df["fico_range_low"]) / 2
    df["sec_risk_score"] = (
        df["sec_app_fico_range_high"] + df["sec_app_fico_range_low"]
    ) / 2
    df = df.drop(
        [
            "fico_range_high",
            "fico_range_low",
            "sec_app_fico_range_high",
            "sec_app_fico_range_low",
            "title",
            "url",
            "funded_amnt",
        ],
        axis=1,
    )

    df["emp_length"] = df["emp_length"].replace({"10+ years": "15", "< 1 year": "0"})
    df["emp_length"] = df["emp_length"].str.extract(r"([0-9]+)").astype("float")
    df = df.rename(columns={"emp_length": "emp_length_years"})

    df["home_ownership"] = df["home_ownership"].str.lower()
    df = df.rename(columns={"addr_state": "state"})

    sub_grade = (
        pd.Series(df["sub_grade"].dropna().unique())
        .sort_values(ascending=False)
        .tolist()
    )
    sub_grade_dict = {}
    grade_dict = {}
    num = 0
    grade_num = 7
    for grade in sub_grade:
        sub_grade_dict[grade] = num
        if (num == 5) | (num % 5 == 0):
            grade_dict[grade[0]] = grade_num
            grade_num -= 1
        num += 1

    df["sub_grade"] = df["sub_grade"].replace(sub_grade_dict)
    df["grade"] = df["grade"].replace(grade_dict)

    df.loc[:, ["earliest_cr_line", "issue_d", "sec_app_earliest_cr_line"]] = df.loc[
        :, ["earliest_cr_line", "issue_d", "sec_app_earliest_cr_line"]
    ].astype("datetime64[D]")
    df["mths_since_earliest_cr_line"] = (
        (df["issue_d"] - df["earliest_cr_line"]) / np.timedelta64(1, "M")
    ).round()
    df = df.drop("earliest_cr_line", axis=1)

    df = df.drop(
        df.columns[df.columns.str.contains("joint") | df.columns.str.contains("sec_")],
        axis=1,
    )

    top_value_percentage = (
        df.loc[:, df.columns[~(df.dtypes == "string")]].apply(
            lambda x: x.value_counts().max(), axis=0
        )
        * 100
        / df.shape[0]
    ).fillna(100)
    df = df.drop(top_value_percentage[top_value_percentage > 98].index, axis=1)

    return df


def columns_to_drop_due_multicoll(df):
    """
    Function that calculates which between two features with correlation value over 0.7 is higher correlated with the target column. Returns a list with all the columns that should be dropped to avoid multicollinearity.
    """
    columns_to_drop = []
    corr_df = pd.DataFrame(columns=["col1", "col2", "corr"])
    numeric_cols = df.columns[
        ((df.dtypes == "float") | (df.dtypes == "int"))
        & (~df.columns.isin(["sub_grade", "int_rate", "grade"]))
    ]
    for column in numeric_cols:
        if column in df.columns:
            col_corr = df.corrwith(df[column])
            multicoll_list = col_corr[
                (col_corr > 0.7) & (col_corr.index != column)
            ].index.tolist()
            if multicoll_list:
                for corr in multicoll_list:
                    if np.abs(df[corr].corr(df["sub_grade"])) < np.abs(
                        df[column].corr(df["sub_grade"])
                    ):
                        columns_to_drop.append(corr)
                    else:
                        columns_to_drop.append(column)
                        break
    return columns_to_drop


def get_best_corrs_missing_cols(df_input, months_columns, columns_to_drop):
    """
    A function that calculates the most correlated features for columns with missing data. Also calculates mean missing column values for each binned correlated column interval for imputation.

    Parameters:
        df_input(pd.DataFrame): All data
        months_columns(list): A list containing features that contain "number_of_months_since_X" data
        columns_to_drop(list): A list containing features to drop to avoid multicollinearity

    Returns:
        best_corrs_means_dict({str:pd.Series}): A dictionary containing series with missing_col mean values for each corr_col interval for each feature pair in max_corr_pairs.
        max_corr_pairs(pd.Series): A series containing missing_col and corr_col pairs.
    """
    df = df_input.copy()
    missing_values = df.columns[
        ((df.isna().sum() * 100 / df.shape[0]) > 1)
        & (~df.columns.isin(columns_to_drop))
        & (df.dtypes == "float")
        & (~df.columns.isin(months_columns))
    ]

    corr_df = pd.DataFrame()
    for column in missing_values:
        corr_df[column] = df.loc[:, df.columns[df.dtypes == "float"]].corrwith(
            df[column].astype("float")
        )
    corr_df[corr_df > 0.99] = np.nan
    corr_df[np.abs(corr_df) < 0.1] = np.nan

    max_corr_pairs = np.abs(corr_df).idxmax()
    max_corr_pairs = max_corr_pairs.dropna(axis=0)
    best_corrs_means_dict = {}

    for column in max_corr_pairs.index:
        df.loc[:, "temp_column"] = pd.cut(df[max_corr_pairs[column]], 10)
        best_corrs_means_dict[column] = (
            df.groupby("temp_column")[column]
            .median()
            .to_frame()
            .reset_index()
            .rename(columns={"temp_column": "bin_column", column: "medians"})
        )
    return best_corrs_means_dict, max_corr_pairs


def get_positive_negative_careers(df_input):
    """
    A function that calculates which carrer names in the "emp_title" column correlate positively and which negatively with target column.

    Returns:
        negative_corr_careers(list): List containing 20 careers that negatively correlate with target column
        positive_corr_careers(list): List containing 20 careers that positively correlate with target column
    """
    df = df_input.copy()
    df["emp_title"] = df["emp_title"].str.lower().str.replace("[^a-zA-Z0-9 ]", "")
    prof_df = (
        pd.Series(
            [
                y
                for x in df["emp_title"]
                .str.lower()
                .str.replace(r"[^a-zA-Z0-9 ]", "")
                .dropna()
                for y in x.split()
            ]
        )
        .value_counts()
        .to_frame("total")
    )

    corr_words = pd.Series(name="correlation")
    for word in prof_df.head(400).index.tolist():
        words_one_hot = df.loc[:, ["emp_title", "sub_grade"]]

        words_one_hot.loc[
            words_one_hot["emp_title"]
            .str.contains(rf"(?:^|\W){word}(?:$|\W)")
            .fillna(False),
            "word",
        ] = 1
        words_one_hot.loc[:, "word"] = (
            words_one_hot.loc[:, "word"].fillna(0).astype(int)
        )
        corr_words[word] = words_one_hot["word"].corr(
            words_one_hot["sub_grade"].astype("float")
        )

    negative_corr_careers = corr_words.sort_values().head(20).index.tolist()
    positive_corr_careers = (
        corr_words.sort_values(ascending=False).head(20).index.tolist()
    )

    return negative_corr_careers, positive_corr_careers


###

###


def objective_booster_class(trial, X: pd.DataFrame, y: pd.Series):

    scalers = trial.suggest_categorical("scaler", ["MinMax", "Standard", "Robust"])
    if scalers == "MinMax":
        scaler = MinMaxScaler()
    elif scalers == "Standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    state = trial.suggest_categorical("state", [True, False])

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.7)
    n_estimators = trial.suggest_int("n_estimators", 10, 300)
    max_depth = trial.suggest_int("max_depth", 2, 6, log=True)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 1e-10, 0.1)

    Preprocessor = Pipeline(
        [
            ("extract_words", ExtractWords(relevant_words)),
            (
                "zip_to_coords",
                ZipStateToCoordinates(state_coord_means, bin_coord_means, state=state),
            ),
            ("unknown_to_nan", UnknownToNan(missing_dict)),
            (
                "add_missing_column",
                AddMissingColumn(list(missing_dict.keys()), one_column=True),
            ),
            ("fill_missing_data", FillMissingData()),
            ("transform_to_log", TransformToLog(to_log_list)),
        ]
    )

    full_pipe = Pipeline(
        [
            ("preprocessor", Preprocessor),
            ("scaler", scaler),
            (
                "model",
                xg.XGBClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_impurity_decrease=min_impurity_decrease,
                    tree_method="gpu_hist",
                ),
            ),
        ]
    )

    models[trial.number] = full_pipe
    score = cross_val_score(
        full_pipe,
        X,
        y,
        n_jobs=-1,
        cv=5,
        scoring="roc_auc",
        verbose=30,
    )
    average_precision = score.mean()
    return average_precision


def objective_booster(trial, X: pd.DataFrame, y: pd.Series):

    scalers = trial.suggest_categorical("scaler", ["MinMax", "Standard", "Robust"])
    if scalers == "MinMax":
        scaler = MinMaxScaler()
    elif scalers == "Standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    state = trial.suggest_categorical("state", [True, False])
    binarize = trial.suggest_categorical("binarize", ["none", "some", "all"])

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.7)
    n_estimators = trial.suggest_int("n_estimators", 10, 300)
    max_depth = trial.suggest_int("max_depth", 2, 6, log=True)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 1e-10, 0.1)

    one_hot_columns = [
        "term",
        "verification_status",
        "initial_list_status",
        "disbursement_method",
        "purpose",
        "home_ownership",
        "application_type",
    ]

    OneHotPipeline = ColumnTransformer(
        [
            (
                "one_hot_columns",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                one_hot_columns,
            )
        ],
        remainder="passthrough",
    )

    Preprocessor = Pipeline(
        [
            (
                "zip_to_coords",
                ZipStateToCoordinates(state_coord_means, bin_coord_means, state=state),
            ),
            (
                "careers",
                FilterCareers(
                    negative_corr_careers, positive_corr_careers, "emp_title"
                ),
            ),
            (
                "filter_values_columns",
                SmallValuesToOther(["purpose", "home_ownership"]),
            ),
            (
                "fill_missing_data",
                FillMissingDataAccepted(
                    best_corrs_means_dict,
                    max_corr_pairs,
                    months_columns,
                    columns_to_drop,
                    binarize=binarize,
                ),
            ),
            ("log_transform", TransformToLog(skewed_columns)),
            ("one_hot_encode", OneHotPipeline),
        ]
    )

    full_pipe = Pipeline(
        [
            ("preprocessor", Preprocessor),
            ("scaler", scaler),
            (
                "model",
                xg.XGBRegressor(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_impurity_decrease=min_impurity_decrease,
                    tree_method="gpu_hist",
                ),
            ),
        ]
    )

    models[trial.number] = full_pipe
    score = cross_val_score(
        full_pipe,
        X,
        y,
        n_jobs=-1,
        cv=5,
        scoring="r2",
        verbose=30,
    )
    print(score)
    r2 = score.mean()
    return r2


###

###


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