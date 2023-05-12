import joblib
import numpy as np
from AUS_Functions_transformers_only import *
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import json

app = Flask(__name__, template_folder="templateFiles", static_folder="staticFiles")

attributes_accepted = ["amount_requested", "loan_title", "risk_score",
    "debt_to_income_ratio", "zip_code", "state", "employment_length_years"]

common_attributes_dict = {
    "amount_requested":"loan_amnt",
    "employment_length_years":"emp_length_years",
    "risk_score":"risk_score",
    "debt_to_income_ratio":"dti",
    "zip_code":"zip_code",
    "state":"state"
}

attributes_grade = ["term", "emp_title", "home_ownership",
    "annual_inc", "verification_status", "purpose", "total_bc_limit",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "pub_rec", "initial_list_status",
    "application_type", "tot_coll_amt", "open_act_il", "open_il_12m",
    "mths_since_rcnt_il", "total_bal_il", "il_util", "max_bal_bc",
    "all_util", "inq_fi", "total_cu_tl", "inq_last_12m", "revol_util",
    "acc_open_past_24mths", "bc_open_to_buy", "bc_util",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_rev_tl",
    "num_bc_tl", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq", "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "disbursement_method", "mths_since_earliest_cr_line"]


grade_unnecesarry_columns = ["mths_since_last_major_derog",
"mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq", "index"]

models_list = [
    "trained_model_int_rate",
    "trained_model_grade",
    "trained_model_accepted",
    "trained_model_1_sub_grades",
    "trained_model_2_sub_grades",
    "trained_model_3_sub_grades",
    "trained_model_4_sub_grades",
    "trained_model_5_sub_grades",
    "trained_model_6_sub_grades",
    "trained_model_7_sub_grades"
]
models_dict = {}
numerical_features_accepted = [
    "amount_requested",
    "risk_score",
    "debt_to_income_ratio",
    "employment_length_years"
]

numerical_features_grade = [
    "loan_amnt",
    "emp_length_years",
    "risk_score",
    "dti",
    "annual_inc",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "pub_rec",
    "tot_coll_amt",
    "open_act_il",
    "open_il_12m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "max_bal_bc",
    "all_util",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",
    "bc_open_to_buy",
    "bc_util",
    "revol_util",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "num_accts_ever_120_pd",
    "num_actv_rev_tl",
    "num_bc_tl",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "mths_since_earliest_cr_line",
    "total_bc_limit"
]

grade_dict = {1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F", 7:"G"}

attributes_accepted_df = pd.DataFrame()


def load_model():
    global models_dict
    for model in models_list:
        models_dict[model] = joblib.load(f"{model}.pkl")


def string_to_numeral(df: pd.DataFrame, numerical_features: list):
    for feature in numerical_features:
        if feature in df.columns:
            print(df[feature])
            df[feature] = int(df[feature])
    return df

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/rejected")
def rejected():
    return render_template("rejected.html")

@app.route("/accepted")
def accepted():
    return render_template("accepted.html")

@app.route("/grade")
def grade():
    return render_template("grade.html")




@app.route("/predict_step_2", methods=["GET", "POST"])
def get_prediction_2():
    if request.method == "POST":
        attributes_list = []
        for elem in attributes_grade:
            attributes_list.append(request.form.get(elem))
        df = pd.DataFrame(columns=attributes_grade)
        df.loc[1] = attributes_list
        for key in common_attributes_dict:
            df.loc[1, common_attributes_dict[key]] = attributes_accepted_df[key].tolist()[0]
        df = string_to_numeral(df, numerical_features_grade)
        for elem in grade_unnecesarry_columns:
            df.loc[1, elem] = 1

        prediction_grade = models_dict["trained_model_grade"].predict(df)
        prediction_grade_rounded = np.round(prediction_grade).astype(int)
        prediction_grade_rounded[prediction_grade_rounded < 1] = 1
        prediction_grade_rounded[prediction_grade_rounded > 7] = 7

        prediction_sub_grade = models_dict[f"trained_model_{prediction_grade_rounded[0]}_sub_grades"].predict(df)
        prediction_sub_grade_rounded = np.round(prediction_sub_grade).astype(int)
        prediction_sub_grade_rounded[prediction_sub_grade_rounded < 1] = 1
        prediction_sub_grade_rounded[prediction_sub_grade_rounded > 5] = 5

        df.loc[:,"grade"] = prediction_grade
        prediction_int_rate = models_dict["trained_model_int_rate"].predict(df)[0]

        prediction_grade = pd.Series(prediction_grade_rounded).replace(grade_dict)[0]
        prediction_sub_grade = prediction_grade + str(prediction_sub_grade_rounded[0])[0]

        return redirect(
            url_for(
                "grade",
                _anchor="anchor",
                prediction_grade = prediction_grade,
                prediction_sub_grade = prediction_sub_grade,
                prediction_int_rate = prediction_int_rate
            )
        )
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def get_prediction():
    global attributes_accepted_df
    if request.method == "POST":
        attributes_list = []
        for elem in attributes_accepted:
            attributes_list.append(request.form.get(elem))
        df = pd.DataFrame(columns=attributes_accepted)
        df.loc[1] = attributes_list
        df["zip_code"] = df["zip_code"]+"xx"
        attributes_accepted_df = df.copy()
        df = string_to_numeral(df, numerical_features_accepted)
        prediction = models_dict["trained_model_accepted"].predict(df)
        if prediction[0] == 0:
            next_url = "rejected"
        else:
            next_url = "accepted"
        return redirect(
            url_for(
                next_url,
                _anchor="anchor"
            )
        )
    return render_template("index.html")


@app.route("/predict_step_2_api", methods=["GET", "POST"])
def get_prediction_2_api():
    if request.method == "POST":
        attributes_json = request.get_json()
        print(attributes_json)
        df = pd.DataFrame.from_dict(request.get_json(), orient="index").T
        df["zip_code"] = df["zip_code"]+"xx"
        df = string_to_numeral(df, numerical_features_grade)

        df_accepted = df.loc[:, attributes_accepted]
        prediction = models_dict["trained_model_accepted"].predict(df)
        if (prediction[0] == 0):
            return {"accepted":str(prediction[0])}
        else:
            for elem in grade_unnecesarry_columns:
                df.loc[1, elem] = 1

            prediction_grade = models_dict["trained_model_grade"].predict(df)
            prediction_grade_rounded = np.round(prediction_grade).astype(int)
            prediction_grade_rounded[prediction_grade_rounded < 1] = 1
            prediction_grade_rounded[prediction_grade_rounded > 7] = 7

            prediction_sub_grade = models_dict[f"trained_model_{prediction_grade_rounded[0]}_sub_grades"].predict(df)
            prediction_sub_grade_rounded = np.round(prediction_sub_grade).astype(int)
            prediction_sub_grade_rounded[prediction_sub_grade_rounded < 1] = 1
            prediction_sub_grade_rounded[prediction_sub_grade_rounded > 5] = 5

            df.loc[:,"grade"] = prediction_grade
            prediction_int_rate = models_dict["trained_model_int_rate"].predict(df)[0]

            prediction_grade = pd.Series(prediction_grade_rounded).replace(grade_dict)[0]
            prediction_sub_grade = prediction_grade + str(prediction_sub_grade_rounded[0])[0]
            return {"accepted":str(prediction[0]),
                    "grade":str(prediction_grade),
                    "sub_grade":str(prediction_sub_grade),
                    "int_rate":str(prediction_int_rate)}


@app.route("/predict_api", methods=["GET", "POST"])
def get_prediction_api():
    if request.method == "POST":
        attributes_json = request.get_json()
        print(attributes_json)
        df = pd.DataFrame.from_dict(request.get_json(), orient="index").T
        df["zip_code"] = df["zip_code"]+"xx"
        df = string_to_numeral(df, numerical_features_accepted)
        prediction = models_dict["trained_model_accepted"].predict(df)
        return {"prediction":str(prediction[0])}

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
