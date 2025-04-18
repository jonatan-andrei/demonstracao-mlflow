# Monitoramento e Re-treinamento
# ○ Implementação de monitoramento de drift de dados com Evidently AI.
# ○ Definição de uma estratégia para re-treinamento automático do modelo.

import pandas as pd
import numpy as np
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder
import os
import json

def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5:
        print("Drift detectado no Dataset")
        os.system("python main_workflow.py")
    else:
        if num_columns_drift >= 2:
            print(f"Drift detectado em {num_columns_drift} colunas! Treinando novo modelo...")
            os.system("python main_workflow.py")
        else:
            print("Modelo ainda está bom, sem necessidade de re-treinamento.")
            print("Nenhum drift detectado nas colunas e no dataset")


def load_new_data():
    df = pd.read_csv("Anonymize_Loan_Default_data.csv", encoding="ISO-8859-1")
    df = df.sample(1000)
    X, y = preprocess_data(df)
    return X, y

def simulate_drift(df_examples):
    new_data = df_examples.copy()
    # Mudamos algumas colunas para simular mudanças nos padrões dos dados
    new_data["funded_amnt"]  *= 1.5
    new_data["term"] *= 2
    
    print("Criado dataset artificialmente alterado para simular drift.")
    return new_data

def preprocess_data(df):
    # Pré-processamento já explicado no arquivo main_workflow
    df = pd.read_csv("Anonymize_Loan_Default_data.csv", encoding="ISO-8859-1")
    df = df.dropna(subset=["loan_amnt", "funded_amnt", "installment", "annual_inc", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"])
    df.drop(df[df['funded_amnt'] == 0].index, inplace=True)
    df = df.drop(columns=['Unnamed: 0', 'id','member_id', 'funded_amnt_inv', 'loan_status', 'zip_code', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'issue_d'])
    df["emp_length"] = df["emp_length"].str.extract("(\d+)").astype(float)
    df["emp_length"] = df["emp_length"].fillna(0)
    df["total_amount_granted"] = (df["loan_amnt"] == df["funded_amnt"]).astype(int)
    df = df.drop(columns=['loan_amnt'])
    df["term"] = df["term"].str.extract("(\d+)").astype(float)
    def categorizar(tempo):
        if pd.isnull(tempo):
            return "never"
        elif tempo <= 24:
            return "2 years"
        elif tempo <= 60:
            return "5 years"
        else:
            return "10 years"
    df["years_since_last_delinq"] = df["mths_since_last_delinq"].apply(categorizar)
    df = df.drop(columns=['mths_since_last_delinq'])
    df['earliest_cr_line_temp'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
    data_recente = df['earliest_cr_line_temp'].max()
    df.sort_values(by='earliest_cr_line_temp', ascending=False)
    df = df.drop(columns=['earliest_cr_line', 'earliest_cr_line_temp'])
    df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)
    df_rf = df.copy(deep=True)
    df_rf["home_ownership"] = (df_rf["home_ownership"] == 'OWN').astype(int)
    df_rf["verification_status"] = ((df_rf["verification_status"] == 'Verified') | (df_rf["verification_status"] == 'Source Verified')).astype(int)
    # Aplicando one hot encoding na coluna years_since_last_delinq
    df_rf = pd.get_dummies(df_rf, columns=['years_since_last_delinq'])
    df_rf = df_rf.drop(columns=['purpose', 'addr_state'])
    X = df_rf.drop(columns=["repay_fail"])
    y = df_rf["repay_fail"]
    return X, y.astype(int)

def get_predictions(data):
    print(data.head())

    # Definindo as colunas esperadas pelo modelo
    columns = [
        "funded_amnt",
        "term",
        "int_rate",
        "installment",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "total_amount_granted",
        "years_since_last_delinq_10 years",
        "years_since_last_delinq_2 years",
        "years_since_last_delinq_5 years",
        "years_since_last_delinq_never"
    ]
    
    # lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in columns}
        instances.append(instance)


    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}
    
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Erro {response.status_code} na requisição:")
        print(response.text)
        return None

    try:
        predictions = response.json().get("predictions")
        print(predictions)
        return predictions
    except ValueError as e:
        print("Erro ao decodificar JSON:", e)
        print("Resposta bruta:", response.text)
        return None
    
# Avaliando degradação do modelo
def evaluate_model(df, y, new_data):
    if new_data is None:
        print("Avaliando modelo com dados originais")
        df["prediction"] = get_predictions(df)
        df["prediction"] = df["prediction"].astype(int)
        print(df["prediction"].unique())
        df["target"] = y
        print(df["target"].unique())
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=df, current_data=df)
        report.save_html("monitoring_report_df.html")
        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        print(f"Coluns drift: {drift_by_columns}")
        return drift_score, drift_by_columns
    else:
        print("Avaliando modelo com dados artificiais")
        new_data["prediction"] = get_predictions(new_data)
        new_data["prediction"] = new_data["prediction"].astype(int)
        print(new_data["prediction"].unique())
        new_data["target"] = y
        print(new_data["target"].unique())
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=df, current_data=new_data)
        report.save_html("monitoring_report_df_new_data.html")
        report_dict = report.as_dict()
        drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
        print(f"Score de drift: {drift_score}")
        drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
        print(f"Coluns drift: {drift_by_columns}")
        return drift_score, drift_by_columns


def main():
    df_examples, y = load_new_data()
    drift_score, drift_by_columns = evaluate_model(df_examples, y, None)
    new_data = simulate_drift(df_examples)
    drift_score, drift_by_columns = evaluate_model(df_examples, y, new_data)
    check_for_drift(drift_score, drift_by_columns)

if __name__ == "__main__":
    main()
