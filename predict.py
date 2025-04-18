import requests

def predict():
    # Defina as colunas esperadas pelo modelo
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

    # Exemplo de instância de entrada para previsão
    # Essa instancia foi copiada da variavel "linha_nova" do arquivo main_workflow.py 
    # Apenas para fins de teste da API
    payload_sample  = {
        "funded_amnt": 9600.0,
        "term": 36.0,
        "int_rate": 11.34,
        "installment": 315.84,
        "emp_length": 7.0,
        "home_ownership": 0,
        "annual_inc": 113000.0,
        "verification_status": 0,
        "dti": 18.54,
        "delinq_2yrs": 0.0,
        "inq_last_6mths": 6.0,
        "open_acc": 32.0,
        "pub_rec": 0.0,
        "revol_bal": 45303.0,
        "revol_util": 46.0,
        "total_acc": 63.0,
        "total_amount_granted": 1,
        "years_since_last_delinq_10 years": True,
        "years_since_last_delinq_2 years": False,
        "years_since_last_delinq_5 years": False,
        "years_since_last_delinq_never": False
    }

    url = "http://127.0.0.1:8000/predict"
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=payload_sample)
    predictions = response.json()
    print(predictions)
    predictions = predictions.get("prediction")
    return predictions

if __name__ == "__main__":
    predict()