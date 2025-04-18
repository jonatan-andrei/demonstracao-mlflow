# Trabalho disciplina MLOps

## Integrantes:
- Breno Lazari da Silva
- Jonatan Andrei Haas
- Ronald Rodrigues Flôres


## Pré-requisitos:
Instale o Python 3.10: https://www.python.org/downloads/release/python-31011/ 

### Instalar bibliotecas:
```
pip install mlflow pandas scikit-learn matplotlib xgboost
```

### Instalar Evidently:
```
pip install evidently==0.6.7
```

## Passo 1 - Faça download do código desse repositório
```
git clone https://github.com/jonatan-andrei/demonstracao-mlflow
```

## Passo 2 - Iniciar o MLFlow:
(precisa ser iniciado na mesma pasta onde está o código)
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Passo 3 - Executar main_workflow.py
(deve ser executado na pasta em que o arquivo está)
```
python main_workflow.py
```
### Nesse momento ocorrerá:
- Carregamento do dataset
- Pré processamento dos dados
- Avaliação de modelos
- Rastreamento e registro no MLflow
- Promoção de modelos para Staging e Production no MLflow

## Passo 4 - Executar flask_api.py para servir o modelo de "produção" numa API (porta 8000)
```
python flask_api.py
```

## Passo 5 - Executar predict.py para chamar API com um payload de exemplo para fazer a predição
```
python predict.py
```

## Passo 6 - Executar monitor_drift.py para avaliar drift do modelo e simular retreinamento
```
python monitor_drift.py
```
### Nesse momento ocorrerá:
- Carregamento do dataset e pré-processamento dos dados originais
- Envio dos dados para a API
- Avaliação de drift usando o Evidently (que não ocorrerá nesse caso, porque são os mesmos dados)
- Alteração em duas colunas do dataset original para simular drift
- Envio dos novos dados para a API
- Detecção de drift nas duas colunas que modificamos
- Retreinamento do modelo, acionando automaticamente o código de main_workflow
