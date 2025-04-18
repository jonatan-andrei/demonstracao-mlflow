import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from mlflow.tracking import MlflowClient

class PreProcessamento:
    def executar(self):
        df = pd.read_csv("Anonymize_Loan_Default_data.csv", encoding="ISO-8859-1")
        print(df.head())

        print("O conjunto de dados possui {} linhas e {} colunas".format(df.shape[0], df.shape[1]))

        # Verificando quantidade de valores nulos nas colunas
        df.isnull().sum()

        # Removendo linhas com valores nulos em colunas muito importantes
        df = df.dropna(subset=["loan_amnt", "funded_amnt", "installment", "annual_inc", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"])

        # Removendo empréstimos com valor zero
        df.drop(df[df['funded_amnt'] == 0].index, inplace=True)

        # Optamos por remover as colunas que não agregariam valor na nossa análise (id, data do pagamento, se o empréstimo foi concedido através de investidores)
        # Também foram removidas colunas que traziam informações já disponíveis em outra coluna, como por exemplo o Zip code, que está melhor representado em addr_state
        # Também removemos colunas que traziam informações sobre o status final do empréstimo (como valor final pago)
        df = df.drop(columns=['Unnamed: 0', 'id','member_id', 'funded_amnt_inv', 'loan_status', 'zip_code', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'issue_d'])

        print("O conjunto de dados possui {} linhas e {} colunas".format(df.shape[0], df.shape[1]))

        # Verificando quantidade de valores nulos nas colunas
        df.isnull().sum()

        df["emp_length"].value_counts()

        # Transformando em valores numéricos
        df["emp_length"] = df["emp_length"].str.extract("(\d+)").astype(float)

        df["emp_length"].value_counts()

        # Adicionando zero para valores nulos
        df["emp_length"] = df["emp_length"].fillna(0)

        mesmos_valores = (df["loan_amnt"] == df["funded_amnt"]).sum()

        print(f"Número de linhas onde loan_amnt e funded_amnt são iguais: {mesmos_valores}")

        # Percebemos que seria mais interessante para o modelo se, em vez de ter duas colunas quase iguais, ter uma coluna booleana que indicasse se o valor total foi concedido ou não
        df["total_amount_granted"] = (df["loan_amnt"] == df["funded_amnt"]).astype(int)

        df = df.drop(columns=['loan_amnt'])

        df["total_amount_granted"].value_counts()

        # Transformando em valor numérico
        df["term"] = df["term"].str.extract("(\d+)").astype(float)

        df["term"].value_counts()

        df["purpose"].value_counts()

        df["addr_state"].value_counts()

        df["mths_since_last_delinq"].max()

        # Para não perder a informação de que o cliente nunca ficou inadimplente, optamos por transformar em uma variável categórica
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

        df["years_since_last_delinq"].value_counts()

        df = df.drop(columns=['mths_since_last_delinq'])

        # Decidimos transformar a coluna earliest_cr_line em uma coluna que indique há quantos meses o cliente tem histórico de crédito
        # Como as datas são mais antigas, optamos por usar como base o mês mais recente
        # O código abaixo foi gerado pelo Google Gemini e adaptado
        df['earliest_cr_line_temp'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')

        data_recente = df['earliest_cr_line_temp'].max()

        df.sort_values(by='earliest_cr_line_temp', ascending=False)

        # Nesse momento notamos que existem muitos erros no cadastro dessa coluna, tendo registros de 2068. Optamos então por removê-la
        df = df.drop(columns=['earliest_cr_line', 'earliest_cr_line_temp'])

        print("O conjunto de dados possui {} linhas e {} colunas".format(df.shape[0], df.shape[1]))

        # Transformando a coluna revol_util em numérica
        df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)

        # Vendo o atual estado do dataset
        df.isnull().sum()

        # Verificando a distribuição dos dados em relação ao atributo alvo
        df["repay_fail"].value_counts()

        # Infelizmente, muitos modelos tem problemas com variáveis categórias. Então criamos uma cópia do dataset e faremos modificações nas colunas
        # para adaptá-las ao modelo Random Forest
        df_rf = df.copy(deep=True)

        df_rf["home_ownership"] = (df_rf["home_ownership"] == 'OWN').astype(int)

        df_rf["home_ownership"].value_counts()

        df_rf["verification_status"] = ((df_rf["verification_status"] == 'Verified') | (df_rf["verification_status"] == 'Source Verified')).astype(int)

        df_rf["verification_status"].value_counts()

        # Aplicando one hot encoding na coluna years_since_last_delinq
        df_rf = pd.get_dummies(df_rf, columns=['years_since_last_delinq'])

        # As colunas purpose e addr_state teriam muitos valores possíveis, o que atrapalharia uma técnica como one hot encoding.
        # Optamos por removê-las nesse experimento.
        df_rf = df_rf.drop(columns=['purpose', 'addr_state'])

        print(df_rf.dtypes)

        return df_rf
    
class ExperimentacaoModelos:
    def executar(self, df_rf):
        # Separando o dataset em treino e teste
        X = df_rf.drop('repay_fail', axis=1)
        y = df_rf['repay_fail']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)

        # Importando o módulo de tracking do MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        # Criando um experimento
        mlflow.set_experiment("ecd15_mlops_trabalho_final")

        # Random Forest Classifier
        with mlflow.start_run(run_name = "experimento_RandomForestClassifier") as run:
            mlflow.log_param("model_type", "RandomForestClassifier")

            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=18)
            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_test)
            signature = infer_signature(X_train, y_pred)
            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(rf_classifier, 
                                     "random_forest_classifier_model", 
                                     signature=signature, 
                                     input_example=X_train, 
                                     registered_model_name="RandomForestClassifier")
            
            print(f"Modelo RandomForestClassifier registrado no MLflow! Run ID: {run.info.run_id}")

        # Os primeiros resultados foram ruins em relação a classe 1, decidimos aumentar as chances do modelo prever a classe 1
        with mlflow.start_run(run_name = "experimento_RandomForestClassifier_ajustado") as run:
            mlflow.log_param("model_type", "RandomForestClassifierAjustado")

            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=18)
            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict_proba(X_test)[:, 1]
            threshold = 0.2
            y_pred_adjusted = (y_pred >= threshold).astype(int)
            signature = infer_signature(X_train, y_pred_adjusted)
            accuracy = accuracy_score(y_test, y_pred_adjusted)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(rf_classifier, 
                                     "random_forest_classifier_model_ajustado",
                                     signature=signature,
                                     input_example=X_train,
                                     registered_model_name="RandomForestClassifierAjustado")
            
            print(f"Modelo RandomForestClassifierAjustado registrado no MLflow! Run ID: {run.info.run_id}")

        # Decidimos então treinar uma árvore de decisão
        with mlflow.start_run(run_name = "experimento_DecisionTreeClassifier") as run:
            mlflow.log_param("model_type", "DecisionTreeClassifier")

            tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=18)
            tree_classifier.fit(X_train, y_train)

            y_pred = tree_classifier.predict(X_test)

            signature = infer_signature(X_train, y_pred)

            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(tree_classifier, 
                                     "decision_tree_classifier_model",
                                     signature=signature,
                                     input_example=X_train,
                                     registered_model_name="DecisionTreeClassifier")

            print(f"Modelo DecisionTreeClassifier registrado no MLflow! Run ID: {run.info.run_id}")

        #KNN
        with mlflow.start_run(run_name = "experimento_KNN") as run:
            mlflow.log_param("model_type", "KNN")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 🔹 Criar e treinar o modelo KNN com K=7
            knn = KNeighborsClassifier(n_neighbors=7)
            knn.fit(X_train_scaled, y_train)

            y_pred = knn.predict(X_test_scaled)

            signature = infer_signature(X_train_scaled, y_pred)

            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(knn, 
                                     "KNN_model",
                                     signature=signature,
                                     input_example=X_train_scaled,
                                     registered_model_name="KNN")

            print(f"Modelo KNN registrado no MLflow! Run ID: {run.info.run_id}")

        # XGBoost
        with mlflow.start_run(run_name = "experimento_XGBoost") as run:
            mlflow.log_param("model_type", "XGBoost")

            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',  # Classificação binária
                eval_metric='logloss',  # Métrica de erro
                use_label_encoder=False,
                n_estimators=100,  # Número de árvores
                learning_rate=0.1,  # Taxa de aprendizado
                max_depth=3,  # Profundidade máxima da árvore
                random_state=45
            )
            xgb_model.fit(X_train, y_train)

            y_pred = xgb_model.predict(X_test)

            signature = infer_signature(X_train, y_pred)

            accuracy = accuracy_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(xgb_model, 
                                     "XGBoost_model",
                                     signature=signature,
                                     input_example=X_train,
                                     registered_model_name="XGBoost")

            print(f"Modelo XGBoost registrado no MLflow! Run ID: {run.info.run_id}")

class PromocaoDeModelos:
    def executar(self):
        # Definir os limites de accuracy para Staging e Production
        staging_threshold = 0.7  # Apenas modelos acima desta accuracy vão para Staging

        best_model_name = None  # Para armazenar o modelo Champion
        best_model_version = 0  # Para armazenar a versão do modelo Champion
        best_accuracy_score = 0  # Para rastrear a melhor accuracy

        client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

        # Listar todos os modelos registrados
        registered_models = client.search_registered_models()

        # No nosso caso testamos diferentes modelos em runs diferentes
        # Entao, iteramos sobre os modelos registrados
        for model in registered_models:
            version = model.latest_versions[-1].version

            # Obter run_id do modelo
            run_id = model.latest_versions[-1].run_id

            # Obter métricas
            metrics = client.get_run(run_id).data.metrics

            # Obter a accuracy do modelo
            if "accuracy" in metrics:
                accuracy = metrics["accuracy"]

            # Adicionar modelos qualificados para Staging
            if float(accuracy) > staging_threshold:
                client.transition_model_version_stage(
                    name=model.name,
                    version=version,
                    stage="Staging"
                )
                print(f"Modelo {model.name} versão {version} com accuracy {accuracy} movido para Staging.")
            else:
                print(f"Modelo {model.name} versão {version} com accuracy {accuracy} não atende aos critérios para Staging.")

            # Verificar se é o melhor modelo
            if float(accuracy) > best_accuracy_score:
                best_accuracy_score = float(accuracy)
                best_model_name = model.name
                best_model_version = version


        # Mover o melhor modelo para Production
        if best_model_name:
            client.transition_model_version_stage(
                name=best_model_name,
                version=best_model_version, 
                stage="Production"
            )
            print(f"Modelo {best_model_name} versão {best_model_version} movido para Production com accuracy {best_accuracy_score}.")
        else:
            print("Nenhum modelo atende aos critérios para ser movido para Production.")

def main():
    df_rf = PreProcessamento().executar()
    ExperimentacaoModelos().executar(df_rf)
    PromocaoDeModelos().executar()

    print("\n")
    print("!! Finalizado workflow principal de pre processamento, experimentacao e promocao de modelos. !!")
    print("!! Proximos passos: !!")
    print("!! 1. Executar flask_api.py para servir o modelo em produção. !!")
    print("!! 2. Realizar inferencias executando predict.py !!")
    print("\n")

if __name__ == "__main__":
    main()