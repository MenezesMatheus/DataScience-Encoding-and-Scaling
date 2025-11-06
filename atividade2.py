from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce

def prepareDataSet():
    # Carregar o dataset "adult" do OpenML
    data = fetch_openml(data_id = 1590, as_frame = True, parser = "auto")
    df = data.frame

    # Lidas com valores ausentes
    categorical_columns = df.select_dtypes(include = ["category", "object"]).columns

    df[categorical_columns] = df[categorical_columns].replace("?", np.nan)

    df.dropna(inplace = True)

    # Separar as features e o alvo
    x = df.drop(columns = "class")
    y_temporary = df["class"]

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y_temporary)

    numeric_features = x.select_dtypes(include = np.number).columns.tolist()
    categorical_features = x.select_dtypes(include = "category").columns.tolist()

    print(f"Features numéricas: {numeric_features}")
    print(f"Features categóricas: {categorical_features}")
    print(f"\nDataset preparado com {len(x)} amostras.")

    return x, y, numeric_features, categorical_features

def splitDataSet(x, y):
    # Separa os dados em conjuntos de treino e teste, proporção 70/30
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, stratify = y)

    print(f"Amostras de treino: {len(x_train)}, Amostras de teste: {len(x_test)}")

    return x_train, x_test, y_train, y_test

def KNNAexperiments(x_train, x_test, y_train, y_test, numeric_features, categorical_features):
    results = []

    # Definindo Encoders
    encoders = {
        "One-Hot": OneHotEncoder(handle_unknown = "ignore", drop = None, sparse_output = False),
        "Dummy": OneHotEncoder(handle_unknown = "ignore", drop = "first", sparse_output = False),
        "Effect": ce.SumEncoder(handle_unknown = "ignore")
    }

    # Definindo Scalers
    scalers = {
        "Standard": StandardScaler(),
        "Min-Max": MinMaxScaler(),
        "Max-Abs": MaxAbsScaler(),
        "Robust": RobustScaler(),
        "Quantile": QuantileTransformer(output_distribution = "uniform"),
        "Quantile-Normal": QuantileTransformer(output_distribution = "normal"),
        "No Scaling": "passthrough"
    }

    for encoder_name, encoder in encoders.items():
        for scaler_name, scaler in scalers.items():
            print(f"Executando KNN com {encoder_name} e {scaler_name}...")

            preprocessor = ColumnTransformer(transformers = [("num", scaler, numeric_features), ("cat", encoder, categorical_features)], remainder = "passthrough")

            # Criando o pipeline
            model_pipeline = Pipeline(steps = [("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors = 5))])

            # Treinando o modelo
            model_pipeline.fit(x_train, y_train)

            # Fazendo previsões
            y_pred = model_pipeline.predict(x_test) 
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average = "weighted")

            # Armazenando os resultados
            results.append({
                "Encoder": encoder_name,
                "Scaler": scaler_name,
                "Accuracy": accuracy,
                "F1-Score": f1
            })

    results_df = pd.DataFrame(results)

    print(f"\nResultados dos experimentos KNN:\n{results_df}")

    return results_df

def analyze(results_df): # Função para analisar e exibir os resultados
    print(f"Tabela de Resultados: Acurácia\n")

    accuracy_table = results_df.pivot(index = "Encoder", columns = "Scaler", values = "Accuracy")

    print(accuracy_table.to_markdown(floatfmt = ".2f"))

    print(f"Tabela de Resultados: F1-Score\n")

    f1_table = results_df.pivot(index = "Encoder", columns = "Scaler", values = "F1-Score")

    print(f1_table.to_markdown(floatfmt = ".2f"))

def plotAdjustments(df, metric_name, title, palette):
    encoders = df["Encoder"].unique()
    scalers = df["Scaler"].unique()

    number_of_encoders = len(encoders)
    number_of_scalers = len(scalers)

    x_positions = np.arange(number_of_encoders)

    width = 0.8
    width_per_bar = width / number_of_scalers

    position_adjustment = x_positions - (width - width_per_bar) / 2

    colors = plt.get_cmap(palette, number_of_scalers)

    figure, ax = plt.subplots(figsize = (17, 6))

    for i, scaler in enumerate(scalers):
        bar_positions = position_adjustment + (i * width_per_bar)

        values = df[df["Scaler"] == scaler][metric_name]

        ax.bar(bar_positions, values, width = width_per_bar, label = scaler, color = colors(i))

    ax.set_title(title, fontsize = 16)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Encoding Technique')  
    ax.set_xticks(x_positions)
    ax.set_xticklabels(encoders)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis ='y', linestyle = '--', alpha = 0.7)
    
    min_value = df[metric_name].min() * 0.98
    max_value = df[metric_name].max() * 1.02
    ax.set_ylim(max(0.7, min_value), min(1.0, max_value))
    
    plt.show()

def plot(results_df): # Função para plotar os resultados
    print(f"Plotando os resultados\n")

    plotAdjustments(results_df, "Accuracy", "Acurácia dos Experimentos KNN", "viridis") # Plotando os gráficos de Acurácia

    plotAdjustments(results_df, "F1-Score", "F1-Score dos Experimentos KNN", "plasma") # Plotando os gráficos de F1-Score

x, y, numeric_features, categorical_features = prepareDataSet()

print("dataset pronto")

x_train, x_test, y_train, y_test = splitDataSet(x, y)

print("dataset dividido")

results_df = KNNAexperiments(x_train, x_test, y_train, y_test, numeric_features, categorical_features)

analyze(results_df)

plot(results_df)