
import os
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import pickle

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

sns.set_style('darkgrid')

#  Criar experimento no mlflow
mlflow.set_tracking_uri(
    'mysql+pymysql://root:1234@localhost:3306/mlflow_tracking_database')
mlflow.set_experiment(experiment_name='Water_Classification')
tags = {
    'Project': 'PI-PUC',
    'Team': 'Data Science',
    'Dataset': 'Water Quality'
}


# Função que retorna a matriz de confusão do modelo
def plot_confusion_matrix(y_test, y_pred):
    fig = plt.figure()
    ax = plt.subplot()
    sns.heatmap(
        confusion_matrix(
            y_test,
            y_pred),
        annot=True,
        fmt='d',
        cbar=False,
        ax=ax)

    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['not potable', 'potable'])
    ax.yaxis.set_ticklabels(['not potable', 'potable'])
    plt.close()
    return fig


# Função que retorna as métricas de avaliação do modelo
def eval_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


# Função que realiza o treinamento do modelo
def train(X_train, X_test, y_train, y_test, model):
    with mlflow.start_run(run_name=model[0]):
        # Registro das tags
        mlflow.set_tags(tags)

        clf = model[1]
        # Ajustar modelo ao conjunto de dados de treinamento
        clf.fit(X_train, y_train)
        # Realizar predições no conjunto de dados de teste
        y_pred = clf.predict(X_test)

        # Salvar modelo
        pickle.dump(clf, open(f'models/{model[0]}.sav', 'wb'))

        # Métricas para avaliar a qualidade das previsões do modelo
        accuracy, precision, recall, f1 = eval_metrics(y_test, y_pred)

        print('\n')
        print(model[1])
        print(f'accuracy: {accuracy:.2f}')
        print(f'precision: {precision:.2f}')
        print(f'recall: {recall:.2f}')
        print(f'f1-score: {f1:.2f}')

        # Matriz de confusão
        conf_matrix = plot_confusion_matrix(y_test, y_pred)
        temp_name = 'confusion_matrix.png'
        conf_matrix.savefig(temp_name)
        mlflow.log_artifact(temp_name, "confusion-matrix-plots")
        try:
            os.remove(temp_name)
        except FileNotFoundError as e:
            print(f'{temp_name} file is not found')

        # Registro das métricas no mlflow
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1-score', f1)

        # Registro do modelo no mlflow
        mlflow.sklearn.log_model(clf, 'model')
        mlflow.log_artifact(local_path='mlflow/train.py', artifact_path='code')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Carregar dados
    data = pd.read_csv('water_potability.csv')

    # Preencher os valores ausentes com o valor médio do atributo
    data['ph'].fillna(data['ph'].mean(), inplace=True)
    data['Sulfate'].fillna(data['Sulfate'].mean(), inplace=True)
    data['Trihalomethanes'].fillna(
        data['Trihalomethanes'].mean(), inplace=True)

    # Remover outilers que estão a mais de 3 desvios padrão da média
    for column in data.columns:
        mean = data[column].mean()
        std = data[column].std()

        data = data[(data[column] <= mean + (3 * std))]

    # Separar atributos/classe
    X = data.drop('Potability', axis=1)
    y = data['Potability'].values

    # Oversampling >> sobreamostragem utilizando o SMOTE
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    # Dividir o dataset em dados de treinamento, teste e validação
    # Treinamento/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)
    # Validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2)

    # Padronizar dados
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    # Salvar scaler ajustado
    pickle.dump(scaler, open('scaler/scaler.pkl', 'wb'))
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Treinar modelo
    train(X_train, X_test, y_train, y_test, ('DecisionTreeClassifier',
          DecisionTreeClassifier(criterion='entropy', max_depth=12)))
    train(
        X_train,
        X_test,
        y_train,
        y_test,
        ('KNeighborsClassifier',
         KNeighborsClassifier(
             n_neighbors=1)))
    train(X_train, X_test, y_train, y_test, ('SVC', SVC(C=1, gamma=1)))
