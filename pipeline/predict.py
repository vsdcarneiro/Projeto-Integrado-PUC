import mlflow
import pickle
import warnings
warnings.filterwarnings('ignore')

# Conectar ao servidor
mlflow.set_tracking_uri('http://localhost:5000')

# Local do modelo
logged_model = 'runs:/111cbbd9ccf34ce5b9ff76c8fe138abc/model'
# Carregar o modelo
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Carregar modelo com 'pickle'
# loaded_model = pickle.load(open('models/SVC.sav', 'rb'))

# Carregar o 'StandardScaler' ajustado
scaler = pickle.load(open('scaler/scaler.pkl', 'rb'))


def predict_quality(X):
    try:
        # Transformar a escala dos valor de 'X'
        X = scaler.transform(X)
        # Realizar previsão
        result = loaded_model.predict(X)
        return result
    except ValueError as e:
        print(e)
