import mlflow
import pickle

# Conectar ao servidor
mlflow.set_tracking_uri('http://localhost:5000')

# Local do modelo
logged_model = 'runs:/39e13c8bcec144f38bc87d439d8307bd/model'
# Carregar o modelo
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Carregar modelo com 'pickle'
# loaded_model = pickle.load(open('models/SVC.sav', 'rb'))

# Prever em um Numpy Array
data = [[9.22512984e+00, 1.93805402e+02, 1.11685292e+04, 9.25447109e+00,
        3.07583374e+02, 5.44659021e+02, 8.16639675e+00, 7.28774595e+01,
        3.89516525e+00],
        [9.55484504e+00, 1.48096691e+02, 1.38596765e+04, 8.19642265e+00,
        3.00150377e+02, 4.51143481e+02, 1.47708629e+01, 7.68780256e+01,
        3.98525051e+00]]

# Carregar o 'StandardScaler' ajustado
scaler = pickle.load(open('scaler/scaler.pkl', 'rb'))
# Transformar a escala dos valor de 'data'
data = scaler.transform(data)
# Realizar previsão
result = loaded_model.predict(data)
print(result)
