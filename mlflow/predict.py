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

# Prever em um array
data = [[8.53482178e+00, 1.21543361e+02, 1.38596765e+04, 9.19642265e+00,
        3.00150377e+02, 4.51143481e+02, 1.47700256e+01, 7.68788629e+01,
        3.98525051e+00],
        [9.55484504e+00, 1.48096691e+02, 1.38596765e+04, 9.22651964e+00,
        3.00150377e+02, 4.51143481e+02, 1.47708629e+01, 7.68780256e+01,
        3.76585051e+00]]

# Carregar o 'StandardScaler' ajustado
scaler = pickle.load(open('scaler/scaler.pkl', 'rb'))
# Transformar a escala dos valor de 'data'
data = scaler.transform(data)
# Realizar previsão
result = loaded_model.predict(data)
print(result)
