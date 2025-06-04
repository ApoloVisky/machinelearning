from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def criar_modelo_lstm(input_shape):
    try:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        raise ValueError(f"Erro ao criar modelo LSTM: {e}")

def treinar_e_prever(model, X, y, dados_finais, scaler):
    try:
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Dados de entrada ou alvo estão vazios.")
        
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)
        pred = model.predict(dados_finais.reshape(1, X.shape[1], 1), verbose=0)
        return float(scaler.inverse_transform(pred)[0][0])
    except Exception as e:
        raise ValueError(f"Erro ao treinar ou prever com LSTM: {e}")