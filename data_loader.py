import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def carregar_dados(ticker: str, inicio='2015-01-01', fim=None, api_key="sua-chave-api") -> pd.DataFrame:
    """
    Carrega dados históricos da ação via Alpha Vantage API.
    
    Args:
        ticker (str): Código da ação (ex: 'VALE3.SA').
        inicio (str): Data inicial no formato 'YYYY-MM-DD'.
        fim (str | None): Data final. Se None, usa data atual.
        api_key (str): Chave de API do Alpha Vantage.
    
    Returns:
        pd.DataFrame: DataFrame com colunas 'Close_BR', 'MA_20', 'MA_50'.
    
    Raises:
        ValueError: Se dados não forem carregados ou estiverem vazios.
    """
    try:
        if fim is None:
            fim = datetime.now().strftime('%Y-%m-%d')

        # Garantir que a data final não esteja no futuro
        fim_data = pd.to_datetime(fim)
        hoje = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        if fim_data >= hoje:
            fim = (hoje - timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Data final ajustada para {fim} (dia anterior à data atual).")

        # Fazer a requisição para a API do Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full"
        response = requests.get(url)
        response.raise_for_status()  # Levanta um erro para respostas HTTP ruins
        data = response.json()

        # Verificar se há erro na resposta da API
        if "Error Message" in data:
            raise ValueError(f"Erro na API Alpha Vantage para o ticker '{ticker}': {data['Error Message']}")
        if "Time Series (Daily)" not in data:
            raise ValueError(f"Dados não encontrados para o ticker '{ticker}' na API Alpha Vantage.")

        # Converter os dados para DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Filtrar pelo período solicitado
        df = df.loc[inicio:fim]

        # Renomear colunas e converter tipos
        df = df.rename(columns={
            "4. close": "Close_BR",
            "5. volume": "Volume"
        })
        df["Close_BR"] = df["Close_BR"].astype(float)
        df["Volume"] = df["Volume"].astype(int)

        # Verificar se o DataFrame está vazio
        if df.empty:
            raise ValueError(f"Dados não encontrados para o ticker '{ticker}' no período {inicio} a {fim}.")

        # Verificar se todos os valores de 'Close_BR' são nulos
        if df['Close_BR'].isna().all():
            raise ValueError(f"Todos os valores de 'Close_BR' são nulos para o ticker '{ticker}' no período {inicio} a {fim}.")

        # Remover linhas onde 'Close_BR' é nulo e verificar se o DataFrame ainda tem dados
        df = df.dropna(subset=['Close_BR'])
        if df.empty:
            raise ValueError(f"Após remover valores nulos, não há dados válidos para o ticker '{ticker}' no período {inicio} a {fim}.")

        print(f"Dados brutos para {ticker}: {df.head()}")

        # Médias móveis
        df['MA_20'] = df['Close_BR'].rolling(window=20).mean()
        df['MA_50'] = df['Close_BR'].rolling(window=50).mean()

        return df
    except Exception as e:
        raise ValueError(f"Erro ao carregar dados do ticker '{ticker}': {str(e)}")

def preparar_dados_lstm(data: np.ndarray, janela: int = 60):
    """
    Prepara os dados para o modelo LSTM.
    """
    try:
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise TypeError("Os dados devem ser um numpy array ou DataFrame.")
        
        if isinstance(data, pd.DataFrame):
            data = data[['Close_BR']].values

        if len(data) < janela:
            raise ValueError(f"É necessário pelo menos {janela} períodos para preparar os dados.")

        scaler = MinMaxScaler()
        dados_normalizados = scaler.fit_transform(data)

        X, y = [], []
        for i in range(janela, len(dados_normalizados)):
            X.append(dados_normalizados[i-janela:i, 0])
            y.append(dados_normalizados[i, 0])

        X = np.array(X).reshape(-1, janela, 1)
        y = np.array(y).astype(np.float32)
        
        return X, y, scaler
    except Exception as e:
        raise ValueError(f"Erro ao preparar dados para LSTM: {str(e)}")