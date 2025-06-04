import yfinance as yf
from yahooquery import search
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def buscar_ticker(nome_ou_ticker):
    """
    Busca uma lista de empresas com base no nome ou ticker fornecido.
    
    Args:
        nome_ou_ticker (str): Nome ou ticker da ação.
    
    Returns:
        list: Lista de tuplas (ticker, nome_empresa) ou [(None, None)] se não encontrado.
    """
    try:
        nome_ou_ticker = nome_ou_ticker.strip()
        if not nome_ou_ticker:
            return [(None, None)]

        # Tenta buscar diretamente como ticker
        ticker_input = nome_ou_ticker.upper()
        ticker = yf.Ticker(ticker_input)
        info = ticker.info
        if 'shortName' in info and info['shortName']:
            return [(ticker_input, info['shortName'])]
    except Exception as e:
        print(f"Erro ao buscar ticker diretamente: {e}")

    # Busca por nome usando yahooquery
    try:
        resultados = search(nome_ou_ticker)
        if not resultados or 'quotes' not in resultados or not resultados['quotes']:
            return [(None, None)]

        empresas = []
        for res in resultados['quotes']:
            if res.get('quoteType') == 'EQUITY':
                ticker = res['symbol']
                nome = res.get('shortname', 'Nome não disponível')
                # Priorizar tickers brasileiros (ex.: .SA) para empresas como Vale
                if nome.lower().startswith('vale') and ticker == 'VALE':
                    empresas.append(('VALE3.SA', f"{nome} (B3: VALE3.SA)"))
                empresas.append((ticker, f"{nome} ({ticker})"))
        
        return empresas if empresas else [(None, None)]
    except Exception as e:
        print(f"Erro na busca por nome: {e}")
        return [(None, None)]

def analisar_tendencia(df):
    """
    Analisa a tendência do preço com base em regressão linear.
    
    Args:
        df (pd.DataFrame): DataFrame com coluna 'Close_BR'.
    
    Returns:
        tuple: (tendência, confiança).
    """
    try:
        if df['Close_BR'].isna().any() or len(df) < 30:
            return "indeterminada", 0.0
        
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close_BR'].values
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X[-30:])
        tendencia = "subindo" if pred[-1] > pred[0] else "descendo"
        
        y_mean = y.mean() if y.mean() != 0 else 1.0
        confianca = min(100.0, abs(model.coef_[0] * 100 / y_mean))
        
        return tendencia, confianca
    except Exception as e:
        print(f"Erro na análise de tendência: {e}")
        return "indeterminada", 0.0

def adicionar_medias_moveis(df: pd.DataFrame, col='Close_BR', janelas=(20, 50)):
    """
    Adiciona médias móveis ao DataFrame.
    """
    df = df.copy()
    for j in janelas:
        if len(df) >= j:
            df[f'MA_{j}'] = df[col].rolling(window=j).mean()
        else:
            df[f'MA_{j}'] = pd.NA
    return df