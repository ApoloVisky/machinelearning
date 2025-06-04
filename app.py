import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from data_loader import carregar_dados, preparar_dados_lstm
from utils import buscar_ticker, analisar_tendencia, adicionar_medias_moveis
from model import criar_modelo_lstm, treinar_e_prever

st.set_page_config(page_title="Análise de Ações Avançada", layout="wide")

@st.cache_data(ttl=300)
def carregar_dados_cache(ticker, inicio, fim):
    try:
        print(f"Carregando dados para {ticker} de {inicio} a {fim}")
        # Substitua "sua-chave-api" pela sua chave de API do Alpha Vantage
        df = carregar_dados(ticker, inicio, fim, api_key="PWRWKPHV3S76LVR7")
        df = adicionar_medias_moveis(df)
        print(f"Dados carregados com sucesso: {df.head()}")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("🔍 Busca e Filtros")
    
    nome_ou_ticker = st.text_input("Digite o nome ou ticker da ação:", value="Vale")
    
    with st.spinner("Buscando empresas..."):
        resultados = buscar_ticker(nome_ou_ticker)
    
    opcoes = []
    tickers = []
    for ticker, nome in resultados:
        if ticker is None:
            opcoes.append("Nenhuma empresa encontrada")
            tickers.append(None)
        else:
            opcoes.append(nome)
            tickers.append(ticker)
    
    if opcoes[0] != "Nenhuma empresa encontrada":
        selecionado = st.selectbox("Selecione a empresa:", options=opcoes)
        ticker_selecionado = tickers[opcoes.index(selecionado)]
    else:
        st.error("Nenhuma empresa encontrada. Tente outro nome ou ticker (ex.: Apple, PETR4.SA).")
        st.stop()

    data_inicio = st.date_input("Data inicial:", value=datetime(2022, 1, 1))
    data_fim = st.date_input("Data final:", value=datetime.today())
    if data_fim < data_inicio:
        st.error("A data final deve ser maior ou igual à data inicial.")
        st.stop()
    if st.button("Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo com sucesso!")

# Título
st.title("Análise de Ações Avançada")

if ticker_selecionado:
    nome_empresa = selecionado.split(' (')[0]
    st.markdown(f"### {nome_empresa} ({ticker_selecionado})")
else:
    st.error("Selecione uma empresa para continuar.")
    st.stop()

# Carregar Dados
with st.spinner("Carregando dados da ação..."):
    df = carregar_dados_cache(ticker_selecionado, data_inicio.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
if df is None or df.empty:
    st.error("Não foi possível carregar dados para este ativo no período selecionado. Verifique o ticker ou o período.")
    st.stop()

# Análise
tendencia, confianca = analisar_tendencia(df)

ultimo_preco = float(df['Close_BR'].iloc[-1])
primeiro_preco = float(df['Close_BR'].iloc[0])
variacao = ((ultimo_preco - primeiro_preco) / primeiro_preco) * 100
volume_medio = int(df['Volume'].mean())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Preço Atual", f"R$ {ultimo_preco:.2f}")
with col2:
    st.metric("Variação no Período", f"{variacao:.2f}%", delta_color="normal")
with col3:
    st.metric("Volume Médio", f"{volume_medio:,}")
with col4:
    classe = "green" if tendencia == "subindo" else "red" if tendencia == "descendo" else "gray"
    st.markdown(f"<span style='color:{classe}; font-weight:bold;'>Tendência: {tendencia}</span>", unsafe_allow_html=True)
    st.markdown(f"Confiança: {confianca:.1f}%")

# Gráfico
st.subheader("Evolução do Preço")
cols_para_plotar = ['Close_BR']
if 'MA_20' in df.columns and not df['MA_20'].isna().all():
    cols_para_plotar.append('MA_20')
if 'MA_50' in df.columns and not df['MA_50'].isna().all():
    cols_para_plotar.append('MA_50')
st.line_chart(df[cols_para_plotar])

# Tabela
st.subheader("Dados Históricos")
st.dataframe(df.sort_index(ascending=False), height=300)

# Exportar
csv = df.to_csv(index=True).encode('utf-8')
st.download_button(
    label="Exportar para CSV",
    data=csv,
    file_name=f"{ticker_selecionado}_dados.csv",
    mime="text/csv"
)

# Análise Técnica
st.subheader("Análise Técnica")
if len(df) >= 50:
    if df['Close_BR'].iloc[-1] > df['MA_20'].iloc[-1] > df['MA_50'].iloc[-1]:
        st.success("📈 Tendência de alta (Média 20 dias acima da média 50 dias)")
    elif df['Close_BR'].iloc[-1] < df['MA_20'].iloc[-1] < df['MA_50'].iloc[-1]:
        st.warning("📉 Tendência de baixa (Média 20 dias abaixo da média 50 dias)")
    else:
        st.info("➡️ Tendência lateral (Médias cruzadas)")
else:
    st.info("ℹ️ Dados insuficientes para análise técnica completa (mínimo 50 períodos).")

# Modelo LSTM
st.subheader("Previsão com Modelo LSTM")
if len(df) < 60:
    st.warning("⚠️ Dados insuficientes para treinar o modelo LSTM (mínimo 60 períodos).")
else:
    if st.button("Treinar e Prever com LSTM"):
        with st.spinner("Preparando dados e treinando modelo LSTM..."):
            try:
                X, y, scaler = preparar_dados_lstm(df[['Close_BR']], janela=60)
                model = criar_modelo_lstm(input_shape=(60, 1))
                dados_finais = df['Close_BR'].values[-60:].reshape(-1, 1)
                dados_finais_normalizados = scaler.transform(dados_finais)
                previsao = treinar_e_prever(model, X, y, dados_finais_normalizados, scaler)
                st.success(f"Previsão do próximo preço: R$ {previsao:.2f}")
            except Exception as e:
                st.error(f"Erro ao treinar ou prever com LSTM: {e}")