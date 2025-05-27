import streamlit as st
st.set_page_config(layout="wide", page_title="Stratégie de Trading Multi-Facteurs", page_icon="📈")



# *** Installation des bibliothèques nécessaires ***
# Décommentez ces lignes lorsque vous exécutez pour la première fois
# !pip install yfinance pandas numpy matplotlib streamlit seaborn vaderSentiment requests python-dotenv plotly

# *** Import des modules ***

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# *** Configuration et chargement des variables d'environnement ***
load_dotenv()  # Charger les variables depuis un fichier .env

# Pour les tests, vous pouvez remplacer cette ligne par votre clé API directement
# Dans un environnement de production, utilisez un fichier .env ou les secrets de Streamlit
API_KEY = os.getenv("FINNHUB_API_KEY", "d012njpr01qv3oh2b3a0d012njpr01qv3oh2b3ag")

# *** Définition des constantes ***
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"
DEFAULT_SMA_SHORT = 20
DEFAULT_SMA_LONG = 50
DEFAULT_RSI_WINDOW = 14
SENTIMENT_DAYS = 7
PER_THRESHOLD = 25
ROE_THRESHOLD = 0.10
DE_THRESHOLD = 1.5

# *** Partie 1: Récupération et traitement des données ***

def get_price_data(ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """
    Récupère les données historiques de prix pour un ticker spécifié
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            st.error(f"Aucune donnée disponible pour {ticker}")
            return pd.DataFrame()
            
        return hist
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {str(e)}")
        return pd.DataFrame()

# *** Partie 2: Indicateurs techniques ***

def add_sma(df, short=DEFAULT_SMA_SHORT, long=DEFAULT_SMA_LONG):
    """
    Ajoute les moyennes mobiles simples (SMA) au DataFrame
    """
    df[f"SMA{short}"] = df["Close"].rolling(window=short).mean()
    df[f"SMA{long}"] = df["Close"].rolling(window=long).mean()
    return df

def compute_rsi(df, window=DEFAULT_RSI_WINDOW):
    """
    Calcule le Relative Strength Index (RSI)
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    return df

def compute_macd(df):
    """
    Calcule le Moving Average Convergence Divergence (MACD)
    """
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    return df

def compute_bollinger_bands(df, window=20, num_std=2):
    """
    Ajoute les bandes de Bollinger au DataFrame
    """
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
    df['Bollinger_Middle'] = rolling_mean
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
    return df

def compute_atr(df, window=14):
    """
    Calcule l'Average True Range (ATR)
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    df['ATR'] = true_range.rolling(window=window).mean()
    return df

def compute_stochastic(df, k_window=14, d_window=3):
    """
    Calcule l'oscillateur stochastique
    """
    lowest_low = df['Low'].rolling(window=k_window).min()
    highest_high = df['High'].rolling(window=k_window).max()
    
    df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_window).mean()
    return df

# *** Partie 3: Génération des signaux techniques ***

def generate_signals(df):
    """
    Génère des signaux d'achat et de vente basés sur les indicateurs techniques
    """
    # Trouver quelle colonne SMA court terme est présente dans le DataFrame
    sma_columns = [col for col in df.columns if col.startswith('SMA')]
    if not sma_columns:
        raise ValueError("Aucune colonne SMA trouvée dans le DataFrame")
    
    # Utiliser la première SMA trouvée (généralement la plus courte)
    sma_short_col = min(sma_columns, key=lambda x: int(x.replace('SMA', '')))
    
    # Signal long si le prix est au-dessus de la SMA courte, RSI pas trop élevé, MACD > Signal_Line
    df["Signal_Long"] = (
        (df["Close"] > df[sma_short_col]) &
        (df["RSI"] < 70) &
        (df["MACD"] > df["Signal_Line"])
    )
    
    # Signal short si le prix est en dessous de la SMA courte, RSI pas trop bas, MACD < Signal_Line
    df["Signal_Short"] = (
        (df["Close"] < df[sma_short_col]) &
        (df["RSI"] > 30) &
        (df["MACD"] < df["Signal_Line"])
    )
    
    # Attribution des signaux
    df["Signal"] = np.select(
        [df["Signal_Long"], df["Signal_Short"]],
        ["long", "short"],
        default="hold"
    )
    
    return df

def generate_advanced_signals(df):
    """
    Génère des signaux plus avancés incluant les bandes de Bollinger et Stochastique
    """
    # Signaux basés sur BB (survente/surachat)
    df["BB_Long"] = df["Close"] < df["Bollinger_Lower"]
    df["BB_Short"] = df["Close"] > df["Bollinger_Upper"]
    
    # Signaux basés sur Stochastique
    df["Stoch_Long"] = (df["Stoch_K"] < 20) & (df["Stoch_K"] > df["Stoch_K"].shift(1))
    df["Stoch_Short"] = (df["Stoch_K"] > 80) & (df["Stoch_K"] < df["Stoch_K"].shift(1))
    
    # Signaux combinés (plus conservateurs)
    df["Advanced_Long"] = (
        df["Signal_Long"] &
        (df["BB_Long"] | df["Stoch_Long"])
    )
    
    df["Advanced_Short"] = (
        df["Signal_Short"] &
        (df["BB_Short"] | df["Stoch_Short"])
    )
    
    # Attribution des signaux avancés
    df["Advanced_Signal"] = np.select(
        [df["Advanced_Long"], df["Advanced_Short"]],
        ["long", "short"],
        default="hold"
    )
    
    return df

# *** Partie 4: Analyse des actualités ***

def get_finnhub_news(ticker, from_date, to_date):
    """
    Récupère les actualités pour un ticker via l'API Finnhub
    """
    url = "https://finnhub.io/api/v1/company-news"
    
    try:
        params = {"symbol": ticker, "from": from_date, "to": to_date, "token": API_KEY}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            st.warning("Limite d'API atteinte. Nouvelle tentative dans quelques secondes...")
            time.sleep(2)  # Attendre avant de réessayer
            return get_finnhub_news(ticker, from_date, to_date)
            
        if response.status_code != 200:
            st.warning(f"Erreur API Finnhub: {response.status_code}. Utilisation de données simulées.")
            # Retourner des données simulées en cas d'erreur API
            return simulate_news_data(ticker)
        
        data = response.json()
        
        # Traitement des articles
        news = []
        for article in data:
            title = article.get('headline', '')
            description = article.get('summary', '')
            if title or description:  # Vérifier qu'il y a du contenu
                news.append({
                    'title': title,
                    'description': description,
                    'publishedAt': datetime.fromtimestamp(article.get('datetime', 0)),
                    'url': article.get('url', '')
                })
        
        if not news:
            st.warning(f"Aucune actualité trouvée pour {ticker}. Utilisation de données simulées.")
            return simulate_news_data(ticker)
            
        return pd.DataFrame(news)
        
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des actualités: {str(e)}. Utilisation de données simulées.")
        return simulate_news_data(ticker)

def simulate_news_data(ticker):
    """
    Génère des données d'actualités simulées en cas d'échec de l'API
    """
    today = datetime.now()
    
    # Créer des actualités fictives basées sur le ticker
    simulated_news = [
        {
            'title': f"{ticker} annonce ses résultats trimestriels",
            'description': f"L'entreprise {ticker} a publié des résultats supérieurs aux attentes des analystes pour ce trimestre.",
            'publishedAt': today - timedelta(days=1),
            'url': '#'
        },
        {
            'title': f"Nouveau produit lancé par {ticker}",
            'description': f"{ticker} a lancé un nouveau produit qui pourrait augmenter significativement ses revenus.",
            'publishedAt': today - timedelta(days=3),
            'url': '#'
        },
        {
            'title': f"Analyse du marché: impact sur {ticker}",
            'description': f"Les analystes prédisent une croissance stable pour {ticker} malgré les conditions de marché difficiles.",
            'publishedAt': today - timedelta(days=5),
            'url': '#'
        }
    ]
    
    return pd.DataFrame(simulated_news)

def analyze_sentiment(df):
    """
    Analyse le sentiment des actualités avec VADER
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # Fonction pour analyser le sentiment d'un texte
    def get_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 0
        return analyzer.polarity_scores(text)["compound"]
    
    # Appliquer l'analyse de sentiment aux titres et descriptions
    df["title_sentiment"] = df["title"].apply(get_sentiment)
    df["description_sentiment"] = df["description"].apply(get_sentiment)
    
    # Combinaison des scores (titre a plus de poids)
    df["sentiment_score"] = (df["title_sentiment"] * 0.7) + (df["description_sentiment"] * 0.3)
    
    return df

def get_recent_news(ticker, days=SENTIMENT_DAYS):
    """
    Récupère et analyse les actualités récentes pour un ticker
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    
    df = get_finnhub_news(ticker, from_date.strftime('%Y-%m-%d'), to_date.strftime('%Y-%m-%d'))
    
    if not df.empty:
        df = analyze_sentiment(df)
        
    return df

def get_news_signal(sentiment_score, seuil_pos=0.2, seuil_neg=-0.2):
    """
    Génère un signal basé sur le sentiment des actualités
    """
    if sentiment_score is None:
        return "hold"
    if sentiment_score >= seuil_pos:
        return "buy"
    elif sentiment_score <= seuil_neg:
        return "sell"
    else:
        return "hold"

# *** Partie 5: Analyse fondamentale ***

def get_fundamentals(ticker):
    """
    Récupère les données fondamentales pour un ticker
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Récupérer les ratios clés
        per = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        de = info.get("debtToEquity")  # En pourcentage
        
        # Données financières supplémentaires
        market_cap = info.get("marketCap")
        dividend_yield = info.get("dividendYield")
        beta = info.get("beta")
        sector = info.get("sector", "Unknown")
        
        return {
            "per": per,
            "roe": roe,
            "de": de,
            "market_cap": market_cap,
            "dividend_yield": dividend_yield,
            "beta": beta,
            "sector": sector
        }
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des fondamentaux: {str(e)}")
        return {
            "per": None,
            "roe": None,
            "de": None,
            "market_cap": None,
            "dividend_yield": None,
            "beta": None,
            "sector": "Unknown"
        }

def is_fundamentally_solid(ticker, per_thresh=PER_THRESHOLD, roe_thresh=ROE_THRESHOLD, de_thresh=DE_THRESHOLD):
    """
    Évalue si une entreprise est fondamentalement solide
    """
    fundamentals = get_fundamentals(ticker)
    
    per = fundamentals["per"]
    roe = fundamentals["roe"]
    de = fundamentals["de"]
    
    if per is None or roe is None or de is None:
        return per, roe, de, "hold"
        
    # Génération du signal
    if per < per_thresh and roe > roe_thresh and (de is None or de < de_thresh * 100):
        signal = "buy"
    elif per > per_thresh * 1.5 or (roe is not None and roe < roe_thresh / 2) or (de is not None and de > de_thresh * 150):
        signal = "sell"
    else:
        signal = "hold"
        
    return per, roe, de, signal

# *** Partie 6: Macro-économie ***

def get_macro_indicators():
    """
    Récupère des indicateurs macroéconomiques via yfinance
    """
    try:
        macro_data = {}
        
        # Indices comme le VIX pour la volatilité du marché
        vix_data = yf.Ticker("^VIX").history(period="1mo")
        if not vix_data.empty:
            macro_data['VIX'] = vix_data['Close'].iloc[-1]
            macro_data['VIX_change'] = ((vix_data['Close'].iloc[-1] / vix_data['Close'].iloc[0]) - 1) * 100
        
        # S&P 500 (tendance générale du marché)
        sp500_data = yf.Ticker("^GSPC").history(period="1mo")
        if not sp500_data.empty:
            macro_data['SP500_change'] = ((sp500_data['Close'].iloc[-1] / sp500_data['Close'].iloc[0]) - 1) * 100
        
        # Taux des bons du Trésor américain à 10 ans
        bonds_data = yf.Ticker("^TNX").history(period="1mo")
        if not bonds_data.empty:
            macro_data['Taux_10ans'] = bonds_data['Close'].iloc[-1]
            macro_data['Taux_10ans_change'] = bonds_data['Close'].iloc[-1] - bonds_data['Close'].iloc[0]
        
        return macro_data
        
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des données macro: {str(e)}")
        return {}

def get_macro_signal(macro_data):
    """
    Génère un signal basé sur les indicateurs macroéconomiques
    """
    # Règles simplifiées pour l'exemple
    if not macro_data:
        return "neutral"
        
    # Signaux négatifs
    if macro_data.get('VIX', 0) > 30 or macro_data.get('VIX_change', 0) > 15:
        return "bearish"  # Forte volatilité = signal baissier
        
    # Signaux positifs
    if macro_data.get('SP500_change', 0) > 5 and macro_data.get('VIX', 100) < 20:
        return "bullish"  # Marché haussier avec faible volatilité
        
    # Par défaut
    return "neutral"

# *** Partie 7: Évaluation des signaux et backtest ***

def evaluate_signals(df):
    """
    Évalue les signaux en calculant les retours sur 5 jours après chaque signal
    """
    # Calculer les retours sur différentes périodes (1, 5, 10 jours)
    df['Return_1d'] = df['Close'].pct_change(periods=1).shift(-1)
    df['Return_5d'] = df['Close'].pct_change(periods=5).shift(-5)
    df['Return_10d'] = df['Close'].pct_change(periods=10).shift(-10)
    
    return df

def generate_final_signal(df, ticker, sentiments, fundamentals=None, macro_signal="neutral"):
    """
    Génère un signal final en combinant technique + news + fondamentaux + macro
    """
    # Récupérer les scores de sentiment
    sentiment_value = sentiments.get(ticker)
    news_signal = get_news_signal(sentiment_value)
    
    # Récupérer le signal fondamental si non fourni
    if fundamentals is None:
        _, _, _, fund_signal = is_fundamentally_solid(ticker)
    else:
        fund_signal = fundamentals.get('signal', 'hold')
    
    # Par défaut, on conserve le signal technique
    df['Final_Signal'] = df['Signal']
    
    # Appliquer des règles pour le signal final (dernière ligne seulement)
    latest_idx = df.index[-1]
    
    # Règles de combinaison
    tech_signal = df.loc[latest_idx, 'Signal']
    
    # 1. Si tous les signaux sont cohérents, signal fort
    if tech_signal == "long" and news_signal == "buy" and fund_signal == "buy" and macro_signal in ["neutral", "bullish"]:
        df.loc[latest_idx, 'Final_Signal'] = "strong_buy"
    elif tech_signal == "short" and news_signal == "sell" and fund_signal == "sell" and macro_signal in ["neutral", "bearish"]:
        df.loc[latest_idx, 'Final_Signal'] = "strong_sell"
    
    # 2. Si signaux contradictoires, neutraliser
    elif tech_signal == "long" and (news_signal == "sell" or fund_signal == "sell" or macro_signal == "bearish"):
        df.loc[latest_idx, 'Final_Signal'] = "hold"
    elif tech_signal == "short" and (news_signal == "buy" or fund_signal == "buy" or macro_signal == "bullish"):
        df.loc[latest_idx, 'Final_Signal'] = "hold"
    
    # Résumé des facteurs
    df.loc[latest_idx, 'Tech_Signal'] = tech_signal
    df.loc[latest_idx, 'News_Signal'] = news_signal
    df.loc[latest_idx, 'Fund_Signal'] = fund_signal
    df.loc[latest_idx, 'Macro_Signal'] = macro_signal
    
    # Remplir pour les lignes précédentes pour éviter les NaN
    df['Tech_Signal'].fillna(method='ffill', inplace=True)
    df['News_Signal'].fillna(news_signal, inplace=True)
    df['Fund_Signal'].fillna(fund_signal, inplace=True)
    df['Macro_Signal'].fillna(macro_signal, inplace=True)
        
    return df

def backtest_signals(df):
    """
    Effectue un backtest simple des signaux
    """
    # Initialiser les colonnes de performance
    df['Long_Return'] = np.where(df['Signal'] == "long", df['Return_5d'], 0)
    df['Short_Return'] = np.where(df['Signal'] == "short", -df['Return_5d'], 0)
    df['Strategy_Return'] = df['Long_Return'] + df['Short_Return']
    
    # Calculer les rendements cumulés
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    df['Cumulative_Market_Return'] = (1 + df['Return_5d']).cumprod() - 1
    
    return df

def resume_backtest(df):
    """
    Affiche un résumé des performances du backtest
    """
    stats = {}
    
    # Nombre de signaux
    nb_longs = len(df[df['Signal'] == "long"])
    nb_shorts = len(df[df['Signal'] == "short"])
    stats["nb_longs"] = nb_longs
    stats["nb_shorts"] = nb_shorts
    
    # Taux de réussite
    win_longs = len(df[(df['Signal'] == "long") & (df['Return_5d'] > 0)])
    win_shorts = len(df[(df['Signal'] == "short") & (df['Return_5d'] < 0)])
    
    if nb_longs > 0:
        success_rate_long = win_longs / nb_longs * 100
        stats["success_rate_long"] = success_rate_long
    else:
        stats["success_rate_long"] = None
        
    if nb_shorts > 0:
        success_rate_short = win_shorts / nb_shorts * 100
        stats["success_rate_short"] = success_rate_short
    else:
        stats["success_rate_short"] = None
    
    # Performance globale
    if 'Cumulative_Strategy_Return' in df.columns and len(df) > 0:
        final_return = df['Cumulative_Strategy_Return'].iloc[-1]
        stats["final_return"] = final_return
        
        # Compare with buy & hold
        if 'Cumulative_Market_Return' in df.columns:
            market_return = df['Cumulative_Market_Return'].iloc[-1]
            stats["market_return"] = market_return
            stats["alpha"] = final_return - market_return
    
    return stats

def calculate_position_size(df, ticker, capital=10000, risk_pct=1):
    """
    Calcule la taille de position optimale basée sur l'ATR et le capital disponible
    """
    # S'assurer que l'ATR est calculé
    if 'ATR' not in df.columns:
        df = compute_atr(df)
    
    latest_close = df['Close'].iloc[-1]
    latest_atr = df['ATR'].iloc[-1]
    
    # Stop loss basé sur ATR (2 ATR)
    stop_loss = 2 * latest_atr
    
    # Capital risqué
    capital_at_risk = capital * (risk_pct / 100)
    
    # Nombre d'actions
    num_shares = capital_at_risk / stop_loss
    
    # Coût total
    position_cost = num_shares * latest_close
    
    return {
        'ticker': ticker,
        'close_price': latest_close,
        'atr': latest_atr,
        'stop_loss_price': latest_close - stop_loss if df['Signal'].iloc[-1] == "long" else latest_close + stop_loss,
        'stop_loss_pct': (stop_loss / latest_close) * 100,
        'capital_at_risk': capital_at_risk,
        'num_shares': int(num_shares),
        'position_cost': position_cost
    }

def tableau_synthetique(resultats, sentiments, per_thresh=PER_THRESHOLD, roe_thresh=ROE_THRESHOLD, de_thresh=DE_THRESHOLD, window=5):
    """
    Crée un tableau synthétique des signaux pour tous les tickers
    """
    def map_to_emoji(signal, is_tech=False):
        if signal in ["buy", "long", "strong_buy"] or (is_tech and signal == "long"):
            return "🟢"
        elif signal in ["sell", "short", "strong_sell"] or (is_tech and signal == "short"):
            return "🔴"
        else:
            return "🟡"

    tableau = []
    for ticker, df in resultats.items():
        if df.empty:
            continue
            
        df_tail = df.tail(window).copy()

        # Moyennes des indicateurs sur les derniers jours
        close_mean = df_tail["Close"].mean()
        sma20_mean = df_tail["SMA20"].mean() if "SMA20" in df_tail.columns else None
        rsi_mean = df_tail["RSI"].mean() if "RSI" in df_tail.columns else None
        macd_mean = df_tail["MACD"].mean() if "MACD" in df_tail.columns else None

        # Récupération du dernier signal technique
        last_signal = df_tail["Signal"].iloc[-1] if "Signal" in df_tail.columns else "hold"
        
        # Signal presse
        avg_sentiment = sentiments.get(ticker)
        signal_presse = "buy" if avg_sentiment is not None and avg_sentiment > 0.05 else (
                         "sell" if avg_sentiment is not None and avg_sentiment < -0.05 else "hold")

        # Signal fondamental
        fundamentals = get_fundamentals(ticker)
        per = fundamentals["per"]
        roe = fundamentals["roe"]
        de = fundamentals["de"]

        signal_fondamental = "hold"
        if per is not None and roe is not None:
            if per < per_thresh and roe > roe_thresh and (de is None or de < de_thresh * 100):
                signal_fondamental = "buy"
            elif per > per_thresh * 1.5 or (roe is not None and roe < roe_thresh / 2) or (de is not None and de > de_thresh * 150):
                signal_fondamental = "sell"

        # Emojis pour la visualisation
        emoji_tech = map_to_emoji(last_signal, is_tech=True)
        emoji_presse = map_to_emoji(signal_presse)
        emoji_fond = map_to_emoji(signal_fondamental)
        signal_final = emoji_tech + emoji_presse + emoji_fond

        # Création de la ligne du tableau
        tableau.append({
            "Ticker": ticker,
            "Date": df_tail.index[-1].strftime('%Y-%m-%d') if not df_tail.empty else None,
            "Close (moy5j)": round(close_mean, 2) if not np.isnan(close_mean) else None,
            "SMA20 (moy5j)": round(sma20_mean, 2) if sma20_mean is not None and not np.isnan(sma20_mean) else None,
            "RSI (moy5j)": round(rsi_mean, 2) if rsi_mean is not None and not np.isnan(rsi_mean) else None,
            "MACD (moy5j)": round(macd_mean, 3) if macd_mean is not None and not np.isnan(macd_mean) else None,
            "Signal_Technique": last_signal,
            "Sentiment_Moyen": round(avg_sentiment, 2) if avg_sentiment is not None else None,
            "Signal_Presse": signal_presse,
            "PER": round(per, 2) if per is not None else None,
            "ROE": round(roe * 100, 2) if roe is not None else None,
            "D/E": round(de, 2) if de is not None else None,
            "Signal_Fondamental": signal_fondamental,
            "Signal_Final": signal_final
        })

    # Création du DataFrame
    return pd.DataFrame(tableau)

# *** Partie 8: Visualisations ***

def plot_trading_strategy(df, ticker, window=60):
    """
    Crée un graphique interactif avec Plotly pour la stratégie de trading
    """
    # Extraire une fenêtre de données récentes
    recent_df = df.tail(window).copy()
    
    # Créer la figure de base avec la courbe de prix
    fig = go.Figure()
    
    # Ajouter la courbe de prix
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Close'],
        mode='lines',
        name='Prix',
        line=dict(color='royalblue', width=2)
    ))
    
    # Ajouter les SMA
    if 'SMA20' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df['SMA20'],
            mode='lines',
            name='SMA20',
            line=dict(color='orange', width=1.5, dash='dot')
        ))
    
    if 'SMA50' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df.index,
            y=recent_df['SMA50'],
            mode='lines',
            name='SMA50',
            line=dict(color='red', width=1.5, dash='dot')
        ))
    
    # Ajouter les signaux d'achat (flèches vertes)
    buy_signals = recent_df[recent_df['Signal'] == "long"]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Signal achat',
            marker=dict(symbol='triangle-up', size=15, color='green')
        ))
    
    # Ajouter les signaux de vente (flèches rouges)
    sell_signals = recent_df[recent_df['Signal'] == "short"]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Signal vente',
            marker=dict(symbol='triangle-down', size=15, color='red')
        ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title=f'Stratégie de trading pour {ticker}',
        xaxis_title='Date',
        yaxis_title='Prix',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_technical_indicators(df, ticker, window=60):
    """
    Crée un graphique complet des indicateurs techniques
    """
    recent_df = df.tail(window).copy()
    
    # Créer la figure avec des sous-graphiques
    fig = go.Figure()
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxes=True, 
                       vertical_spacing=0.05,
                       row_heights=[0.5, 0.15, 0.15, 0.2],
                       subplot_titles=('Prix', 'MACD', 'RSI', 'Volume'))
    
    # Graphique des prix
    fig.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df['Close'], name='Prix',
        line=dict(color='royalblue')
    ), row=1, col=1)
    
    # Ajouter SMA
    fig.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df['SMA20'], name='SMA20',
        line=dict(color='orange', dash='dot')
    ), row=1, col=1)
    
    if 'SMA50' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=recent_df['SMA50'], name='SMA50',
            line=dict(color='red', dash='dot')
        ), row=1, col=1)
    
    # Ajouter les Bandes de Bollinger si disponibles
    if all(x in recent_df.columns for x in ['Bollinger_Upper', 'Bollinger_Lower']):
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=recent_df['Bollinger_Upper'], name='Boll Upper',
            line=dict(color='rgba(250,174,50,0.4)'), showlegend=True
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=recent_df['Bollinger_Lower'], name='Boll Lower',
            line=dict(color='rgba(250,174,50,0.4)'), 
            fill='tonexty', fillcolor='rgba(250,174,50,0.1)', showlegend=True
        ), row=1, col=1)
    
    # MACD
    if all(x in recent_df.columns for x in ['MACD', 'Signal_Line']):
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=recent_df['MACD'], name='MACD',
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=recent_df['Signal_Line'], name='Signal Line',
            line=dict(color='red')
        ), row=2, col=1)
        
        # Histogramme MACD
        if 'MACD_Histogram' in recent_df.columns:
            colors = ['green' if val >= 0 else 'red' for val in recent_df['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=recent_df.index, y=recent_df['MACD_Histogram'], name='Histogram',
                marker_color=colors
            ), row=2, col=1)
    
    # RSI
    if 'RSI' in recent_df.columns:
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=recent_df['RSI'], name='RSI',
            line=dict(color='purple')
        ), row=3, col=1)
        
        # Lignes de survente/surachat
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=[70] * len(recent_df), name='Surachat',
            line=dict(color='red', dash='dash'), showlegend=False
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=recent_df.index, y=[30] * len(recent_df), name='Survente',
            line=dict(color='green', dash='dash'), showlegend=False
        ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=recent_df.index, y=recent_df['Volume'], name='Volume',
        marker_color='rgba(0,0,250,0.3)'
    ), row=4, col=1)
    
    # Mise en page
    fig.update_layout(
        title=f'Analyse technique pour {ticker}',
        height=800,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white'
    )
    
    # Mettre à jour les axes y
    fig.update_yaxes(title_text='Prix', row=1, col=1)
    fig.update_yaxes(title_text='MACD', row=2, col=1)
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text='Volume', row=4, col=1)
    
    return fig

def visualiser_strategie(df, ticker):
    """
    Visualisation matplotlib de la courbe de prix avec les signaux d'achat et de vente
    """
    plt.figure(figsize=(12, 6))
    
    # Tracer la courbe de prix
    plt.plot(df['Close'], label='Prix de clôture', color='blue', lw=2)
    
    # Tracer la SMA20
    plt.plot(df['SMA20'], label='SMA20', color='orange', linestyle='--', lw=2)
    
    # Tracer les points d'achat (flèches vertes)
    achat = df[df['Signal'] == "long"]
    if not achat.empty:
        plt.scatter(achat.index, achat['Close'], marker='^', color='green', s=100, label='Point d\'achat', zorder=5)
    
    # Tracer les points de vente (flèches rouges)
    vente = df[df['Signal'] == "short"]
    if not vente.empty:
        plt.scatter(vente.index, vente['Close'], marker='v', color='red', s=100, label='Point de vente', zorder=5)
    
    # Ajouter des titres et labels
    plt.title(f"Stratégie de trading pour {ticker}", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Prix', fontsize=12)
    plt.legend(loc='best')
    
    # Rotation des labels de l'axe x
    plt.xticks(rotation=45)
    
    # Afficher le graphique
    plt.tight_layout()
    
    return plt

def stats_strategie(df):
    """
    Calcule les statistiques de la stratégie
    """
    stats = {}
    
    # Vérifier si les colonnes nécessaires existent
    if 'Return_5d' not in df.columns:
        df = evaluate_signals(df)
        
    if 'Cumulative_Strategy_Return' not in df.columns:
        df = backtest_signals(df)
    
    # Retour cumulé de la stratégie
    stats['Retour Cumulé'] = df['Cumulative_Strategy_Return'].iloc[-1] if not df.empty else None
    
    # Calcul du nombre de trades gagnants et perdants
    try:
        trades_gagnants = len(df[df['Signal'] == "long"][df['Return_5d'] > 0])
        trades_perdants = len(df[df['Signal'] == "short"][df['Return_5d'] < 0])
        
        stats['Trades gagnants'] = trades_gagnants
        stats['Trades perdants'] = trades_perdants
        stats['Taux de gain'] = trades_gagnants / (trades_gagnants + trades_perdants) if trades_gagnants + trades_perdants > 0 else 0
        
        # Return des derniers trades
        stats['Dernier retour'] = df['Return_5d'].iloc[-1] if not df.empty else None
    except Exception as e:
        st.warning(f"Erreur lors du calcul des statistiques: {str(e)}")
    
    return stats

# *** Partie 9: Mode d'Emploi (NOUVELLE FONCTION) ***

def create_user_guide():
    """
    Crée la section Mode d'Emploi avec des explications détaillées sur l'utilisation de l'application
    """
    st.title("📘 Mode d'Emploi - Stratégie de Trading Multi-Facteurs")
    
    # Création d'un système d'onglets pour organiser les différentes parties du guide
    guide_tabs = st.tabs([
        "Introduction", 
        "Concepts de base", 
        "Indicateurs techniques",
        "Analyse des actualités",
        "Analyse fondamentale",
        "Signaux de trading",
        "Backtest",
        "Graphiques",
        "Gestion des risques",
        "FAQ & Glossaire"
    ])
    
    # Tab 1: Introduction
    with guide_tabs[0]:
        st.header("Introduction")
        st.markdown("""
        Ce guide est destiné à vous aider à comprendre et à utiliser efficacement l'application de Stratégie de Trading Multi-Facteurs, 
        même si vous n'avez pas d'expérience préalable en trading. Il explique les concepts clés, les indicateurs techniques et comment 
        interpréter les résultats pour prendre des décisions d'investissement plus éclairées.
        
        Utilisez les onglets ci-dessus pour naviguer entre les différentes sections du guide.
        """)
    
    # Tab 2: Concepts de base du trading
    with guide_tabs[1]:
        st.header("Concepts de base du trading")
        
        st.subheader("Les positions longues et courtes")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Position longue (long)** : Acheter une action en espérant que son prix augmente. C'est la forme d'investissement la plus courante.")
        with col2:
            st.info("**Position courte (short)** : Parier sur la baisse du prix d'une action. Techniquement, vous empruntez des actions pour les vendre immédiatement, puis les racheter plus tard à un prix inférieur.")
        
        st.subheader("Tendances du marché")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("**Marché haussier (bullish)** : Période où les prix des actifs augmentent généralement.")
        with col2:
            st.error("**Marché baissier (bearish)** : Période où les prix des actifs diminuent généralement.")
        with col3:
            st.warning("**Marché latéral (sideways)** : Période où les prix fluctuent dans une fourchette sans tendance claire.")
        
        st.subheader("Périodes d'analyse")
        st.markdown("""
        L'application permet de sélectionner différentes périodes d'analyse :
        - **1mo** : 1 mois
        - **3mo** : 3 mois
        - **6mo** : 6 mois
        - **1y** : 1 an
        - **2y** : 2 ans
        
        Choisissez une période plus courte pour une analyse à court terme et une période plus longue pour identifier les tendances de fond.
        """)
    
    # Tab 3: Indicateurs techniques
    with guide_tabs[2]:
        st.header("Comprendre les indicateurs techniques")
        
        # Moyennes Mobiles Simples (SMA)
        st.subheader("Moyennes Mobiles Simples (SMA)")
        st.markdown("""
        Les moyennes mobiles aident à identifier la tendance générale en lissant les fluctuations de prix.
        
        - **SMA court terme (par défaut : 20 jours)** : Reflète la tendance récente du prix.
        - **SMA long terme (par défaut : 50 jours)** : Reflète la tendance de fond.
        - **Interprétation** : Lorsque la SMA court terme passe au-dessus de la SMA long terme, c'est souvent considéré comme un signal d'achat (croisement haussier). L'inverse est considéré comme un signal de vente (croisement baissier).
        """)
        
        # Créons une simple visualisation pour illustrer les SMA
        st.image("https://www.investopedia.com/thmb/TphjF_jEcn3W6PbQHwi3QUoIOXY=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Moving_Average_Strategy_Sep_2020-01-a9654d6ca1f94cfb9c7963c3bd108f19.jpg", 
                 caption="Illustration des croisements de moyennes mobiles (source: Investopedia)")
        
        # RSI
        st.subheader("Indice de Force Relative (RSI)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.error("**RSI > 70** : Suracheté (signal potentiel de vente)")
        with col2:
            st.success("**RSI < 30** : Survendu (signal potentiel d'achat)")
        with col3:
            st.warning("**RSI entre 30 et 70** : Zone neutre")
        
        # MACD
        st.subheader("MACD (Moving Average Convergence Divergence)")
        st.markdown("""
        Le MACD est un indicateur de tendance qui montre la relation entre deux moyennes mobiles exponentielles.
        
        - **MACD > Signal Line** : Signal d'achat potentiel.
        - **MACD < Signal Line** : Signal de vente potentiel.
        - **Histogramme MACD** : Représente la différence entre le MACD et la ligne de signal. Les barres vertes indiquent une tendance haussière, les rouges une tendance baissière.
        """)
        
        # Bandes de Bollinger
        st.subheader("Bandes de Bollinger")
        st.markdown("""
        Les bandes de Bollinger utilisent l'écart-type pour créer une enveloppe autour du prix.
        
        - **Prix proche de la bande supérieure** : Potentiellement suracheté.
        - **Prix proche de la bande inférieure** : Potentiellement survendu.
        - **Rétrécissement des bandes** : Indique une faible volatilité, souvent avant un mouvement significatif du prix.
        - **Élargissement des bandes** : Indique une augmentation de la volatilité.
        """)
        
        # Stochastique
        st.subheader("Stochastique")
        st.markdown("""
        L'oscillateur stochastique compare le prix de clôture actuel à une fourchette de prix sur une période donnée.
        
        - **Stochastique K < 20 et en hausse** : Signal d'achat potentiel.
        - **Stochastique K > 80 et en baisse** : Signal de vente potentiel.
        """)
        
        # ATR
        st.subheader("ATR (Average True Range)")
        st.markdown("""
        L'ATR mesure la volatilité du marché.
        
        - **ATR élevé** : Forte volatilité.
        - **ATR faible** : Faible volatilité.
        - **Application** : Utilisé dans notre application pour calculer les stop loss et la taille des positions.
        """)
    
    # Tab 4: Analyse des actualités
    with guide_tabs[3]:
        st.header("Analyse des actualités et sentiment")
        
        st.subheader("Score de sentiment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("**Score > 0.2** : Sentiment positif, signal d'achat potentiel")
        with col2:
            st.error("**Score < -0.2** : Sentiment négatif, signal de vente potentiel")
        with col3:
            st.warning("**Score entre -0.2 et 0.2** : Sentiment neutre")
        
        st.subheader("Interprétation des actualités")
        st.markdown("""
        Les actualités sont listées avec leur score de sentiment correspondant (vert pour positif, rouge pour négatif). 
        
        Exemples d'actualités et leur interprétation:
        
        🟢 **"AAPL dépasse les attentes pour le T3"** [0.45] : Très positif, pourrait soutenir une hausse du prix
        
        🟡 **"Nouveau produit lancé par MSFT avec des projections modérées"** [0.12] : Légèrement positif, impact limité
        
        🔴 **"TSLA rappelle 20 000 véhicules pour défaut critique"** [-0.62] : Très négatif, risque de baisse du prix
        
        Utilisez ces informations pour comprendre les facteurs pouvant influencer le prix à court terme.
        """)
        
        st.subheader("Comment le sentiment est calculé")
        st.markdown("""
        Notre application utilise l'analyse de sentiment VADER (Valence Aware Dictionary and sEntiment Reasoner), une méthode spécialement conçue pour analyser le sentiment des textes sur les médias sociaux et les actualités financières.
        
        Le processus comprend:
        1. Récupération des actualités récentes via l'API Finnhub
        2. Analyse du titre et de la description de chaque article
        3. Calcul d'un score composite (70% pour le titre, 30% pour la description)
        4. Moyenne des scores pour obtenir un sentiment global
        """)
    
    # Tab 5: Analyse fondamentale
    with guide_tabs[4]:
        st.header("Analyse fondamentale")
        
        # PER
        st.subheader("PER (Price-to-Earnings Ratio)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("**PER bas (< 25)** : Potentiellement sous-évalué")
        with col2:
            st.error("**PER élevé (> 25)** : Potentiellement surévalué")
        
        st.markdown("""
        Le PER compare le prix de l'action aux bénéfices par action. Il indique combien les investisseurs sont prêts à payer pour chaque dollar de bénéfice.
        
        **Interprétation** : Un PER bas peut être un signal d'achat, mais doit être comparé à la moyenne du secteur. Par exemple, les entreprises technologiques ont tendance à avoir des PER plus élevés que les banques.
        """)
        
        # ROE
        st.subheader("ROE (Return on Equity)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("**ROE élevé (> 10%)** : Bonne rentabilité")
        with col2:
            st.error("**ROE faible (< 10%)** : Rentabilité potentiellement problématique")
        
        st.markdown("""
        Le ROE mesure la rentabilité d'une entreprise par rapport à ses capitaux propres. Il indique l'efficacité avec laquelle une entreprise utilise ses fonds propres pour générer des bénéfices.
        
        **Interprétation** : Un ROE élevé est généralement un bon signe pour l'entreprise, indiquant une bonne efficacité opérationnelle.
        """)
        
        # Ratio Dette/Capitaux propres
        st.subheader("Ratio Dette/Capitaux propres (D/E)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("**D/E < 1.5** : Niveau d'endettement raisonnable")
        with col2:
            st.error("**D/E > 1.5** : Niveau d'endettement potentiellement élevé")
        
        st.markdown("""
        Le ratio D/E mesure le niveau d'endettement d'une entreprise par rapport à ses capitaux propres. Il indique la proportion de financement provenant de la dette par rapport aux capitaux propres.
        
        **Interprétation** : Un ratio D/E élevé peut indiquer un risque financier plus important, surtout en période de taux d'intérêt élevés ou de ralentissement économique.
        """)
    
    # Tab 6: Signaux de trading
    with guide_tabs[5]:
        st.header("Signaux de trading et leur interprétation")
        
        st.subheader("Types de signaux")
        st.markdown("""
        L'application génère plusieurs types de signaux basés sur différentes analyses :
        
        - **Signal technique** : Basé uniquement sur les indicateurs techniques.
        - **Signal d'actualités** : Basé sur l'analyse du sentiment des actualités récentes.
        - **Signal fondamental** : Basé sur les ratios financiers de l'entreprise.
        - **Signal macro** : Basé sur des indicateurs macroéconomiques comme le VIX et le S&P 500.
        """)
        
        st.subheader("Signal final")
        st.markdown("""
        Le signal final combine les signaux individuels :
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🟢🟢🟢 strong_buy** : Tous les signaux sont positifs")
            st.markdown("**🟢 long/buy** : Signal d'achat")
        with col2:
            st.markdown("**🟡 hold** : Signal neutre")
        with col3:
            st.markdown("**🔴 short/sell** : Signal de vente")
            st.markdown("**🔴🔴🔴 strong_sell** : Tous les signaux sont négatifs")
        
        st.subheader("Tableau de bord")
        st.markdown("""
        Le tableau de bord présente une vue synthétique de plusieurs actifs avec leurs indicateurs clés et signaux. 
        
        Les émojis colorés (🟢🟡🔴) permettent de visualiser rapidement les signaux pour chaque facteur d'analyse:
        - Premier emoji : Signal technique
        - Deuxième emoji : Signal basé sur les actualités
        - Troisième emoji : Signal fondamental
        
        Exemple: 🟢🟡🔴 signifie technique positif, actualités neutres, fondamentaux négatifs.
        """)
        
        st.subheader("Exemple d'interprétation")
        st.markdown("""
        Voici comment interpréter les différentes combinaisons de signaux :
        
        **🟢🟢🟢 (strong_buy)** : Tous les facteurs sont alignés positivement. C'est généralement le moment idéal pour établir une position longue avec une taille de position plus importante.
        
        **🟢🟢🟡 ou 🟢🟡🟢** : Deux facteurs positifs, un neutre. Signal d'achat avec une confiance modérée.
        
        **🟢🔴🟡** : Signaux contradictoires. Le signal technique est positif, mais les actualités sont négatives. Dans ce cas, considérez d'autres facteurs ou attendez une meilleure configuration.
        
        **🔴🔴🔴 (strong_sell)** : Tous les facteurs sont alignés négativement. Bon candidat pour une position courte ou pour vendre des positions existantes.
        """)
    
    # Tab 7: Backtest
    with guide_tabs[6]:
        st.header("Comprendre le backtest")
        
        st.subheader("Qu'est-ce qu'un backtest ?")
        st.markdown("""
        Un backtest est une simulation qui applique votre stratégie de trading à des données historiques pour voir comment elle aurait performé dans le passé. C'est un outil essentiel pour évaluer l'efficacité d'une stratégie avant de l'appliquer avec de l'argent réel.
        """)
        
        st.subheader("Métriques clés du backtest")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **Retour Cumulé** : Performance totale de la stratégie sur la période testée.
            - **Taux de réussite** : Pourcentage de trades gagnants.
            - **Alpha** : Différence entre le retour de la stratégie et celui du marché (Buy & Hold).
            """)
        with col2:
            st.markdown("""
            - **Nombre de signaux** : Combien de trades la stratégie a généré.
            - **Ratio gain/perte** : Rapport entre le gain moyen des trades gagnants et la perte moyenne des trades perdants.
            - **Drawdown maximum** : Perte maximale subie pendant la période de test.
            """)
        
        st.subheader("Optimisation des paramètres")
        st.markdown("""
        L'onglet Backtest permet de tester différentes combinaisons de paramètres pour trouver ceux qui auraient généré les meilleurs résultats historiques.
        
        - **Meilleure combinaison** : Celle qui maximise l'alpha tout en maintenant un drawdown acceptable.
        - **Attention** : L'optimisation excessive peut mener au "curve fitting", où les paramètres sont trop adaptés aux données passées et peuvent mal performer sur des données futures.
        
        ⚠️ **Important** : Les performances passées ne garantissent pas les résultats futurs. Utilisez les backtests comme un guide, pas comme une garantie.
        """)
        
        st.subheader("Comment utiliser l'onglet Backtest")
        st.markdown("""
        1. Entrez le ticker que vous souhaitez tester
        2. Sélectionnez la période de backtest (plus longue pour des résultats plus robustes)
        3. Sélectionnez les paramètres à tester (SMA court, SMA long, RSI)
        4. Définissez les plages de valeurs à tester pour chaque paramètre
        5. Cliquez sur "Lancer le backtest"
        6. Analysez les résultats et identifiez les paramètres optimaux
        7. Faites attention aux combinaisons qui donnent de bons résultats sur différentes périodes
        """)
    
    # Tab 8: Graphiques
    with guide_tabs[7]:
        st.header("Guide des graphiques")
        
        st.subheader("Graphique de prix et signaux")
        st.markdown("""
        - **Ligne bleue** : Prix de clôture.
        - **Lignes pointillées** : Moyennes mobiles (orange pour court terme, rouge pour long terme).
        - **Triangles verts** : Signaux d'achat.
        - **Triangles rouges** : Signaux de vente.
        - **Zone colorée** (si présente) : Bandes de Bollinger.
        """)
        
        st.subheader("Graphique des indicateurs techniques")
        st.markdown("""
        - **Panneau supérieur** : Prix avec moyennes mobiles et bandes de Bollinger.
        - **Deuxième panneau** : MACD et ligne de signal, avec histogramme.
        - **Troisième panneau** : RSI avec lignes de survente/surachat à 30 et 70.
        - **Panneau inférieur** : Volume de transactions.
        """)
        
        st.subheader("Graphique de performance cumulée")
        st.markdown("""
        - **Ligne verte** : Performance de la stratégie.
        - **Ligne grise pointillée** : Performance du Buy & Hold (référence du marché).
        
        Ce graphique vous permet de comparer visuellement la performance de votre stratégie par rapport à une simple stratégie d'achat et de conservation.
        """)
        
        st.subheader("Comment interpréter les divergences")
        st.markdown("""
        Les divergences entre le prix et les indicateurs peuvent fournir des signaux puissants:
        
        - **Divergence bullish** : Le prix fait un nouveau plus bas, mais l'indicateur (RSI ou MACD) ne confirme pas ce plus bas. Potentiel signal de retournement à la hausse.
        
        - **Divergence bearish** : Le prix fait un nouveau plus haut, mais l'indicateur ne confirme pas ce plus haut. Potentiel signal de retournement à la baisse.
        
        Cherchez ces schémas sur les graphiques pour des opportunités de trading supplémentaires.
        """)
    
    # Tab 9: Gestion des risques
    with guide_tabs[8]:
        st.header("Gestion des risques")
        
        st.subheader("Taille de position")
        st.markdown("""
        L'application calcule la taille de position optimale basée sur :
        - **Capital disponible** : Montant total à investir.
        - **Pourcentage de risque** : Pourcentage du capital que vous êtes prêt à risquer sur chaque trade.
        - **Stop Loss** : Niveau de prix auquel vous fermeriez une position perdante pour limiter les pertes.
        
        **Formule simplifiée** :  
        Nombre d'actions = (Capital × % risque) ÷ Distance au stop loss par action
        
        Par exemple, avec 10 000€ de capital, 1% de risque et un stop loss à 2€ sous le prix d'entrée, vous achèteriez 50 actions (10 000€ × 1% ÷ 2€ = 50).
        """)
        
        st.subheader("Stop Loss basé sur l'ATR")
        st.markdown("""
        Le stop loss est calculé en utilisant l'ATR pour s'adapter à la volatilité du marché :
        - **Pour les positions longues** : Prix d'entrée moins 2 × ATR.
        - **Pour les positions courtes** : Prix d'entrée plus 2 × ATR.
        
        Cette méthode est plus sophistiquée qu'un stop loss fixe car elle tient compte de la volatilité de l'actif. Un actif plus volatil aura un stop loss plus éloigné.
        """)
        
        st.subheader("Règles de gestion des risques recommandées")
        st.markdown("""
        1. **Ne risquez jamais plus de 1-2% du capital sur un seul trade**
        2. **Diversifiez vos positions** (utilisez le tableau de bord multi-tickers)
        3. **Respectez toujours vos stop loss**
        4. **Prenez des profits partiels** lorsque les objectifs de prix sont atteints
        5. **Gardez une réserve de liquidités** pour les opportunités futures
        6. **Évitez de trader pendant les annonces majeures** (résultats trimestriels, décisions de taux d'intérêt, etc.)
        7. **Documentez vos trades** pour apprendre de vos erreurs et succès
        
        ⚠️ **Important** : Aucune stratégie n'est parfaite. Même avec la meilleure gestion des risques, des pertes sont inévitables. L'objectif est de s'assurer que les gains des trades gagnants dépassent les pertes des trades perdants sur le long terme.
        """)
    
    # Tab 10: FAQ & Glossaire
    with guide_tabs[9]:
        st.header("FAQ & Glossaire")
        
        st.subheader("Foire Aux Questions")
        
        with st.expander("Que signifie 'tenir compte du contexte macro' ?"):
            st.markdown("""
            Le contexte macroéconomique fait référence aux conditions économiques globales qui peuvent influencer tous les marchés. 
            
            Notre application analyse des indicateurs comme le VIX (indice de volatilité) et les tendances du S&P 500 pour évaluer le sentiment général du marché.
            
            En période de forte volatilité (VIX élevé) ou de tendance baissière du marché global, même les actions techniquement solides peuvent sous-performer.
            """)
        
        with st.expander("Comment interpréter des signaux contradictoires ?"):
            st.markdown("""
            Des signaux contradictoires sont courants (par exemple, signal technique positif mais fondamentaux négatifs). Dans ce cas :
            - Si vous êtes un trader à court terme, privilégiez les signaux techniques et les actualités.
            - Si vous êtes un investisseur à long terme, donnez plus de poids aux fondamentaux.
            
            L'application neutralise automatiquement le signal final en cas de contradiction majeure.
            """)
        
        with st.expander("Quelle est la différence entre un signal 'long' et 'strong_buy' ?"):
            st.markdown("""
            - Un signal **long** est généralement basé uniquement sur l'analyse technique.
            - Un signal **strong_buy** indique que tous les facteurs (technique, actualités, fondamentaux, macro) sont alignés positivement.
            
            Un **strong_buy** offre généralement une meilleure probabilité de succès et pourrait justifier une position plus importante.
            """)
        
        with st.expander("Comment choisir la période d'analyse ?"):
            st.markdown("""
            - Pour le day trading ou swing trading à court terme : 1mo ou 3mo.
            - Pour les investissements à moyen terme : 6mo ou 1y.
            - Pour l'analyse des tendances longues : 1y ou 2y.
            
            Adaptez la période à votre horizon d'investissement personnel.
            """)
        
        with st.expander("Est-ce que les performances passées garantissent les résultats futurs ?"):
            st.markdown("""
            Non. Les backtests sont utiles pour évaluer une stratégie, mais les marchés évoluent constamment. 
            
            Utilisez les résultats comme une indication, pas comme une garantie. Les conditions de marché changent, et une stratégie qui a bien fonctionné par le passé peut moins bien fonctionner à l'avenir.
            """)
        
        with st.expander("Que faire si l'API ne récupère pas les actualités ?"):
            st.markdown("""
            L'application utilise des données simulées lorsque l'API est indisponible. 
            
            Ces données servent à illustrer la fonctionnalité mais ne doivent pas être utilisées pour prendre des décisions réelles. 
            
            Vous verrez un avertissement lorsque des données simulées sont utilisées.
            """)
        
        st.subheader("Glossaire des termes de trading")
        
        glossary = {
            "Alpha": "Performance supplémentaire d'un investissement par rapport à un indice de référence.",
            "Beta": "Mesure de la volatilité d'un actif par rapport au marché.",
            "Drawdown": "Baisse en pourcentage depuis un sommet jusqu'à un creux avant d'atteindre un nouveau sommet.",
            "Momentum": "Vitesse à laquelle le prix d'un actif change.",
            "Volatilité": "Amplitude des variations de prix d'un actif sur une période donnée.",
            "Volume": "Nombre d'unités d'un actif échangées pendant une période spécifique.",
            "Capitalisation boursière": "Valeur totale des actions en circulation d'une entreprise.",
            "Dividende": "Partie des bénéfices d'une entreprise versée aux actionnaires.",
            "Support": "Niveau de prix où la demande est suffisamment forte pour empêcher le prix de baisser davantage.",
            "Résistance": "Niveau de prix où l'offre est suffisamment forte pour empêcher le prix de monter davantage.",
            "Gap": "Écart entre le prix de clôture d'une journée et le prix d'ouverture de la journée suivante.",
            "Breakout": "Mouvement de prix qui dépasse un niveau de support ou de résistance établi.",
            "Retracement": "Mouvement temporaire dans la direction opposée à la tendance principale.",
            "Double Top/Bottom": "Formation graphique en forme de M (double sommet) ou W (double creux) indiquant un potentiel renversement de tendance."
        }
        
        # Afficher le glossaire sous forme de tableau
        glossary_df = pd.DataFrame({"Terme": glossary.keys(), "Définition": glossary.values()})
        st.dataframe(glossary_df, use_container_width=True)

# *** Partie 10: Application Streamlit ***

def create_streamlit_app():
    # Titre et introduction
    st.title("📊 Stratégie de Trading Multi-Facteurs")
    
    # Dans la section sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres")
    
    # Input pour le ticker individuel
    ticker_input = st.sidebar.text_input("Ticker", "AAPL").upper()
    
    # Définir la liste des tickers par défaut
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    predefined_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "DIS", "JPM", "BA", "KO", "PEP", "WMT", "V", "MA"]
    
    # Ajouter un champ pour permettre aux utilisateurs d'ajouter leur propre ticker
    new_ticker = st.sidebar.text_input("Ajouter un nouveau ticker").upper()
    add_ticker_button = st.sidebar.button("Ajouter au tableau de bord")
    
    # Créer une liste personnalisée en session
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = predefined_tickers.copy()
    
    # Ajouter le nouveau ticker à la liste si demandé
    if add_ticker_button and new_ticker and new_ticker not in st.session_state.custom_tickers:
        st.session_state.custom_tickers.append(new_ticker)
        st.sidebar.success(f"Ticker {new_ticker} ajouté avec succès!")
    
    # Sélection multiple pour les tickers (tableau de bord)
    selected_tickers = st.sidebar.multiselect(
        "Tickers pour le tableau de bord",
        options=st.session_state.custom_tickers,
        default=default_tickers
    )
    
    # Bouton pour réinitialiser la liste des tickers personnalisés
    if st.sidebar.button("Réinitialiser la liste des tickers"):
        st.session_state.custom_tickers = predefined_tickers.copy()
        st.sidebar.info("Liste des tickers réinitialisée.")
    
    # Paramètres d'analyse
    st.sidebar.subheader("Paramètres d'analyse")
    period = st.sidebar.selectbox("Période d'analyse", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    # Options avancées dans un expander
    with st.sidebar.expander("Options avancées"):
        sma_short = st.slider("SMA Court Terme", 5, 50, DEFAULT_SMA_SHORT)
        sma_long = st.slider("SMA Long Terme", 20, 200, DEFAULT_SMA_LONG)
        rsi_window = st.slider("Fenêtre RSI", 7, 21, DEFAULT_RSI_WINDOW)
        sentiment_days = st.slider("Jours d'actualités", 3, 14, SENTIMENT_DAYS)
        
        capital = st.number_input("Capital de trading (€)", min_value=1000, max_value=1000000, value=10000, step=1000)
        risk_pct = st.slider("% de risque par trade", 0.5, 5.0, 1.0, 0.5)
    
    # Navigation par onglets
    tabs = st.tabs(["Analyse individuelle", "Tableau de bord", "Backtest", "Mode d'Emploi", "À propos"])
    
    # Onglet 1: Analyse individuelle
    with tabs[0]:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Analyse pour un ticker")
            ticker = st.text_input("Entrez un ticker", ticker_input)
            
            if st.button("Analyser", key="analyze_single"):
                # Conteneur pour les résultats
                results_container = st.container()
                
                with st.spinner(f"Analyse en cours pour {ticker}..."):
                    # Récupération et traitement des données
                    data = get_price_data(ticker, period=period)
                    
                    if data.empty:
                        st.error(f"Aucune donnée disponible pour {ticker}")
                    else:
                        # Traitement des données
                        data = add_sma(data, short=sma_short, long=sma_long)
                        data = compute_rsi(data, window=rsi_window)
                        data = compute_macd(data)
                        data = compute_bollinger_bands(data)
                        data = compute_atr(data)
                        data = generate_signals(data)
                        data = evaluate_signals(data)
                        
                        # Récupérer les actualités et le sentiment
                        news_df = get_recent_news(ticker, days=sentiment_days)
                        sentiment_score = news_df["sentiment_score"].mean() if not news_df.empty else None
                        
                        # Données fondamentales
                        fundamentals = get_fundamentals(ticker)
                        fund_signal = "buy" if (
                            fundamentals["per"] is not None and 
                            fundamentals["per"] < PER_THRESHOLD and 
                            fundamentals["roe"] is not None and 
                            fundamentals["roe"] > ROE_THRESHOLD
                        ) else "sell" if (
                            fundamentals["per"] is not None and 
                            fundamentals["per"] > PER_THRESHOLD * 1.5 or 
                            fundamentals["roe"] is not None and 
                            fundamentals["roe"] < ROE_THRESHOLD / 2
                        ) else "hold"
                        
                        # Données macro
                        macro_data = get_macro_indicators()
                        macro_signal = get_macro_signal(macro_data)
                        
                        # Génération du signal final
                        data = generate_final_signal(
                            data, 
                            ticker, 
                            {ticker: sentiment_score},
                            {'signal': fund_signal},
                            macro_signal
                        )
                        
                        # Backtest
                        data = backtest_signals(data)
                        backtest_stats = resume_backtest(data)
                        
                        # Calcul du sizing de position
                        position_info = calculate_position_size(data, ticker, capital=capital, risk_pct=risk_pct)
                        
                        # Affichage des résultats
                        with results_container:
                            # Signal et récapitulatif
                            final_signal = data['Final_Signal'].iloc[-1]
                            signal_color = {
                                'long': 'green', 'short': 'red', 'hold': 'orange',
                                'strong_buy': 'darkgreen', 'strong_sell': 'darkred'
                            }.get(final_signal, 'blue')
                            
                            st.markdown(f"### Signal pour {ticker}: <span style='color:{signal_color}'>{final_signal.upper()}</span>", unsafe_allow_html=True)
                            
                            # Affichage des facteurs qui ont conduit au signal
                            factors_col1, factors_col2 = st.columns(2)
                            
                            with factors_col1:
                                st.markdown("#### Facteurs d'analyse")
                                st.markdown(f"🔹 **Technique**: {data['Tech_Signal'].iloc[-1]}")
                                st.markdown(f"🔹 **Actualités**: {data['News_Signal'].iloc[-1]}")
                                st.markdown(f"🔹 **Fondamental**: {data['Fund_Signal'].iloc[-1]}")
                                st.markdown(f"🔹 **Macro**: {data['Macro_Signal'].iloc[-1]}")
                            
                            with factors_col2:
                                st.markdown("#### Indicateurs clés")
                                st.markdown(f"📈 **Prix actuel**: ${data['Close'].iloc[-1]:.2f}")
                                st.markdown(f"📊 **RSI**: {data['RSI'].iloc[-1]:.1f}")
                                st.markdown(f"📰 **Sentiment**: {sentiment_score:.2f}" if sentiment_score is not None else "📰 **Sentiment**: N/A")
                                st.markdown(f"💵 **PER**: {fundamentals['per']:.1f}" if fundamentals['per'] is not None else "💵 **PER**: N/A")
                            
                            # Sizing de position
                            st.markdown("#### Position recommandée")
                            pos_col1, pos_col2 = st.columns(2)
                            
                            with pos_col1:
                                st.metric("Nombre d'actions", f"{position_info['num_shares']}", 
                                          f"€{position_info['position_cost']:.0f}")
                                
                            with pos_col2:
                                st.metric("Stop Loss", f"€{position_info['stop_loss_price']:.2f}", 
                                          f"{-position_info['stop_loss_pct']:.1f}%" if data['Signal'].iloc[-1] == "long" else f"{position_info['stop_loss_pct']:.1f}%")
                            
                            # Graphique de la stratégie
                            st.subheader("Stratégie de Trading")
                            fig = plot_trading_strategy(data, ticker)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Indicateurs techniques
                            with st.expander("Indicateurs techniques détaillés"):
                                fig_tech = plot_technical_indicators(data, ticker)
                                st.plotly_chart(fig_tech, use_container_width=True)
                            
                            # Résultats du backtest
                            st.subheader("Résultats du Backtest")
                            if backtest_stats:
                                bt_col1, bt_col2, bt_col3 = st.columns(3)
                                
                                with bt_col1:
                                    success_rate_long = backtest_stats.get("success_rate_long")
                                    if success_rate_long is not None:
                                        st.metric("Succès signaux Long", f"{success_rate_long:.1f}%", 
                                                 f"{backtest_stats.get('nb_longs')} signaux")
                                
                                with bt_col2:
                                    success_rate_short = backtest_stats.get("success_rate_short")
                                    if success_rate_short is not None:
                                        st.metric("Succès signaux Short", f"{success_rate_short:.1f}%", 
                                                 f"{backtest_stats.get('nb_shorts')} signaux")
                                
                                with bt_col3:
                                    final_return = backtest_stats.get("final_return")
                                    market_return = backtest_stats.get("market_return")
                                    if final_return is not None and market_return is not None:
                                        alpha = final_return - market_return
                                        st.metric("Alpha", f"{alpha:.2%}", 
                                                 f"vs. {market_return:.2%} (Buy & Hold)")
                            
                            # Graphique des performances cumulées
                            if 'Cumulative_Strategy_Return' in data.columns:
                                st.subheader("Performance Cumulée")
                                perf_fig = go.Figure()
                                
                                perf_fig.add_trace(go.Scatter(
                                    x=data.index[-120:],
                                    y=data['Cumulative_Strategy_Return'][-120:] * 100,
                                    mode='lines',
                                    name='Stratégie',
                                    line=dict(color='green', width=2)
                                ))
                                
                                perf_fig.add_trace(go.Scatter(
                                    x=data.index[-120:],
                                    y=data['Cumulative_Market_Return'][-120:] * 100,
                                    mode='lines',
                                    name='Buy & Hold',
                                    line=dict(color='gray', width=1.5, dash='dot')
                                ))
                                
                                perf_fig.update_layout(
                                    title="Performance cumulée (%)",
                                    xaxis_title="Date",
                                    yaxis_title="Rendement (%)",
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(perf_fig, use_container_width=True)
                            
                            # Afficher les actualités récentes
                            if not news_df.empty:
                                st.subheader("Actualités récentes")
                                for i, (_, row) in enumerate(news_df[:5].iterrows()):
                                    sentiment_color = "green" if row["sentiment_score"] > 0.05 else ("red" if row["sentiment_score"] < -0.05 else "gray")
                                    st.markdown(f"**{row['title']}** - <span style='color:{sentiment_color}'>[{row['sentiment_score']:.2f}]</span>", unsafe_allow_html=True)
                                    st.markdown(f"*{row['description'][:200]}...*")
                                    st.markdown(f"[Lien]({row['url']})")
                                    if i < 4:
                                        st.markdown("---")
        
        with col2:
            # Cette colonne est utilisée pour l'affichage des résultats
            st.info("👈 Entrez un ticker et cliquez sur 'Analyser' pour voir les résultats détaillés")

    # Onglet 2: Tableau de bord
    with tabs[1]:
        st.subheader("Tableau de bord multi-tickers")
        
        if st.button("Générer le tableau de bord", key="generate_dashboard"):
            if not selected_tickers:
                st.warning("Veuillez sélectionner au moins un ticker dans la barre latérale.")
            else:
                # Conteneur pour les résultats
                dashboard_container = st.container()
                
                with st.spinner("Génération du tableau de bord en cours..."):
                    # Récupération et traitement des données pour tous les tickers
                    resultats = {}
                    sentiments = {}
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(selected_tickers):
                        try:
                            # Actualiser la barre de progression
                            progress = (i + 1) / len(selected_tickers)
                            progress_bar.progress(progress)
                            
                            # Récupérer les actualités
                            news_df = get_recent_news(ticker, days=sentiment_days)
                            if not news_df.empty:
                                sentiments[ticker] = news_df["sentiment_score"].mean()
                            else:
                                sentiments[ticker] = None
                            
                            # Récupérer et traiter les données
                            data = get_price_data(ticker, period=period)
                            if not data.empty:
                                data = add_sma(data, short=sma_short, long=sma_long)
                                data = compute_rsi(data, window=rsi_window)
                                data = compute_macd(data)
                                data = compute_bollinger_bands(data)
                                data = compute_atr(data)
                                data = generate_signals(data)
                                data = evaluate_signals(data)
                                resultats[ticker] = data
                        except Exception as e:
                            st.error(f"Erreur pour {ticker}: {str(e)}")
                    
                    # Supprimer la barre de progression
                    progress_bar.empty()
                    
                    # Générer le tableau synthétique
                    tableau = tableau_synthetique(resultats, sentiments)
                    
                    # Afficher le tableau
                    with dashboard_container:
                        if not tableau.empty:
                            st.dataframe(tableau, use_container_width=True, height=400)
                            
                            # Sélection d'un ticker pour affichage détaillé
                            selected_ticker = st.selectbox("Sélectionner un ticker pour détails", tableau["Ticker"].tolist())
                            
                            if selected_ticker and selected_ticker in resultats:
                                with st.expander(f"Détails pour {selected_ticker}", expanded=True):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Graphique de trading
                                        fig = plot_trading_strategy(resultats[selected_ticker], selected_ticker)
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col2:
                                        # Performance et backtest
                                        resultats[selected_ticker] = backtest_signals(resultats[selected_ticker])
                                        stats = stats_strategie(resultats[selected_ticker])
                                        
                                        st.subheader("Statistiques")
                                        stats_df = pd.DataFrame([{
                                            "Retour cumulé": f"{stats['Retour Cumulé']*100:.2f}%" if stats['Retour Cumulé'] is not None else "N/A",
                                            "Taux de réussite": f"{stats['Taux de gain']*100:.2f}%" if 'Taux de gain' in stats else "N/A",
                                            "Trades gagnants": stats.get('Trades gagnants', 'N/A'),
                                            "Trades perdants": stats.get('Trades perdants', 'N/A'),
                                            "Dernier retour": f"{stats['Dernier retour']*100:.2f}%" if stats.get('Dernier retour') is not None else "N/A"
                                        }])
                                        
                                        st.dataframe(stats_df.T, use_container_width=True)
                        else:
                            st.warning("Aucune donnée disponible pour générer le tableau de bord.")
    
    # Onglet 3: Backtest
    with tabs[2]:
        st.subheader("Backtest et optimisation")
        
        backtest_col1, backtest_col2 = st.columns([1, 2])
        
        with backtest_col1:
            backtest_ticker = st.text_input("Ticker pour backtest", ticker_input)
            backtest_period = st.selectbox("Période de backtest", ["6mo", "1y", "2y", "5y", "10y"], index=1)
            
            # Paramètres pour les tests
            st.subheader("Paramètres à tester")
            
            test_sma_short = st.checkbox("Tester SMA court terme", value=True)
            if test_sma_short:
                sma_short_min = st.number_input("SMA court min", 5, 30, 10, 5)
                sma_short_max = st.number_input("SMA court max", 15, 100, 30, 5)
                sma_short_step = st.number_input("SMA court pas", 1, 10, 5, 1)
            
            test_sma_long = st.checkbox("Tester SMA long terme", value=True)
            if test_sma_long:
                sma_long_min = st.number_input("SMA long min", 20, 100, 40, 10)
                sma_long_max = st.number_input("SMA long max", 50, 200, 100, 10)
                sma_long_step = st.number_input("SMA long pas", 5, 20, 10, 5)
            
            test_rsi = st.checkbox("Tester RSI", value=False)
            if test_rsi:
                rsi_min = st.number_input("RSI min", 5, 20, 10, 1)
                rsi_max = st.number_input("RSI max", 10, 30, 20, 1)
                rsi_step = st.number_input("RSI pas", 1, 5, 2, 1)
            
            if st.button("Lancer le backtest"):
                with st.spinner(f"Exécution du backtest pour {backtest_ticker}..."):
                    # Récupération des données
                    backtest_data = get_price_data(backtest_ticker, period=backtest_period)
                    
                    if backtest_data.empty:
                        st.error(f"Aucune donnée disponible pour {backtest_ticker}")
                    else:
                        # Générer les combinaisons de paramètres
                        param_combinations = []
                        
                        if test_sma_short:
                            sma_short_values = range(sma_short_min, sma_short_max + 1, sma_short_step)
                        else:
                            sma_short_values = [DEFAULT_SMA_SHORT]
                            
                        if test_sma_long:
                            sma_long_values = range(sma_long_min, sma_long_max + 1, sma_long_step)
                        else:
                            sma_long_values = [DEFAULT_SMA_LONG]
                            
                        if test_rsi:
                            rsi_values = range(rsi_min, rsi_max + 1, rsi_step)
                        else:
                            rsi_values = [DEFAULT_RSI_WINDOW]
                        
                        # Construire les combinaisons valides (SMA court < SMA long)
                        for short in sma_short_values:
                            for long in sma_long_values:
                                if short < long:  # Condition valide
                                    for rsi in rsi_values:
                                        param_combinations.append({
                                            'sma_short': short,
                                            'sma_long': long,
                                            'rsi': rsi
                                        })
                        
                        # Exécuter le backtest pour chaque combinaison
                        results = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        total_combinations = len(param_combinations)
                        
                        for i, params in enumerate(param_combinations):
                            # Mise à jour de la barre de progression
                            progress = (i + 1) / total_combinations
                            progress_bar.progress(progress)
                            
                            # Préparation des données avec ces paramètres
                            df_copy = backtest_data.copy()
                            df_copy = add_sma(df_copy, short=params['sma_short'], long=params['sma_long'])
                            df_copy = compute_rsi(df_copy, window=params['rsi'])
                            df_copy = compute_macd(df_copy)
                            df_copy = generate_signals(df_copy)
                            df_copy = evaluate_signals(df_copy)
                            df_copy = backtest_signals(df_copy)
                            
                            # Calcul des métriques de performance
                            if 'Cumulative_Strategy_Return' in df_copy.columns and len(df_copy) > 0:
                                strat_return = df_copy['Cumulative_Strategy_Return'].iloc[-1]
                                market_return = df_copy['Cumulative_Market_Return'].iloc[-1]
                                
                                # Nombre de signaux
                                nb_longs = len(df_copy[df_copy['Signal'] == "long"])
                                nb_shorts = len(df_copy[df_copy['Signal'] == "short"])
                                
                                # Taux de réussite
                                win_longs = len(df_copy[(df_copy['Signal'] == "long") & (df_copy['Return_5d'] > 0)])
                                win_shorts = len(df_copy[(df_copy['Signal'] == "short") & (df_copy['Return_5d'] < 0)])
                                
                                success_rate_long = win_longs / nb_longs * 100 if nb_longs > 0 else 0
                                success_rate_short = win_shorts / nb_shorts * 100 if nb_shorts > 0 else 0
                                
                                # Drawdown maximum
                                cumul_returns = df_copy['Cumulative_Strategy_Return']
                                rolling_max = cumul_returns.cummax()
                                drawdown = (cumul_returns - rolling_max) / (1 + rolling_max)
                                max_drawdown = drawdown.min()
                                
                                # Ajouter aux résultats
                                results.append({
                                    'SMA court': params['sma_short'],
                                    'SMA long': params['sma_long'],
                                    'RSI': params['rsi'],
                                    'Retour stratégie': strat_return * 100,  # en pourcentage
                                    'Retour marché': market_return * 100,    # en pourcentage
                                    'Alpha': (strat_return - market_return) * 100,
                                    'Max Drawdown': max_drawdown * 100,
                                    'Nb signaux long': nb_longs,
                                    'Nb signaux short': nb_shorts,
                                    'Taux réussite long': success_rate_long,
                                    'Taux réussite short': success_rate_short
                                })
                        
                        # Convertir en DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Trier par performance (Alpha)
                        results_df = results_df.sort_values('Alpha', ascending=False)
                        
                        # Afficher les résultats
                        st.subheader("Résultats du backtest")
                        st.dataframe(results_df.style.format({
                            'Retour stratégie': '{:.2f}%',
                            'Retour marché': '{:.2f}%',
                            'Alpha': '{:.2f}%',
                            'Max Drawdown': '{:.2f}%',
                            'Taux réussite long': '{:.1f}%',
                            'Taux réussite short': '{:.1f}%'
                        }), use_container_width=True)
                        
                        # Afficher les meilleurs paramètres
                        if not results_df.empty:
                            best_params = results_df.iloc[0]
                            
                            st.success(f"""
                            **Meilleurs paramètres trouvés:**
                            - SMA Court: {int(best_params['SMA court'])}
                            - SMA Long: {int(best_params['SMA long'])}
                            - RSI: {int(best_params['RSI'])}
                            - Alpha généré: {best_params['Alpha']:.2f}%
                            - Drawdown maximum: {best_params['Max Drawdown']:.2f}%
                            """)
                            
                            # Appliquer les meilleurs paramètres et visualiser
                            best_df = backtest_data.copy()
                            best_df = add_sma(best_df, short=int(best_params['SMA court']), long=int(best_params['SMA long']))
                            best_df = compute_rsi(best_df, window=int(best_params['RSI']))
                            best_df = compute_macd(best_df)
                            best_df = generate_signals(best_df)
                            best_df = evaluate_signals(best_df)
                            best_df = backtest_signals(best_df)
                            
                            fig = plot_trading_strategy(best_df, backtest_ticker)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Graphique de comparaison des performances
                            perf_fig = go.Figure()
                                
                            perf_fig.add_trace(go.Scatter(
                                x=best_df.index,
                                y=best_df['Cumulative_Strategy_Return'] * 100,
                                mode='lines',
                                name='Stratégie optimisée',
                                line=dict(color='green', width=2)
                            ))
                            
                            perf_fig.add_trace(go.Scatter(
                                x=best_df.index,
                                y=best_df['Cumulative_Market_Return'] * 100,
                                mode='lines',
                                name='Buy & Hold',
                                line=dict(color='gray', width=1.5, dash='dot')
                            ))
                            
                            perf_fig.update_layout(
                                title="Performance cumulée avec paramètres optimisés (%)",
                                xaxis_title="Date",
                                yaxis_title="Rendement (%)",
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(perf_fig, use_container_width=True)
                            
        with backtest_col2:
            # Cette colonne est utilisée pour afficher les résultats du backtest
            st.info("👈 Configurez les paramètres du backtest et cliquez sur 'Lancer le backtest' pour optimiser votre stratégie")
    
    # Onglet 4: Mode d'Emploi (NOUVEAU)
    with tabs[3]:
        create_user_guide()
    
    # Onglet 5: À propos
    with tabs[4]:
        st.subheader("À propos de cette application")
        
        st.markdown("""
        ## Stratégie de Trading Multi-Facteurs avec Visualisation Interactive

        ### Pourquoi ce projet ?
        Les marchés financiers sont influencés par :
        - 📈 des indicateurs techniques (prix, tendances)
        - 📰 des événements d'actualité (décisions politiques, annonces d'entreprises)
        - 📊 des données fondamentales (santé financière des entreprises)

        Notre objectif est de construire une application complète d'aide à la décision pour détecter les bons moments pour acheter ou vendre à découvert une action.

        ### Ce que notre programme fait
        Notre outil :
        1. Récupère automatiquement les données boursières d'une action
        2. Calcule plusieurs indicateurs techniques (SMA, RSI, MACD, Bollinger)
        3. Récupère et analyse les news récentes liées à l'action (via API)
        4. Évalue la solidité financière de l'entreprise à travers des ratios fondamentaux
        5. Génère un signal d'achat ou de short, selon l'analyse combinée
        6. Fait un backtest historique pour évaluer la stratégie
        7. Présente les résultats sur une interface web claire avec:
           - Résumé des signaux
           - Graphiques interactifs
           - Statistiques de performance

        ### À quels besoins ça répond ?
        - Comment automatiser une veille financière intelligente
        - Comment éviter les biais émotionnels dans le trading
        - Comment combiner analyse technique + fondamentale + actualités
        - Comment créer un outil intuitif qu'on pourrait utiliser chaque semaine

        ### Pistes pour aller plus loin
        - Ajouter la possibilité de tester plusieurs stratégies (choix des indicateurs à la volée)
        - Intégrer des données macroéconomiques supplémentaires
        - Connecter l'app à un compte de trading démo (pour tester en conditions réelles)
        - Ajouter un moteur de recommandation automatique d'actions à suivre

        ### En résumé
        Un outil complet, intelligent, interactif et pédagogique qui peut être utilisé chaque semaine par un investisseur pour prendre des décisions plus rationnelles et documentées.
        """)

# *** Point d'entrée principal de l'application ***

if __name__ == "__main__":
    create_streamlit_app()
