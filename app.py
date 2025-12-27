import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(page_title="Dynamic Pricing System", layout="wide")

st.title("⛽ Dynamic Pricing Recommendation System")
st.caption("ML-based pricing engine using XGBoost with business guardrails")

# ------------------------------------
# BUSINESS RULES
# ------------------------------------
MAX_DAILY_CHANGE = 0.05     # 5%
MIN_MARGIN = 0.03           # 3%
COMPETITIVE_LIMIT = 0.02   # 2%

# ------------------------------------
# LOAD DATA (NO UPLOAD)
# ------------------------------------
DATA_PATH = "oil.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ oil.csv not found. Please place it in the project root.")
        st.stop()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

df = load_data()

# ------------------------------------
# FEATURE ENGINEERING
# ------------------------------------
def feature_engineering(df):
    df = df.copy()

    df["comp_avg_price"] = df[["comp1_price","comp2_price","comp3_price"]].mean(axis=1)
    df["price_vs_comp"] = df["price"] - df["comp_avg_price"]
    df["price_vs_cost"] = df["price"] - df["cost"]

    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_7"] = df["price"].shift(7)

    df["price_ma_7"] = df["price"].rolling(7).mean()
    df["price_ma_14"] = df["price"].rolling(14).mean()

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month

    df.dropna(inplace=True)

    # Demand proxy
    df["demand_proxy"] = -1 * df["price_vs_comp"] + 0.5 * df["price_ma_7"]

    return df

df_fe = feature_engineering(df)

# ------------------------------------
# TRAIN MODEL
# ------------------------------------
@st.cache_resource
def train_model(df):
    FEATURES = [
        "price","cost","comp_avg_price",
        "price_vs_comp","price_vs_cost",
        "price_lag_1","price_lag_7",
        "price_ma_7","price_ma_14",
        "day_of_week","month"
    ]

    X = df[FEATURES]
    y = df["demand_proxy"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model, FEATURES

model, FEATURES = train_model(df_fe)

# ------------------------------------
# PRICE OPTIMIZATION
# ------------------------------------
def recommend_price(today, model):
    last_price = today["price"]
    cost = today["cost"]

    comp_avg = np.mean([
        today["comp1_price"],
        today["comp2_price"],
        today["comp3_price"]
    ])

    candidate_prices = np.linspace(
        last_price * (1 - MAX_DAILY_CHANGE),
        last_price * (1 + MAX_DAILY_CHANGE),
        20
    )

    best_price, best_profit = None, -np.inf

    for p in candidate_prices:
        if p < cost * (1 + MIN_MARGIN):
            continue
        if abs(p - comp_avg) / comp_avg > COMPETITIVE_LIMIT:
            continue

        row = {
            "price": p,
            "cost": cost,
            "comp_avg_price": comp_avg,
            "price_vs_comp": p - comp_avg,
            "price_vs_cost": p - cost,
            "price_lag_1": last_price,
            "price_lag_7": last_price,
            "price_ma_7": last_price,
            "price_ma_14": last_price,
            "day_of_week": today["day_of_week"],
            "month": today["month"]
        }

        X_pred = pd.DataFrame([row])[FEATURES]
        demand = model.predict(X_pred)[0]
        profit = (p - cost) * demand

        if profit > best_profit:
            best_profit = profit
            best_price = p

    return round(best_price, 2), round(best_profit, 2)

# ------------------------------------
# TODAY INPUTS
# ------------------------------------
st.sidebar.header("📥 Today's Inputs")

today_data = {
    "price": st.sidebar.number_input("Last Price", value=float(df["price"].iloc[-1])),
    "cost": st.sidebar.number_input("Today's Cost", value=float(df["cost"].iloc[-1])),
    "comp1_price": st.sidebar.number_input("Competitor 1 Price", value=float(df["comp1_price"].iloc[-1])),
    "comp2_price": st.sidebar.number_input("Competitor 2 Price", value=float(df["comp2_price"].iloc[-1])),
    "comp3_price": st.sidebar.number_input("Competitor 3 Price", value=float(df["comp3_price"].iloc[-1])),
    "day_of_week": st.sidebar.selectbox("Day of Week", list(range(7))),
    "month": st.sidebar.selectbox("Month", list(range(1,13)))
}

# ------------------------------------
# OUTPUT
# ------------------------------------
if st.sidebar.button("🚀 Recommend Price"):
    price, profit = recommend_price(today_data, model)

    st.success("Pricing Recommendation Generated")

    col1, col2 = st.columns(2)
    col1.metric("💰 Recommended Price", f"₹ {price}")
    col2.metric("📈 Expected Profit Index", profit)

    st.subheader("📄 Input Summary")
    st.json(today_data)

# ------------------------------------
# DATA PREVIEW
# ------------------------------------
with st.expander("📊 View Processed Data"):
    st.dataframe(df_fe.tail(20))
