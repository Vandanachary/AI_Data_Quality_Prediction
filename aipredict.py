import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# IBM CLOUD CONFIG
# --------------------------------------------------
API_KEY = os.getenv("IBM_API_KEY")
DEPLOYMENT_URL = "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/b5ef2044-9f2a-47f3-b45c-ffda62b4312d/predictions?version=2021-05-01"


# --------------------------------------------------
# IBM AUTH TOKEN
# --------------------------------------------------
def get_ibm_token():
    token_url = "https://iam.cloud.ibm.com/identity/token"
    payload = {
        "apikey": API_KEY,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }

    response = requests.post(token_url, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_data_quality(user_inputs):
    token = get_ibm_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input_data": [
            {
                "fields": [
                    "order_id",
                    "customer_id",
                    "currency",
                    "order_date",
                    "status",
                    "country_code",
                    "payment_method",
                    "fraud_flag",
                    "delivery_days",
                    "customer_age",
                    "product_category"
                ],
                "values": [user_inputs]
            }
        ]
    }

    response = requests.post(DEPLOYMENT_URL, json=payload, headers=headers)
    response.raise_for_status()

    result = response.json()
    return result["predictions"][0]["values"][0][0]


# --------------------------------------------------
# DATA CLEANING + ANOMALY DETECTION
# --------------------------------------------------
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    df.dropna(subset=["customer_id", "payment_method"], inplace=True)

    if "order_amount" in df.columns:
        df["order_amount"].fillna(df["order_amount"].median(), inplace=True)

        z_scores = np.abs(
            (df["order_amount"] - df["order_amount"].mean())
            / df["order_amount"].std()
        )
        df["anomaly"] = z_scores > 1.5
    else:
        df["anomaly"] = False

    df.drop_duplicates(inplace=True)

    return df


# --------------------------------------------------
# PLOT FUNCTION
# --------------------------------------------------
def create_order_amount_plot(df, title):
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=df["order_amount"],
            boxpoints="outliers",
            name="Order Amount"
        )
    )   

    fig.update_layout(
        title=title,
        yaxis_title="Order Amount",
        height=500
    )

    return fig


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="AI Data Quality Monitoring",
    layout="wide"
)

st.title("AI-Based Data Quality Monitoring System")

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Quality Dashboard"])

# --------------------------------------------------
# TAB 1: PREDICTION
# --------------------------------------------------
with tab1:
    st.subheader("Predict Data Quality Issues")

    with st.form("prediction_form"):
        order_id = st.number_input("Order ID", min_value=10000)
        customer_id = st.number_input("Customer ID", min_value=1000)
        currency = st.selectbox("Currency", ["USD", "INR", "EUR", "BTC"])
        order_date = st.date_input("Order Date")
        status = st.selectbox("Order Status", ["Completed", "Pending", "Cancelled"])
        country_code = st.text_input("Country Code")
        payment_method = st.selectbox(
            "Payment Method",
            ["Credit Card", "Debit Card", "PayPal", "Bitcoin"]
        )
        fraud_flag = st.selectbox("Fraud Flag", ["Yes", "No"])
        delivery_days = st.number_input("Delivery Days", min_value=1)
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100)
        product_category = st.selectbox(
            "Product Category",
            ["Electronics", "Clothing", "Automotive", "Toys"]
        )

        submitted = st.form_submit_button("Check Data Quality")

    if submitted:
        user_inputs = [
            order_id,
            customer_id,
            currency,
            str(order_date),
            status,
            country_code,
            payment_method,
            fraud_flag,
            delivery_days,
            customer_age,
            product_category
        ]

        try:
            prediction = predict_data_quality(user_inputs)
            st.success(f"Predicted Data Quality Issue: {prediction}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# TAB 2: DASHBOARD
# --------------------------------------------------
with tab2:
    st.subheader("Data Quality Dashboard")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    POSSIBLE_PATHS = [
        os.path.join(BASE_DIR, "orders_data_set_final.csv"),
        os.path.join(BASE_DIR, "..", "orders_data_set_final.csv"),
        os.path.join(os.getcwd(), "orders_data_set_final.csv")
    ]

    CSV_PATH = None
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            CSV_PATH = path
            break

    if CSV_PATH is None:
        st.error("orders_data_set_final.csv not found in expected locations.")
        st.write("Checked paths:")
        for p in POSSIBLE_PATHS:
            st.write(p)
        st.stop()

    df = load_and_clean_data(CSV_PATH)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Anomalies Detected", int(df["anomaly"].sum()))

    if "order_amount" in df.columns:
        fig = create_order_amount_plot(df, "Order Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")
    st.dataframe(df[df["anomaly"]])