import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# 💡 Dark Mode Toggle and Theme Styling
dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #111 !important;
        color: #eee !important;
    }
    .block-container {
        color: #eee !important;
    }
    .stDataFrame, .css-1v0mbdj {
        background-color: #1e1e1e !important;
        color: #eee !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .block-container {
        color: #000000 !important;
    }
    .stDataFrame, .css-1v0mbdj {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Page settings
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Online Payment Fraud Detection")
st.markdown("Upload a CSV of transactions. We'll predict which are fraudulent and show insights.")

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("fraud_model.pkl", "rb"))

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
        # 💰 Amount Range Filter
    if 'amount' in data.columns:
        st.sidebar.markdown("### 💰 Filter by Amount Range")
        min_amt = float(data['amount'].min())
        max_amt = float(data['amount'].max())
        amt_range = st.sidebar.slider("Select amount range", min_amt, max_amt, (min_amt, max_amt))
        data = data[(data['amount'] >= amt_range[0]) & (data['amount'] <= amt_range[1])]

    st.subheader("📋 Preview of Uploaded Data")
    st.dataframe(data.head(), use_container_width=True)

    # Drop target if already present
    input_data = data.drop(columns=['isFraud'], errors='ignore')

    # Predict
    predictions = model.predict(input_data)
    data["Predicted_Fraud"] = predictions

    # Downloadable result
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv, "fraud_predictions.csv", "text/csv")

    # Show prediction stats
    fraud_count = np.sum(predictions)
    total = len(predictions)
    st.success(f"✅ Total Transactions: {total} | 🚨 Predicted Fraudulent: {fraud_count} ({fraud_count/total:.2%})")

    # 📊 Layout for insights
    col1, col2 = st.columns(2)

    with col1:
        # Pie Chart: Fraud vs Non-Fraud
        labels = ['Non-Fraud', 'Fraud']
        sizes = [total - fraud_count, fraud_count]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        # Bar plot: Fraud Count by Transaction Type (if available)
        if 'type' in data.columns:
            fig2, ax2 = plt.subplots(figsize=(8,4))
            sns.countplot(data=data, x='type', hue='Predicted_Fraud', palette='Set2', ax=ax2)
            ax2.set_title("Transaction Types by Fraud Prediction")
            st.pyplot(fig2)

    # Histogram of fraud amounts
    if 'amount' in data.columns:
        st.subheader("💰 Distribution of Fraudulent Transaction Amounts")
        fraud_amounts = data[data['Predicted_Fraud'] == 1]['amount']
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.histplot(fraud_amounts, bins=50, kde=True, color='crimson', ax=ax3)
        ax3.set_xlabel("Amount")
        ax3.set_ylabel("Count")
        ax3.set_title("Distribution of Fraudulent Transaction Amounts")
        st.pyplot(fig3)
    
    st.markdown("---")
    
    

    fraud_count = np.sum(predictions)
    total = len(predictions)

    # 💥 Mind-Blowing Fraud Analysis
    st.markdown("##  Fraud Analysis Dashboard")

# 🌟 Styled animated metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Total Transactions", total)
    with col2:
        st.metric("⚠️ Fraudulent", fraud_count, delta=f"{fraud_count/total:.2%}")
    with col3:
        st.metric("✅ Safe Transactions", total - fraud_count)

# 🧠 Transaction Type vs Fraud with Animation Effect
    if 'type' in data.columns:
        st.markdown("### 🎬 Animated Fraud by Transaction Type")
        type_data = data.groupby(['type', 'Predicted_Fraud']).size().unstack(fill_value=0)
        type_data = type_data.rename(columns={0: 'Non-Fraud', 1: 'Fraud'})

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        type_data.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], ax=ax5)
        ax5.set_ylabel("Count")
        ax5.set_title("Fraud vs Non-Fraud by Type (Stacked)")
        st.pyplot(fig5)

# ⏱️ Timeline View of Fraudulent Transactions
    if 'step' in data.columns:
        st.markdown("### 🕒 Timeline of Fraudulent Transactions (Step-wise)")

        step_fraud = data[data['Predicted_Fraud'] == 1].groupby('step').size()
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=step_fraud.index, y=step_fraud.values, marker='o', color='crimson', ax=ax6)
        ax6.set_title("Fraudulent Transactions Over Time")
        ax6.set_xlabel("Step (Time Window)")
        ax6.set_ylabel("Number of Frauds")
        st.pyplot(fig6)

# 🎉 Fraud Catch Message
    st.markdown("""
<div style='
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    padding: 1.2rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    animation: pulse 2s infinite;
'>
    🔍 We just flagged 🔥 <span style="font-size:1.5rem;">{}</span> frauds out of <span style="font-size:1.5rem;">{}</span> transactions! 🚨
</div>
<style>
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 rgba(255,75,43, 0.7); }}
  70% {{ box-shadow: 0 0 0 10px rgba(255,75,43, 0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(255,75,43, 0); }}
}}
</style>
""".format(fraud_count, total), unsafe_allow_html=True)


# 🌟 Neon Style
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        border: 1px solid #ff4b2b;
        border-radius: 12px;
        padding: 10px;
        background: #1a1a1a;
        box-shadow: 0 0 15px #ff416c;
        color: #fff;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
    from { box-shadow: 0 0 5px #ff416c; }
    to { box-shadow: 0 0 20px #ff416c; }
    }
    </style>
    """, unsafe_allow_html=True)

    
st.caption("Made by Tarun Desetti 🐳")
