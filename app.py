import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cafe Sales Prediction",
    page_icon="☕",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

with open("cafe_sales_model.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("☕ Cafe Sales Prediction System")
st.write(
    "Predict the total amount spent by a customer "
    "using machine learning."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Customer Details")

quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    max_value=100,
    value=1
)

price_per_unit = st.sidebar.number_input(
    "Price Per Unit",
    min_value=0.0,
    max_value=10000.0,
    value=100.0
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Cash",
        "Credit Card",
        "Digital Wallet"
    ]
)

location = st.sidebar.selectbox(
    "Location",
    [
        "In-store",
        "Takeaway"
    ]
)

product_category = st.sidebar.selectbox(
    "Product Category",
    [
        "Coffee",
        "Tea",
        "Bakery",
        "Dessert",
        "Sandwich",
        "Other"
    ]
)

# --------------------------------------------------
# Input Data
# --------------------------------------------------

input_data = pd.DataFrame({
    "Quantity": [quantity],
    "Price Per Unit": [price_per_unit],
    "Payment Method": [payment_method],
    "Location": [location],
    "Product Category": [product_category]
})

# --------------------------------------------------
# Display Input
# --------------------------------------------------

st.subheader("Customer Input")

st.dataframe(
    input_data,
    use_container_width=True
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Total Sales", type="primary"):

    prediction = model.predict(input_data)

    predicted_sales = prediction[0]

    st.success(
        f"💰 Predicted Total Spent: ₹{predicted_sales:,.2f}"
    )

    st.metric(
        label="Predicted Sales",
        value=f"₹{predicted_sales:,.2f}"
    )

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.divider()

st.subheader("📊 Dataset Preview")

try:
    df = pd.read_csv("Cleaned_cafe_sales.csv")

    st.write(
        f"Dataset contains **{df.shape[0]} rows** "
        f"and **{df.shape[1]} columns**."
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

except Exception as e:
    st.warning("Dataset could not be loaded.")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Cafe Sales Prediction | Machine Learning + Streamlit"
)
