import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

FREIGHT_MODEL_PATH = BASE_DIR / "models" / "predict_freight_model.pkl"
INVOICE_MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_freight_model():
    return joblib.load(FREIGHT_MODEL_PATH)


@st.cache_resource
def load_invoice_model():
    return joblib.load(INVOICE_MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)


# ==========================================
# 4. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.main-title {
    font-size: 36px;
    font-weight: bold;
    color: white;
}

.subtitle {
    color: #aaaaaa;
    font-size: 17px;
}

div[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 5. HEADER
# ==========================================

st.markdown(
    '<p class="main-title">📦 Vendor Invoice Intelligence Portal</p>',
    unsafe_allow_html=True
)

st.markdown("""
### AI-Driven Freight Cost Prediction & Invoice Risk Detection

This machine learning application helps businesses:

- Forecast freight costs accurately.
- Detect risky or abnormal vendor invoices.
- Analyze vendor invoice information.
- Support financial decision-making.
""")

st.divider()


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "🚚 Freight Cost Prediction",
        "🚨 Invoice Risk Detection"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("""
### 📊 Available Models

- Linear Regression (Freight)
- Random Forest Classifier (Invoice Risk)

### ⚙️ Technology

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- SQLite
""")


# ==========================================
# 7. FREIGHT COST PREDICTION
# ==========================================

if selected_model == "🚚 Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")

    st.write(
        "Enter invoice details to predict the expected freight cost."
    )

    try:

        freight_model = load_freight_model()

        with st.form("freight_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                quantity = st.number_input(
                    "📦 Quantity",
                    min_value=1,
                    value=1200,
                    step=100
                )

            with col2:

                invoice_dollars = st.number_input(
                    "💰 Invoice Dollars",
                    min_value=0.01,
                    value=18500.0,
                    step=500.0
                )

            predict_freight = st.form_submit_button(
                "🔮 Predict Freight Cost",
                use_container_width=True
            )

        if predict_freight:

            input_data = pd.DataFrame({
                "Dollars": [invoice_dollars]
            })

            # Predict freight cost
            prediction = freight_model.predict(input_data)[0]

            st.divider()

            st.success("Freight Cost Prediction Successful!")

            # Results
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Invoice Amount",
                f"${invoice_dollars:,.2f}"
            )

            col2.metric(
                "Quantity",
                f"{quantity:,}"
            )

            col3.metric(
                "Predicted Freight",
                f"${prediction:,.2f}"
            )

            # Freight percentage
            freight_percentage = (
                prediction / invoice_dollars
            ) * 100

            st.subheader("📊 Freight Cost Analysis")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Freight Percentage",
                    f"{freight_percentage:.2f}%"
                )

            with col2:

                st.metric(
                    "Predicted Freight Cost",
                    f"${prediction:,.2f}"
                )

            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=freight_percentage,
                    title={
                        "text": "Freight Cost as % of Invoice"
                    },
                    number={
                        "suffix": "%"
                    },
                    gauge={
                        "axis": {
                            "range": [0, max(10, freight_percentage * 1.5)]
                        },
                        "bar": {
                            "color": "#00cc96"
                        },
                        "steps": [
                            {
                                "range": [
                                    0,
                                    max(5, freight_percentage * 0.5)
                                ],
                                "color": "#173c2c"
                            },
                            {
                                "range": [
                                    max(5, freight_percentage * 0.5),
                                    max(10, freight_percentage * 1.5)
                                ],
                                "color": "#262936"
                            }
                        ]
                    }
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Comparison chart
            chart_data = pd.DataFrame({
                "Category": [
                    "Invoice Amount",
                    "Predicted Freight"
                ],
                "Amount": [
                    invoice_dollars,
                    prediction
                ]
            })

            fig = px.bar(
                chart_data,
                x="Category",
                y="Amount",
                color="Category",
                
                title="Invoice Amount vs Predicted Freight"
            )

            fig.update_layout(
                template="plotly_dark",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                f"Expected freight cost: ${prediction:,.2f}"
            )

    except FileNotFoundError:

        st.error(
            "Freight model not found. "
            "Please train the regression model first."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ==========================================
# 8. INVOICE RISK DETECTION
# ==========================================

elif selected_model == "🚨 Invoice Risk Detection":

    st.header("🚨 Invoice Risk Detection")

    st.write(
        "Enter invoice and purchase details to identify potentially risky invoices."
    )

    try:

        invoice_model = load_invoice_model()

        scaler = load_scaler()

        with st.form("invoice_risk_form"):

            col1, col2 = st.columns(2)

            with col1:

                invoice_quantity = st.number_input(
                    "📦 Invoice Quantity",
                    min_value=1,
                    value=1200
                )

                invoice_dollars = st.number_input(
                    "💰 Invoice Dollars",
                    min_value=0.01,
                    value=18500.0
                )

                freight = st.number_input(
                    "🚚 Freight Cost",
                    min_value=0.0,
                    value=120.0
                )

            with col2:

                total_item_quantity = st.number_input(
                    "📦 Total Purchase Quantity",
                    min_value=1,
                    value=1200
                )

                total_item_dollars = st.number_input(
                    "💵 Total Purchase Dollars",
                    min_value=0.01,
                    value=18500.0
                )

            predict_invoice = st.form_submit_button(
                "🔍 Detect Invoice Risk",
                use_container_width=True
            )

        if predict_invoice:

            # Prepare input data
            input_data = pd.DataFrame({

                "invoice_quantity": [invoice_quantity],

                "invoice_dollars": [invoice_dollars],

                "Freight": [freight],

                "total_item_quantity": [total_item_quantity],

                "total_item_dollars": [total_item_dollars]

            })

            # Scale input data
            scaled_data = scaler.transform(input_data)

            # Predict risk
            prediction = invoice_model.predict(scaled_data)[0]

            # Get probability
            probabilities = invoice_model.predict_proba(
                scaled_data
            )[0]

            classes = list(invoice_model.classes_)

            risk_probability = (
                probabilities[classes.index(1)]
                if 1 in classes else 0
            )

            low_risk_probability = (
                probabilities[classes.index(0)]
                if 0 in classes else 0
            )

            st.divider()

            # Prediction result
            if prediction == 1:

                st.error(
                    "🚨 HIGH RISK INVOICE DETECTED!"
                )

                st.warning(
                    "This invoice has been classified as potentially risky. "
                    "Manual verification is recommended."
                )

            else:

                st.success(
                    "✅ INVOICE APPROVED - LOW RISK"
                )

                st.info(
                    "The model classified this invoice as low risk."
                )

            # Invoice metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Invoice Amount",
                f"${invoice_dollars:,.2f}"
            )

            col2.metric(
                "Freight Cost",
                f"${freight:,.2f}"
            )

            col3.metric(
                "Invoice Quantity",
                f"{invoice_quantity:,}"
            )

            st.divider()

            # ==================================
            # RISK PROBABILITY
            # ==================================

            st.subheader("📊 Invoice Risk Probability")

            col1, col2 = st.columns(2)

            col1.metric(
                "High Risk Probability",
                f"{risk_probability * 100:.2f}%"
            )

            col2.metric(
                "Low Risk Probability",
                f"{low_risk_probability * 100:.2f}%"
            )

            st.progress(float(risk_probability))

            # ==================================
            # DONUT CHART
            # ==================================

            risk_data = pd.DataFrame({
                "Risk Category": [
                    "High Risk",
                    "Low Risk"
                ],
                "Probability": [
                    risk_probability * 100,
                    low_risk_probability * 100
                ]
            })

            fig = px.pie(
                risk_data,
                names="Risk Category",
                values="Probability",
                hole=0.55,
                title="Invoice Risk Distribution",
                color="Risk Category",
                color_discrete_map={
                    "High Risk": "#ff4b4b",
                    "Low Risk": "#00cc96"
                }
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==================================
            # PURCHASE COMPARISON
            # ==================================

            st.subheader("📦 Invoice vs Purchase Analysis")

            comparison_data = pd.DataFrame({

                "Category": [
                    "Invoice Quantity",
                    "Purchase Quantity",
                    "Invoice Dollars",
                    "Purchase Dollars"
                ],

                "Value": [
                    invoice_quantity,
                    total_item_quantity,
                    invoice_dollars,
                    total_item_dollars
                ]

            })

            fig = px.bar(
                comparison_data,
                x="Category",
                y="Value",
                color="Category",
               
                title="Invoice and Purchase Comparison"
            )

            fig.update_layout(
                template="plotly_dark",
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==================================
            # FINAL RECOMMENDATION
            # ==================================

            st.subheader("💡 System Recommendation")

            if prediction == 1:

                st.warning("""
                **Recommended Action: Manual Verification**

                - Verify invoice amount.
                - Check purchase records.
                - Review freight charges.
                - Confirm vendor details.
                """)

            else:

                st.success("""
                **Recommended Action: Low Risk**

                - Invoice classified as low risk.
                - No immediate risk detected by the model.
                - Continue with the normal approval workflow.
                """)

    except FileNotFoundError:

        st.error(
            "Invoice classification model or scaler not found. "
            "Please train the classification model first."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <center>
    <h4>📦 Vendor Invoice Intelligence Portal</h4>
    Machine Learning Project | Freight Prediction & Invoice Risk Detection
    </center>
    """,
    unsafe_allow_html=True
)