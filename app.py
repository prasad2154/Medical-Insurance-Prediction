"""
app.py
Production-grade Medical Insurance Cost Prediction Web Application.
Built with Streamlit, Scikit-learn, Pandas, and Plotly.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_dataset,
    load_saved_model,
    load_model_metrics,
    calculate_bmi_category,
    format_currency,
    convert_charge,
    get_dataset_path,
    get_model_path,
)
from src.preprocessing import format_input_dataframe, validate_inputs

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Custom CSS Injection
# ==============================================================================
def inject_custom_css():
    """Loads and injects assets/style.css into the Streamlit DOM."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


inject_custom_css()


# ==============================================================================
# Cached Resources
# ==============================================================================
@st.cache_resource(show_spinner="Loading machine learning model pipeline...")
def get_model():
    """Cached loader for the trained ML regression pipeline."""
    try:
        return load_saved_model()
    except Exception as e:
        logger.error("Error loading model: %s", e)
        st.error(f"⚠️ Model Loading Error: {e}")
        return None


@st.cache_data(show_spinner="Loading insurance dataset...")
def get_data():
    """Cached loader for the insurance dataset."""
    try:
        return load_dataset()
    except Exception as e:
        logger.error("Error loading data: %s", e)
        st.error(f"⚠️ Dataset Loading Error: {e}")
        return None


@st.cache_data
def get_metrics():
    """Cached loader for model benchmark metrics."""
    return load_model_metrics()


# Load global artifacts
df = get_data()
model_pipeline = get_model()
metrics_data = get_metrics()


# ==============================================================================
# Sidebar Navigation & Branding
# ==============================================================================
def render_sidebar():
    """Renders modern sidebar navigation, project metadata, and currency toggle."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>🏥 InsurAI Health</h2>
                <p>Medical Insurance Cost Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Main Navigation
        selected_page = st.radio(
            "Application Navigation",
            options=[
                "🏠 Home",
                "💰 Insurance Cost Prediction",
                "📊 Analytics Dashboard",
                "🤖 Model Performance",
                "🔍 Feature Insights",
            ],
            index=0,
            key="navigation_selection",
        )

        st.markdown("---")

        # Global Currency Setting
        st.markdown("#### ⚙️ Settings")
        currency = st.selectbox(
            "Display Currency",
            options=["INR (₹) - Indian Rupee", "USD ($) - US Dollar"],
            index=0,
            help="Select display currency. Toggle between Indian Rupees (₹) and US Dollars ($).",
        )
        currency_code = "INR" if "INR" in currency else "USD"

        pricing_mode = "indian_market"
        if currency_code == "INR":
            pricing_choice = st.radio(
                "🇮🇳 Pricing Scale",
                options=[
                    "Indian Market Parity (~₹25k - ₹45k/yr)",
                    "Direct Forex Rate (1 USD = ₹83.5)",
                ],
                index=0,
                help=(
                    "Indian Market Parity calibrates model charges to real-life Indian health insurance "
                    "family floater rates (~₹30,000–₹40,000/yr for a family of 4). "
                    "Direct Forex converts raw US hospital bills directly at ₹83.5/USD."
                ),
            )
            pricing_mode = "indian_market" if "Indian Market" in pricing_choice else "direct_forex"

        st.markdown("---")

        # Quick Project Info Card
        champ_name = metrics_data.get("champion_model", "Gradient Boosting")
        champ_metrics = metrics_data.get("models_comparison", {}).get(champ_name, {})
        champ_r2 = champ_metrics.get("R2", 0.8790)
        champ_mae = champ_metrics.get("MAE", 2442.87)

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.04); border: 1px solid #1E293B; border-radius: 10px; padding: 1rem;">
                <p style="color: #5EEAD4; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.35rem;">🏆 CHAMPION MODEL</p>
                <p style="color: #FFFFFF; font-size: 1rem; font-weight: 600; margin-bottom: 0.2rem;">{champ_name}</p>
                <p style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 0.6rem;">R² Score: <strong>{champ_r2 * 100:.1f}%</strong> | MAE: <strong>{format_currency(champ_mae, currency_code, pricing_mode)}</strong></p>
                <div style="height: 1px; background: #334155; margin: 0.5rem 0;"></div>
                <p style="color: #64748B; font-size: 0.75rem; margin: 0;">Production Build v2.4.0<br>Trained on 1,338 records</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Developed by ML & Streamlit Engineering")

        return selected_page, currency_code, pricing_mode


selected_page, currency_code, pricing_mode = render_sidebar()


# ==============================================================================
# PAGE 1: 🏠 Home
# ==============================================================================
def render_home_page():
    """Renders the Home landing page."""
    # Hero Banner
    st.markdown(
        """
        <div class="hero-header">
            <span class="hero-badge">⚡ Enterprise ML Analytics</span>
            <h1 class="hero-title">Medical Insurance Cost Prediction</h1>
            <p class="hero-subtitle">
                AI-powered estimation of annual medical insurance charges using high-precision machine learning regression. 
                Evaluate actuarial risk, understand health cost drivers, and generate instant demographic-driven cost estimates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards
    rec_count = len(df) if df is not None else 1338
    feat_count = (df.shape[1] - 1) if df is not None else 6
    best_model_name = metrics_data.get("champion_model", "Gradient Boosting")
    best_r2 = metrics_data.get("models_comparison", {}).get(best_model_name, {}).get("R2", 0.8790)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Dataset Volume</div>
                <div class="kpi-value">{rec_count:,}+ Records</div>
                <div class="kpi-subtext">US Beneficiary Cohort</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Model Input Features</div>
                <div class="kpi-value">{feat_count} Features</div>
                <div class="kpi-subtext">Demographic & Biometric</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Production Model</div>
                <div class="kpi-value">{best_model_name}</div>
                <div class="kpi-subtext">Optimized Ensemble Pipeline</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Model Accuracy (R²)</div>
                <div class="kpi-value">{best_r2 * 100:.1f}%</div>
                <div class="kpi-subtext">Explained Variance</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2-Column Overview Section
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🎯 Project Purpose & Mission</h3>
                <p style="color: #475569; line-height: 1.6; font-size: 0.95rem;">
                    Medical insurance underwriting traditionally relies on static risk tables that struggle with 
                    compound health risks. This application deploys a trained Scikit-learn Machine Learning pipeline 
                    to model complex, non-linear relationships between health attributes and annual charges.
                </p>
                <div style="margin-top: 1rem;">
                    <ul style="color: #334155; font-size: 0.9rem; line-height: 1.8;">
                        <li><strong>Standardized Pipeline:</strong> Full Scikit-learn ColumnTransformer with One-Hot categorical encoding and numerical scaling.</li>
                        <li><strong>Zero Data Leakage:</strong> Enforces clean train/test separation across all cross-validation folds.</li>
                        <li><strong>Multi-Model Comparison:</strong> Evaluates Linear Regression, Random Forest, and Gradient Boosting Regressors.</li>
                        <li><strong>Interactive Simulator:</strong> Instant cost projections with real-time WHO BMI risk tiering.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🚀 Ready to calculate an insurance quote?")
        st.write("Simulate insurance costs with customized age, BMI, smoking status, and family profile.")
        if st.button("🔮 Launch Prediction Engine", key="home_cta_btn"):
            st.session_state["navigation_selection"] = "💰 Insurance Cost Prediction"
            st.rerun()

    with col_right:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🔍 Key Prediction Drivers</h3>
                <div style="margin-bottom: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #0F172A;">Smoking Status</span>
                        <span style="font-size: 0.8rem; color: #0D9488; font-weight: 700;">~67% Influence</span>
                    </div>
                    <div style="background: #E2E8F0; height: 6px; border-radius: 3px;">
                        <div style="background: #0D9488; width: 67%; height: 6px; border-radius: 3px;"></div>
                    </div>
                    <p style="font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">Increases expected charges by $20,000+ on average.</p>
                </div>
                <div style="margin-bottom: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #0F172A;">Body Mass Index (BMI)</span>
                        <span style="font-size: 0.8rem; color: #06B6D4; font-weight: 700;">~20% Influence</span>
                    </div>
                    <div style="background: #E2E8F0; height: 6px; border-radius: 3px;">
                        <div style="background: #06B6D4; width: 20%; height: 6px; border-radius: 3px;"></div>
                    </div>
                    <p style="font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">Severe surcharge spikes when BMI exceeds 30 in smokers.</p>
                </div>
                <div style="margin-bottom: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #0F172A;">Age Escalation</span>
                        <span style="font-size: 0.8rem; color: #3B82F6; font-weight: 700;">~10% Influence</span>
                    </div>
                    <div style="background: #E2E8F0; height: 6px; border-radius: 3px;">
                        <div style="background: #3B82F6; width: 10%; height: 6px; border-radius: 3px;"></div>
                    </div>
                    <p style="font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">Predictable actuarial baseline escalation (~$250/year).</p>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #0F172A;">Dependents, Sex & Region</span>
                        <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 700;">~3% Influence</span>
                    </div>
                    <div style="background: #E2E8F0; height: 6px; border-radius: 3px;">
                        <div style="background: #94A3B8; width: 3%; height: 6px; border-radius: 3px;"></div>
                    </div>
                    <p style="font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">Minor geographical and dependent adjustments.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Dataset Explorer
    if df is not None:
        st.markdown("---")
        st.markdown("### 📋 Dataset Explorer & Audit")
        tab1, tab2, tab3 = st.tabs(["📄 First 10 Records", "📊 Statistical Summary", "🔍 Quality & Null Check"])

        with tab1:
            st.dataframe(df.head(10), use_container_width=True)

        with tab2:
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

        with tab3:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Missing Values", f"{df.isna().sum().sum()}")
            with col_b:
                st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
            with col_c:
                st.metric("Data Types", f"{len(df.dtypes.unique())} Unique (int, float, object)")


# ==============================================================================
# PAGE 2: 💰 Insurance Cost Prediction
# ==============================================================================
def render_prediction_page(currency_code="INR", pricing_mode="indian_market"):
    """Renders the interactive medical insurance cost prediction simulator."""
    pricing_badge_text = "🇮🇳 Indian Market Parity (~₹25k - ₹45k/yr)" if (currency_code == "INR" and pricing_mode == "indian_market") else ("Direct Forex (83.5x)" if currency_code == "INR" else "US Healthcare Scale")
    
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
                    💰 Insurance Cost Estimator
                </h1>
                <span class="badge badge-teal" style="font-size: 0.82rem; padding: 0.4rem 0.85rem;">
                    Scale: {pricing_badge_text}
                </span>
            </div>
            <p style="color: #64748B; font-size: 1.05rem;">
                Adjust customer demographic and health metrics to generate an instant ML-estimated annual insurance charge.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if currency_code == "INR" and pricing_mode == "indian_market":
        st.info("💡 **Realistic Indian Market Calibration Active:** Premium rates are calibrated to standard Indian middle-class family floater policies (Star Health, HDFC ERGO, Care Insurance ~₹30,000–₹40,000/year for a family of 4). Toggle 'Pricing Scale' in the sidebar to view raw US Dollar hospital conversions.")

    # Form Container
    with st.container():
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Customer Profile Details")

        col_input1, col_input2 = st.columns(2, gap="large")

        with col_input1:
            st.markdown("#### 👤 Demographics")
            age = st.slider(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
                help="Age of primary beneficiary (18 to 100)",
            )

            sex = st.selectbox(
                "Biological Sex",
                options=["Male", "Female"],
                index=0,
                help="Biological sex of the policyholder",
            )

            region = st.selectbox(
                "Residential Region (US)",
                options=["Southwest", "Southeast", "Northwest", "Northeast"],
                index=0,
                help="Beneficiary's residential area in the United States",
            )

        with col_input2:
            st.markdown("#### 🩺 Biometric & Health Profile")
            bmi = st.slider(
                "Body Mass Index (BMI)",
                min_value=10.0,
                max_value=60.0,
                value=27.4,
                step=0.1,
                help="BMI = weight (kg) / [height (m)]². Normal range: 18.5 - 24.9",
            )

            # Live BMI classification feedback
            bmi_info = calculate_bmi_category(bmi)
            st.markdown(
                f"""
                <div style="margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 0.85rem; color: #475569; font-weight: 500;">WHO Category:</span>
                    <span class="badge {bmi_info['badge_class']}">{bmi_info['category']}</span>
                    <span style="font-size: 0.8rem; color: #64748B;">({bmi_info['risk_multiplier']})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            children = st.slider(
                "Number of Dependent Children",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                help="Number of dependent children covered by the insurance policy",
            )

            smoker = st.radio(
                "Tobacco / Smoking Status",
                options=["No", "Yes"],
                index=0,
                horizontal=True,
                help="Does the beneficiary smoke cigarettes or use tobacco products regularly?",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Predict Insurance Cost", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Logic Execution
    if predict_button:
        if model_pipeline is None:
            st.error("⚠️ Machine Learning model is not available. Please ensure models/insurance_model.pkl exists.")
            return

        with st.spinner("Processing features through ML preprocessing & regression pipeline..."):
            try:
                # Format single-row dataframe
                input_df = format_input_dataframe(
                    age=age,
                    sex=sex,
                    bmi=bmi,
                    children=children,
                    smoker=smoker,
                    region=region,
                )

                # Generate model prediction in baseline currency (USD)
                raw_prediction = float(model_pipeline.predict(input_df)[0])
                prediction_val = max(1000.0, raw_prediction)

                monthly_val = prediction_val / 12.0
                formatted_annual = format_currency(prediction_val, currency=currency_code, pricing_mode=pricing_mode)
                formatted_monthly = format_currency(monthly_val, currency=currency_code, pricing_mode=pricing_mode)

                # Benchmark comparison against national dataset average ($13,270)
                baseline_mean = metrics_data.get("dataset_metadata", {}).get("charges_mean", 13270.42)
                diff_pct = ((prediction_val - baseline_mean) / baseline_mean) * 100
                diff_direction = "higher" if diff_pct > 0 else "lower"
                diff_badge_class = "badge-rose" if diff_pct > 25 else ("badge-teal" if diff_pct < 0 else "badge-amber")

                # Clean single-string HTML card with 0 indentation to prevent markdown <pre><code> triggers
                card_html = (
                    f'<div class="prediction-card">'
                    f'<span class="prediction-tag">✨ ML Pipeline Prediction Complete</span>'
                    f'<div style="font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">'
                    f'Estimated Annual Medical Insurance Premium'
                    f'</div>'
                    f'<div class="prediction-amount">{formatted_annual}</div>'
                    f'<div class="prediction-monthly">Approx. <strong>{formatted_monthly}</strong> / month</div>'
                    f'<p style="color: #E2E8F0; font-size: 0.98rem; line-height: 1.5; margin-bottom: 1.25rem;">'
                    f'Based on the demographic and biometric information provided, the estimated annual health insurance charge is approximately '
                    f'<strong style="color: #2DD4BF;">{formatted_annual}</strong>.'
                    f'</p>'
                    f'<div class="prediction-profile-pills">'
                    f'<span class="profile-pill">👤 Age: <strong>{age}</strong></span>'
                    f'<span class="profile-pill">⚧ Sex: <strong>{sex}</strong></span>'
                    f'<span class="profile-pill">⚖️ BMI: <strong>{bmi:.1f}</strong> ({bmi_info["category"]})</span>'
                    f'<span class="profile-pill">🚬 Smoker: <strong>{smoker}</strong></span>'
                    f'<span class="profile-pill">👶 Children: <strong>{children}</strong></span>'
                    f'<span class="profile-pill">📍 Region: <strong>{region}</strong></span>'
                    f'</div>'
                    f'<div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.12); display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;">'
                    f'<span style="font-size: 0.85rem; color: #94A3B8;">Cohort Benchmark:</span>'
                    f'<span class="badge {diff_badge_class}" style="font-size: 0.8rem;">'
                    f'{abs(diff_pct):.1f}% {diff_direction} than cohort average ({format_currency(baseline_mean, currency_code, pricing_mode)})'
                    f'</span>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                # Risk Factor Breakdown
                st.markdown("### 🔬 Actuarial Risk Analysis")
                c_risk1, c_risk2, c_risk3 = st.columns(3)
                with c_risk1:
                    if smoker.lower() == "yes":
                        surcharge_str = format_currency(18000.0, currency_code, pricing_mode)
                        st.error(f"🚨 **High Risk Surcharge: Active Smoker**\n\nTobacco use adds approximately **+{surcharge_str}** loading to annual coverage.")
                    else:
                        st.success("✅ **Standard Non-Smoker Credit**\n\nNon-smoker status qualifies you for baseline actuarial pricing.")

                with c_risk2:
                    if bmi >= 30.0:
                        st.warning(f"⚠️ **Elevated BMI ({bmi:.1f})**\n\nBMI is in the Obese category. Combined with tobacco use, this significantly amplifies premium rates.")
                    else:
                        st.info(f"📊 **BMI Status ({bmi:.1f})**\n\n{bmi_info['category']}: Within manageable risk boundaries.")

                with c_risk3:
                    age_risk_val = format_currency(age * 260.0, currency_code, pricing_mode)
                    st.info(f"🎂 **Age Impact ({age} Yrs)**\n\nAge accounts for approximately **{age_risk_val}** baseline actuarial risk across life expectancy.")

                # Legal / Medical Disclaimer
                st.markdown(
                    """
                    <div class="disclaimer-box">
                        <strong>⚠️ Notice & Educational Disclaimer:</strong> 
                        This prediction is an ML-based estimate for educational and demonstration purposes. It should not be considered an actual insurance quote, financial assessment, or medical advice. Actual health premiums depend on provider underwriting guidelines, specific medical histories, localized network tiers, and insurance plan deductibles.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                logger.error("Prediction failed: %s", e)
                st.error(f"❌ Failed to generate prediction: {e}")
                st.error(f"❌ Failed to generate prediction: {e}")


# ==============================================================================
# PAGE 3: 📊 Analytics Dashboard
# ==============================================================================
def render_analytics_page():
    """Renders comprehensive Exploratory Data Analysis & visual charts using Plotly."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
                📊 Exploratory Healthcare Analytics
            </h1>
            <p style="color: #64748B; font-size: 1.05rem;">
                Comprehensive visualization of demographic, behavioral, and biometric drivers across 1,338 insurance policies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Dataset not available. Please ensure insurance.csv is present in the workspace.")
        return

    # Top KPI Banner
    total_records = len(df)
    mean_charge = df["charges"].mean()
    median_charge = df["charges"].median()
    mean_bmi = df["bmi"].mean()
    mean_age = df["age"].mean()
    smoker_pct = (df["smoker"] == "yes").mean() * 100

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Cohort Population</div>
                <div class="kpi-value">{total_records:,}</div>
                <div class="kpi-subtext">Beneficiary Sample</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Mean Annual Charge</div>
                <div class="kpi-value">{format_currency(mean_charge, currency_code)}</div>
                <div class="kpi-subtext">Standard Average</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Median Annual Charge</div>
                <div class="kpi-value">{format_currency(median_charge, currency_code)}</div>
                <div class="kpi-subtext">50th Percentile</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average BMI</div>
                <div class="kpi-value">{mean_bmi:.1f}</div>
                <div class="kpi-subtext">WHO Overweight Tier</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Age</div>
                <div class="kpi-value">{mean_age:.1f} yrs</div>
                <div class="kpi-subtext">Age Spread 18 - 64</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Smoker Ratio</div>
                <div class="kpi-value">{smoker_pct:.1f}%</div>
                <div class="kpi-subtext">{(df['smoker'] == 'yes').sum()} Active Smokers</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Interactive Visualizations Grid
    st.markdown("### 📈 Core Actuarial Relationships")

    c1, c2 = st.columns(2)

    # 1. Age vs Charges
    with c1:
        fig_age = px.scatter(
            df,
            x="age",
            y="charges",
            color="smoker",
            color_discrete_map={"yes": "#EF4444", "no": "#0D9488"},
            labels={"age": "Age (Years)", "charges": f"Charges ({currency_code})", "smoker": "Smoker"},
            title="1. Age vs. Annual Charges (Stratified by Smoker)",
            hover_data=["bmi", "children", "region"],
            trendline="ols",
            opacity=0.75,
        )
        fig_age.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # 2. BMI vs Charges
    with c2:
        fig_bmi = px.scatter(
            df,
            x="bmi",
            y="charges",
            color="smoker",
            color_discrete_map={"yes": "#EF4444", "no": "#0D9488"},
            labels={"bmi": "Body Mass Index (BMI)", "charges": f"Charges ({currency_code})", "smoker": "Smoker"},
            title="2. BMI vs. Charges (Obesity Tipping Point at BMI=30)",
            hover_data=["age", "children", "region"],
            opacity=0.75,
        )
        # Vertical reference line at BMI = 30
        fig_bmi.add_vline(x=30, line_width=2, line_dash="dash", line_color="#F59E0B", annotation_text="Obesity (BMI 30)", annotation_position="top left")
        fig_bmi.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

    c3, c4 = st.columns(2)

    # 3. Smoker vs Charges Box Plot
    with c3:
        fig_smoker = px.box(
            df,
            x="smoker",
            y="charges",
            color="smoker",
            color_discrete_map={"yes": "#EF4444", "no": "#0D9488"},
            labels={"smoker": "Smoking Status", "charges": f"Charges ({currency_code})"},
            title="3. Distribution Divergence: Smoker vs. Non-Smoker",
            points="all",
        )
        fig_smoker.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_smoker, use_container_width=True)

    # 4. Region-wise Charges
    with c4:
        region_stats = df.groupby("region")["charges"].agg(["mean", "median"]).reset_index()
        fig_region = px.bar(
            region_stats,
            x="region",
            y=["mean", "median"],
            barmode="group",
            labels={"value": f"Charges ({currency_code})", "region": "Region", "variable": "Metric"},
            title="4. Regional Health Charges (Mean vs. Median)",
            color_discrete_sequence=["#0D9488", "#06B6D4"],
        )
        fig_region.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_region, use_container_width=True)

    c5, c6 = st.columns(2)

    # 5. Sex vs Charges
    with c5:
        fig_sex = px.violin(
            df,
            x="sex",
            y="charges",
            color="sex",
            box=True,
            points="outliers",
            color_discrete_sequence=["#3B82F6", "#EC4899"],
            labels={"sex": "Biological Sex", "charges": f"Charges ({currency_code})"},
            title="5. Gender Comparison: Male vs. Female Charges",
        )
        fig_sex.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    # 6. Children vs Charges
    with c6:
        fig_children = px.box(
            df,
            x="children",
            y="charges",
            color="children",
            labels={"children": "Number of Dependent Children", "charges": f"Charges ({currency_code})"},
            title="6. Dependent Count vs. Medical Charges",
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig_children.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_children, use_container_width=True)

    st.markdown("---")

    # Additional Distribution Deep Dive
    st.markdown("### 🔬 Population Density & Distributions")
    tab_dist1, tab_dist2, tab_dist3 = st.tabs(["📊 Target Variable (Charges)", "⚖️ BMI Distribution", "🎂 Age Distribution"])

    with tab_dist1:
        fig_dist_charges = px.histogram(
            df,
            x="charges",
            nbins=40,
            marginal="box",
            color_discrete_sequence=["#0D9488"],
            title="Right-Skewed Distribution of Annual Insurance Charges",
            labels={"charges": f"Charges ({currency_code})"},
        )
        fig_dist_charges.update_layout(template="plotly_white")
        st.plotly_chart(fig_dist_charges, use_container_width=True)
        st.caption("Notice the bimodal/trimodal peaks corresponding to (1) non-smokers, (2) overweight non-smokers/older cohorts, and (3) smokers with high BMI.")

    with tab_dist2:
        fig_dist_bmi = px.histogram(
            df,
            x="bmi",
            nbins=35,
            marginal="box",
            color_discrete_sequence=["#06B6D4"],
            title="Normal/Bell Curve Distribution of Body Mass Index (BMI)",
            labels={"bmi": "BMI (kg/m²)"},
        )
        fig_dist_bmi.add_vline(x=25, line_dash="dot", line_color="#10B981", annotation_text="Overweight (25)")
        fig_dist_bmi.add_vline(x=30, line_dash="dash", line_color="#EF4444", annotation_text="Obese (30)")
        fig_dist_bmi.update_layout(template="plotly_white")
        st.plotly_chart(fig_dist_bmi, use_container_width=True)

    with tab_dist3:
        fig_dist_age = px.histogram(
            df,
            x="age",
            nbins=30,
            marginal="rug",
            color_discrete_sequence=["#6366F1"],
            title="Uniform Age Distribution Across Working Cohort (18 - 64)",
            labels={"age": "Age (Years)"},
        )
        fig_dist_age.update_layout(template="plotly_white")
        st.plotly_chart(fig_dist_age, use_container_width=True)


# ==============================================================================
# PAGE 4: 🤖 Model Performance
# ==============================================================================
def render_model_performance_page():
    """Renders model benchmarking, evaluation metrics, and regression diagnostics."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
                🤖 Machine Learning Model Evaluation
            </h1>
            <p style="color: #64748B; font-size: 1.05rem;">
                Rigorous evaluation and benchmarking of regression models on unseen test holdout data (80/20 split).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models_comparison = metrics_data.get("models_comparison", {})
    champion = metrics_data.get("champion_model", "Gradient Boosting")

    # Comparison Table
    if models_comparison:
        table_rows = []
        for model_name, metrics in models_comparison.items():
            is_champ = model_name == champion
            table_rows.append({
                "Model Architecture": f"{model_name} {'🏆 (Champion)' if is_champ else ''}",
                "MAE ($)": f"${metrics.get('MAE', 0):,.2f}",
                "RMSE ($)": f"${metrics.get('RMSE', 0):,.2f}",
                "R² Score": f"{metrics.get('R2', 0):.4f}",
                "Status": "Selected for Deployment" if is_champ else "Benchmarked Baseline",
            })

        metrics_df = pd.DataFrame(table_rows)
        st.markdown("### 🏆 Model Leaderboard & Comparison")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Metrics payload loading or training pipeline in progress...")

    st.markdown("---")

    # Diagnostic Plots
    st.markdown("### 📉 Champion Regression Diagnostics")
    diag_col1, diag_col2 = st.columns(2)

    sample_eval = metrics_data.get("sample_evaluation_points", [])

    if sample_eval:
        eval_df = pd.DataFrame(sample_eval)

        # 1. Actual vs Predicted Scatter
        with diag_col1:
            fig_actual_pred = go.Figure()
            fig_actual_pred.add_trace(go.Scatter(
                x=eval_df["actual"],
                y=eval_df["predicted"],
                mode="markers",
                marker=dict(size=9, color="#0D9488", opacity=0.8, line=dict(width=1, color="#042F2E")),
                name="Test Instances",
                text=[f"Age: {r['age']}, BMI: {r['bmi']}, Smoker: {r['smoker']}" for _, r in eval_df.iterrows()],
            ))
            # 45-degree reference line
            min_val = min(eval_df["actual"].min(), eval_df["predicted"].min()) * 0.9
            max_val = max(eval_df["actual"].max(), eval_df["predicted"].max()) * 1.05
            fig_actual_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#EF4444", dash="dash", width=2),
                name="Ideal Prediction (y = x)",
            ))
            fig_actual_pred.update_layout(
                title="Actual vs. Predicted Charges ($)",
                xaxis_title="Actual Ground Truth ($)",
                yaxis_title="Model Estimated ($)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_actual_pred, use_container_width=True)

        # 2. Residual Distribution
        with diag_col2:
            eval_df["residuals"] = eval_df["actual"] - eval_df["predicted"]
            fig_res = px.scatter(
                eval_df,
                x="predicted",
                y="residuals",
                labels={"predicted": "Fitted / Predicted Values ($)", "residuals": "Residual Error ($)"},
                title="Residual Error vs. Fitted Values (Homoscedasticity)",
                color_discrete_sequence=["#3B82F6"],
                opacity=0.8,
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=2)
            fig_res.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_res, use_container_width=True)

    # Simple-Language Metric Guides
    st.markdown("---")
    st.markdown("### 💡 Understanding the Evaluation Metrics")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4 style="color: #0D9488; margin-top: 0;">R² Score (Coefficient of Determination)</h4>
                <p style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                    Measures the percentage of total variance in medical charges explained by our inputs. 
                    An <strong>R² of ~0.88</strong> indicates that our model successfully accounts for 88% of all cost variations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4 style="color: #06B6D4; margin-top: 0;">MAE (Mean Absolute Error)</h4>
                <p style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                    The average dollar discrepancy between estimated charges and actual hospital bills. 
                    An <strong>MAE of ~$2,442</strong> means predictions are, on average, within $2,442 of real charges.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4 style="color: #3B82F6; margin-top: 0;">RMSE (Root Mean Squared Error)</h4>
                <p style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                    Similar to MAE, but penalizes large outlier errors more severely. 
                    Our <strong>RMSE of ~$4,334</strong> verifies robust performance even on rarer high-cost treatment outliers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 5: 🔍 Feature Insights
# ==============================================================================
def render_feature_insights_page():
    """Renders Feature Importance visual and domain explanations."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
                🔍 Actuarial Feature Importance
            </h1>
            <p style="color: #64748B; font-size: 1.05rem;">
                Interpretability and feature contribution ranking derived from the champion Gradient Boosting tree model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feat_importances = metrics_data.get("feature_importance", {})

    if feat_importances:
        feat_df = pd.DataFrame([
            {"Feature": k, "Importance": v, "Percentage": f"{v * 100:.1f}%"}
            for k, v in feat_importances.items()
        ]).sort_values("Importance", ascending=True)

        # Plotly Horizontal Bar
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Percentage",
            title="Tree-Based Relative Feature Importance (Gradient Boosting)",
            color="Importance",
            color_continuous_scale="Teal",
            labels={"Importance": "Gini Impurity Reduction Ratio", "Feature": "Input Feature"},
        )
        fig_feat.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Feature importance dictionary not found. Run training script to regenerate.")

    # Domain Breakdown Cards
    st.markdown("### 📚 Domain Actuarial Explanations")

    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🚬 Smoker Status (Primary Driver: ~67%)</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    Tobacco use is the single most dominant risk factor in health actuarial underwriting. 
                    Smokers exhibit statistically elevated rates of cardiovascular disease, lung disorders, 
                    and surgical complications, leading to a massive median charge premium spike of over <strong>+$20,000</strong>.
                </p>
            </div>
            <div class="feature-card">
                <h4>⚖️ Body Mass Index (Second Driver: ~20%)</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    BMI displays non-linear interaction effects. For non-smokers, high BMI causes gradual cost increases; 
                    however, when high BMI (≥ 30, Obese tier) combines with active smoking, health insurance charges multiply 
                    exponentially due to elevated comorbid risks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feat_col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🎂 Beneficiary Age (Third Driver: ~10%)</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    As policyholders age from 18 to 64, baseline healthcare utilization steadily rises due to chronic disease onset, 
                    screening frequencies, and pharmaceuticals. In our model, each additional year of age introduces a predictable 
                    linear escalation of approximately <strong>$250 to $275/year</strong>.
                </p>
            </div>
            <div class="feature-card">
                <h4>👨‍👩‍👧‍👦 Dependents, Region & Sex (Minor Drivers: ~3%)</h4>
                <p style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    While dependents add nominal child-rearing and pediatric coverage costs, regional differences account 
                    for local hospital facility charge structures (with Southeast slightly higher on average). Biological sex 
                    exhibits minimal divergence after adjusting for smoking and BMI.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Actuarial Causality Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box" style="margin-top: 1rem;">
            <strong>ℹ️ Scientific Disclaimer on Feature Importance:</strong>
            Feature importance scores indicate how frequently and effectively a feature was selected by decision trees to split samples and reduce prediction loss. 
            It reflects <em>predictive utility</em> within this dataset and should not be confused with strict clinical or epidemiological causality.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Router
# ==============================================================================
if selected_page == "🏠 Home":
    render_home_page()
elif selected_page == "💰 Insurance Cost Prediction":
    render_prediction_page()
elif selected_page == "📊 Analytics Dashboard":
    render_analytics_page()
elif selected_page == "🤖 Model Performance":
    render_model_performance_page()
elif selected_page == "🔍 Feature Insights":
    render_feature_insights_page()
