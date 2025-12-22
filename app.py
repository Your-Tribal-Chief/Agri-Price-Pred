import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agri-Smart BD | AI Price Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a stunning, professional dashboard design
st.markdown("""
    <style>
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content area styling */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* All text elements */
    p, span, div, label, .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #1a1a1a !important;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border-left: 5px solid #28a745 !important;
    }
    
    .stSuccess > div, .stInfo > div, .stWarning > div {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stInfo {
        border-left-color: #17a2b8 !important;
    }
    
    .stWarning {
        border-left-color: #ffc107 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox label, .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Selectbox dropdown styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
    }
    
    /* Selectbox selected value */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
    }
    
    /* Dropdown menu options list */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    /* Individual dropdown options */
    [role="option"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Dropdown option on hover */
    [role="option"]:hover {
        background-color: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Selected option in dropdown */
    [aria-selected="true"] {
        background-color: #764ba2 !important;
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0,0,0,0.1) !important;
    }
    
    /* Cards effect for metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
    
    /* Footer styling */
    footer {
        color: #1a1a1a !important;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Loads the pre-processed Bangladesh agricultural datasets.
    Uses caching to speed up performance.
    """
    try:
        # Load CSV files
        price_df = pd.read_csv('bd_crop_price_data.csv')
        prod_df = pd.read_csv('bd_crop_production_data.csv')
        soil_df = pd.read_csv('bd_soil_analysis_data.csv')
        
        # Ensure Date column is in datetime format
        price_df['Price_Date'] = pd.to_datetime(price_df['Price_Date'])
        
        return price_df, prod_df, soil_df
    except FileNotFoundError as e:
        return None, None, None

# Load the data
price_df, prod_df, soil_df = load_data()

# Error handling if data is missing
if price_df is None:
    st.error("🚨 Error: Dataset files not found! Please ensure 'bd_crop_price_data.csv' exists in the folder.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🌾 Agri-Smart BD")
st.sidebar.markdown("**AI-Powered Farm Intelligence**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Module:",
    ["📊 Price Forecasting (AI)", "💰 Best Market Finder", "🌱 Soil & Crop Advisor"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Farm-to-Market Intelligence\n"
    "**Team:** Million Minds\n"
    "**Event:** AI Build-a-thon 2025"
)

# -----------------------------------------------------------------------------
# 4. MODULE 1: AI PRICE FORECASTING
# -----------------------------------------------------------------------------
if menu == "📊 Price Forecasting (AI)":
    st.title("📊 AI-Powered Price Forecasting")
    st.markdown("<h3 style='color: #1a1a1a; font-weight: 500;'>Predict future crop prices using Machine Learning (Random Forest) to help farmers make better selling decisions.</h3>", unsafe_allow_html=True)
    st.divider()

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        # Select District
        district_list = sorted(price_df['District_Name'].unique())
        selected_district = st.selectbox("📍 Select District", district_list)
    
    with col2:
        # Select Crop (Filtered by District)
        available_crops = sorted(price_df[price_df['District_Name'] == selected_district]['Crop_Name'].unique())
        selected_crop = st.selectbox("🌽 Select Crop", available_crops)

    # --- Data Processing ---
    # Filter data for specific district and crop
    filtered_df = price_df[
        (price_df['District_Name'] == selected_district) & 
        (price_df['Crop_Name'] == selected_crop)
    ].sort_values('Price_Date')

    if len(filtered_df) > 10:
        # --- MACHINE LEARNING SECTION ---
        
        # 1. Feature Engineering: Convert Date to Ordinal (Number) for Regression
        filtered_df['Date_Ordinal'] = filtered_df['Price_Date'].map(datetime.datetime.toordinal)
        
        # 2. Define Features (X) and Target (y)
        X = filtered_df[['Date_Ordinal']]
        y = filtered_df['Price_Tk_kg']
        
        # 3. Train Model (Random Forest Regressor)
        # Note: We train on the fly to adapt to the specific crop/location trend
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 4. Generate Future Dates (Next 30 Days)
        last_date = filtered_df['Price_Date'].max()
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        
        # 5. Predict Future Prices
        future_prices = model.predict(future_ordinals)
        
        # 6. Prepare Data for Visualization
        future_df = pd.DataFrame({
            'Price_Date': future_dates,
            'Price_Tk_kg': future_prices,
            'Type': 'Forecast (AI Prediction)'
        })
        
        filtered_df['Type'] = 'Historical Data'
        combined_df = pd.concat([filtered_df[['Price_Date', 'Price_Tk_kg', 'Type']], future_df])

        # --- VISUALIZATION ---
        st.subheader(f"Price Trend Analysis: {selected_crop}")
        
        # Interactive Line Chart using Plotly
        fig = px.line(
            combined_df, 
            x='Price_Date', 
            y='Price_Tk_kg', 
            color='Type',
            color_discrete_map={
                "Historical Data": "#1f77b4", # Blue
                "Forecast (AI Prediction)": "#00cc96" # Green
            },
            title=f"30-Day Price Forecast for {selected_crop} in {selected_district}",
            labels={'Price_Tk_kg': 'Price (BDT / kg)', 'Price_Date': 'Date'}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- INSIGHTS & METRICS ---
        current_price = filtered_df.iloc[-1]['Price_Tk_kg']
        avg_forecast_price = future_prices.mean()
        
        # Determine Trend Logic
        if avg_forecast_price > current_price:
            trend_label = "Increasing Trend 📈"
            trend_color = "normal"
        else:
            trend_label = "Decreasing Trend 📉"
            trend_color = "inverse"

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Market Price", f"৳ {current_price:.2f}")
        m2.metric("Predicted Avg (Next 30 Days)", f"৳ {avg_forecast_price:.2f}")
        m3.metric("Market Sentiment", trend_label, delta_color=trend_color)

        # Actionable Advice
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2); margin-top: 1rem;'>
            <h3 style='color: white !important; margin: 0;'>💡 AI Recommendation</h3>
            <p style='color: white !important; font-size: 1.1rem; margin-top: 0.5rem;'>
                Based on the prediction, if you sell <b>{selected_crop}</b> next week, 
                you might get an average price of <b style='font-size: 1.3rem;'>৳{future_prices[:7].mean():.2f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Not enough historical data available to generate an accurate forecast for this selection.")

# -----------------------------------------------------------------------------
# 5. MODULE 2: BEST MARKET FINDER
# -----------------------------------------------------------------------------
elif menu == "💰 Best Market Finder":
    st.title("💰 Smart Market Finder")
    st.markdown("<h3 style='color: #1a1a1a; font-weight: 500;'>Analyze real-time prices across different districts to find the <b>highest profit</b> location.</h3>", unsafe_allow_html=True)
    st.divider()

    # Select Crop
    all_crops = sorted(price_df['Crop_Name'].unique())
    target_crop = st.selectbox("🔍 Which crop do you want to sell?", all_crops)

    # Logic: Get the latest price for this crop from every district
    latest_date_in_db = price_df['Price_Date'].max()
    
    # Filter data (taking a 60-day window to ensure we find recent records)
    recent_data = price_df[
        (price_df['Crop_Name'] == target_crop) & 
        (price_df['Price_Date'] >= latest_date_in_db - datetime.timedelta(days=60))
    ]

    # Get the single latest entry for each district
    latest_prices_by_district = recent_data.sort_values('Price_Date').groupby('District_Name').tail(1)

    if not latest_prices_by_district.empty:
        # Find the max price district
        best_market = latest_prices_by_district.sort_values('Price_Tk_kg', ascending=False).iloc[0]
        max_price = best_market['Price_Tk_kg']
        best_district = best_market['District_Name']

        # Display Recommendation with stunning design
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 2rem; border-radius: 15px; color: white; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.3); margin: 1rem 0;'>
            <h2 style='color: white !important; margin: 0; font-size: 2rem;'>🏆 Top Recommendation</h2>
            <p style='color: white !important; font-size: 1.3rem; margin-top: 1rem;'>
                Send your <b>{target_crop}</b> to <b style='font-size: 1.5rem;'>{best_district}</b>!
            </p>
            <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                <p style='color: white !important; margin: 0; font-size: 0.9rem;'>Highest Price in {best_district}</p>
                <p style='color: white !important; margin: 0; font-size: 2.5rem; font-weight: 700;'>৳ {max_price:.2f} / kg</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visualization: Bar Chart Comparison
        st.subheader("Price Comparison Across Districts")
        
        fig_bar = px.bar(
            latest_prices_by_district.sort_values('Price_Tk_kg', ascending=True),
            x='Price_Tk_kg',
            y='District_Name',
            orientation='h',
            title=f"Current Market Prices for {target_crop}",
            labels={'Price_Tk_kg': 'Price (BDT/kg)', 'District_Name': 'District'},
            color='Price_Tk_kg',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.warning("No recent market data found for this crop.")

# -----------------------------------------------------------------------------
# 6. MODULE 3: SOIL & CROP ADVISOR
# -----------------------------------------------------------------------------
elif menu == "🌱 Soil & Crop Advisor":
    st.title("🌱 Intelligent Crop Advisor")
    st.markdown("<h3 style='color: #1a1a1a; font-weight: 500;'>Scientific recommendations based on soil health and historical yield data.</h3>", unsafe_allow_html=True)
    st.divider()

    # Input: Select District
    target_district = st.selectbox("📍 Select Your Farm Location", sorted(soil_df['District_Name'].unique()))

    # --- Soil Analysis ---
    # Get soil data for the selected district (taking the first record found)
    soil_record = soil_df[soil_df['District_Name'] == target_district].iloc[0]

    st.subheader(f"🧪 Soil Health Report: {target_district}")
    
    # Display Soil Metrics in Columns with enhanced styling
    st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 Soil Type", soil_record['Soil_Type'])
    c2.metric("⚗️ pH Level", f"{soil_record['pH_Level']:.2f}")
    c3.metric("🧬 Nitrogen (N)", f"{soil_record['Nitrogen_Content_kg_ha']:.1f} kg/ha")
    c4.metric("🌿 Organic Matter", f"{soil_record['Organic_Matter_Percent']:.1f}%")

    # --- Crop Recommendation ---
    st.subheader("🌾 Recommended Crops for High Yield")
    st.markdown(f"<p style='color: #1a1a1a; font-size: 1.1rem;'>Based on the soil analysis and historical production data of <b>{target_district}</b>, the following crops are recommended:</p>", unsafe_allow_html=True)

    # Logic: Find crops with highest yield (Quintals/Hectare) in this district
    district_prod = prod_df[prod_df['District_Name'] == target_district]
    
    # Group by crop and get average yield (in case of multiple seasons)
    top_crops = district_prod.groupby('Crop_Name')['Yield_Quintals_per_Ha'].mean().sort_values(ascending=False).head(5)

    # Display Top 5 Crops with beautiful cards
    for idx, (crop, yield_val) in enumerate(top_crops.items(), 1):
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.2rem; border-radius: 10px; color: white; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 0.8rem 0;'>
            <h3 style='color: white !important; margin: 0; display: flex; align-items: center;'>
                <span style='background: rgba(255,255,255,0.3); padding: 0.3rem 0.8rem; 
                             border-radius: 50%; margin-right: 1rem; font-size: 1.2rem;'>{idx}</span>
                ✅ {crop}
            </h3>
            <p style='color: white !important; font-size: 1rem; margin: 0.5rem 0 0 3rem;'>
                Average Yield: <b style='font-size: 1.2rem;'>{yield_val:.1f}</b> quintals/hectare
            </p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; padding: 2rem;'>"
    "<h4 style='color: #1a1a1a; margin: 0;'>Built for <b style='color: #667eea;'>Million Minds for Bangladesh AI Build-a-thon</b></h4>"
    "<p style='color: #2c3e50; margin-top: 0.5rem;'>Powered by Python & Streamlit | 🌾 Empowering Farmers with AI</p>"
    "</div>", 
    unsafe_allow_html=True
)