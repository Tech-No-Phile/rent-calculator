import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mumbai Rent Pro | AI & Heatmap",
    page_icon="🏙️",
    layout="wide"
)

# --- ADVANCED UI STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        border-radius: 12px;
        border: none;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5);
        color: white;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1E3A8A;
        margin: 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Titles */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(#1E3A8A, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PROCESSING ENGINE ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Mumbai_House_Rent.csv")
    
    # Cleaning numeric columns from strings (e.g., '350 sq.ft')
    def clean_sqft(x):
        if isinstance(x, str):
            x = x.replace(' sq.ft', '').replace('Missing', '')
            return float(x) if x != '' else np.nan
        return x

    df['area'] = df['Build_up_area(sq.ft)'].apply(clean_sqft)
    
    # Feature Engineering for Bedrooms
    def extract_beds(x):
        if 'RK' in x: return 0.5
        try: return float(x.split(' ')[0])
        except: return 1.0
    
    df['beds'] = df['Type'].apply(extract_beds)
    
    # Handling missing values and formatting
    df['bathrooms_num'] = pd.to_numeric(df['Bathrooms'].replace('Missing', '1'), errors='coerce').fillna(1)
    df['balcony_num'] = pd.to_numeric(df['Balcony'].replace('Missing', '0'), errors='coerce').fillna(0)
    df['parking_num'] = df['Parking'].fillna(0)
    
    # Hardcoded coordinates for Mumbai Localities in the dataset
    # This allows the Heatmap and Map to function without an external API
    coords = {
        'Andheri': [19.12, 72.84], 'Bandra': [19.05, 72.83], 'Bhandup': [19.15, 72.94],
        'Byculla': [18.98, 72.83], 'Chembur': [19.06, 72.90], 'Colaba': [18.91, 72.81],
        'Dadar': [19.02, 72.84], 'Dharavi': [19.04, 72.85], 'Fort': [18.93, 72.83],
        'Ghatkopar': [19.09, 72.91], 'Girgaon': [18.95, 72.81], 'Goregaon': [19.16, 72.85],
        'Govandi': [19.05, 72.92], 'Grant Road': [18.96, 72.81], 'Jogeshwari': [19.13, 72.85],
        'Juhu': [19.11, 72.82], 'Khar': [19.07, 72.83], 'Kurla': [19.07, 72.88],
        'Lalbaug': [18.99, 72.84], 'Lokhandwala': [19.14, 72.82], 'Mahalakshmi': [18.98, 72.82],
        'Mahim': [19.04, 72.84], 'Malabar Hill': [18.95, 72.80], 'Malad': [19.18, 72.85],
        'Marine Drive': [18.94, 72.82], 'Masjid': [18.95, 72.83], 'Matunga': [19.03, 72.85],
        'Mulund': [19.17, 72.95], 'Nariman Point': [18.92, 72.82], 'Parel': [19.00, 72.84],
        'Powai': [19.12, 72.91], 'Prabhadevi': [19.02, 72.83], 'Santacruz': [19.08, 72.84],
        'Sion': [19.04, 72.86], 'Tardeo': [18.97, 72.81], 'Vidyavihar': [19.08, 72.90],
        'Vikhroli': [19.11, 72.93], 'Vile Parle': [19.10, 72.84], 'Wadala': [19.02, 72.86],
        'Worli': [19.00, 72.82]
    }
    
    df['lat'] = df['Locality'].map(lambda x: coords.get(x, [19.076, 72.877])[0])
    df['lon'] = df['Locality'].map(lambda x: coords.get(x, [19.076, 72.877])[1])
    
    return df, coords

df, locality_coords = load_and_clean_data()

# --- AI MODEL TRAINING ---
@st.cache_resource
def train_model(data):
    df_train = data.dropna(subset=['area', 'Rent/Month']).copy()
    
    # Define features
    categorical_cols = ["Locality", "Type", "Furnishing"]
    features = ["Locality", "Type", "Furnishing", "area", "beds", "bathrooms_num", "balcony_num", "parking_num"]
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        encoders[col] = le
        
    X = df_train[features]
    y = df_train["Rent/Month"]
    
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, encoders

model, encoders = train_model(df)

# --- UI HEADER ---
st.markdown("<h1 style='text-align:center;'>🏙️ Mumbai Real Estate AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Visualizing Market Intensity & AI-Driven Rent Estimation</p>", unsafe_allow_html=True)
st.write("---")

# --- MAIN DASHBOARD LAYOUT ---
col_inputs, col_map = st.columns([1, 1.3])

with col_inputs:
    st.markdown("### 📋 Property Configuration")
    
    with st.container():
        # Location & Type
        c1, c2 = st.columns(2)
        with c1:
            loc_input = st.selectbox("📍 Locality", sorted(df["Locality"].unique()))
            type_input = st.selectbox("🏡 Property Type", sorted(df["Type"].unique()))
        with c2:
            furn_input = st.selectbox("🛋️ Furnishing", sorted(df["Furnishing"].unique()))
            area_input = st.slider("📐 Area (sq ft)", 200, 5000, 1000)
        
        # Room Details
        c3, c4, c5 = st.columns(3)
        with c3: beds_in = st.number_input("🛏️ BHK", 0.5, 10.0, 2.0, step=0.5)
        with c4: baths_in = st.number_input("🚿 Baths", 1, 8, 2)
        with c5: park_in = st.number_input("🚗 Parking", 0, 4, 1)

    st.write(" ")
    predict_btn = st.button("🚀 Analyze Market Value")

with col_map:
    st.markdown("### 🔥 Rental Intensity Heatmap")
    # Base Map focused on Mumbai
    m = folium.Map(location=[19.076, 72.877], zoom_start=11, tiles="CartoDB Positron")
    
    # Prepare data for Heatmap: [[lat, lon, weight]]
    heat_data = df[['lat', 'lon', 'Rent/Month']].dropna().values.tolist()
    
    # Add Heatmap layer
    HeatMap(heat_data, radius=18, blur=12, min_opacity=0.4).add_to(m)
    
    # Add marker for selected locality
    target_coords = locality_coords.get(loc_input, [19.076, 72.877])
    folium.Marker(
        target_coords, 
        popup=f"Target: {loc_input}", 
        icon=folium.Icon(color='blue', icon='home')
    ).add_to(m)
    
    # Display the Map
    st_folium(m, width="100%", height=450)

# --- PREDICTION LOGIC & RESULTS ---
if predict_btn:
    try:
        # Prepare input for model
        input_data = pd.DataFrame([{
            "Locality": encoders["Locality"].transform([loc_input])[0],
            "Type": encoders["Type"].transform([type_input])[0],
            "Furnishing": encoders["Furnishing"].transform([furn_input])[0],
            "area": area_input,
            "beds": beds_in,
            "bathrooms_num": baths_in,
            "balcony_num": 0,
            "parking_num": park_in
        }])
        
        prediction = model.predict(input_data)[0]
        
        st.write("---")
        st.markdown("<h2 style='text-align:center;'>💎 AI Valuation Results</h2>", unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3)
        
        # Custom Metric Cards
        with r1:
            st.markdown(f"""<div class="metric-card">
                <p class="metric-label">Estimated Rent</p>
                <p class="metric-value">₹{int(prediction):,}</p>
            </div>""", unsafe_allow_html=True)
            
        with r2:
            st.markdown(f"""<div class="metric-card">
                <p class="metric-label">Fair Range (Min)</p>
                <p class="metric-value" style="color:#28a745;">₹{int(prediction*0.9):,}</p>
            </div>""", unsafe_allow_html=True)
            
        with r3:
            st.markdown(f"""<div class="metric-card">
                <p class="metric-label">Premium Range (Max)</p>
                <p class="metric-value" style="color:#FF4B2B;">₹{int(prediction*1.15):,}</p>
            </div>""", unsafe_allow_html=True)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# --- FOOTER ---
st.write("---")
st.caption("Developed for Mumbai Real Estate Analytics | Powered by Random Forest Regressor & Folium Maps")