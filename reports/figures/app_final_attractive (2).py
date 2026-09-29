"""
HomeMetric - Smart Property Valuation & Discovery Platform
===========================================================
Streamlit Web Application:
- Step 1: HomeMetric Branding & Architecture.
- Step 2: Welcome / User Profile Page.
- Step 3: Property Requirements Page.
- Step 4: Demo Property Matching Results & Scoring Engine.
- Step 5: Dedicated Property Details Page.
- Step 6: Enhanced AI Price Estimation for Selected Demo Properties.
- Step 7: Rule-Based Context-Aware AI Property Assistant Chatbot.
- Step 8B: UI Visual Refinement (Compact Layout, Balanced Spacing, Clean Hierarchy).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib



# -------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="HomeMetric — Smart Property Valuation & Discovery Platform",
    page_icon="🏠",
    layout="wide",
)

# Professional Property-Tech Refined CSS Design System
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; color:#172033; }
.stApp {
    background:
      radial-gradient(circle at 5% 5%, rgba(59,130,246,.22), transparent 25%),
      radial-gradient(circle at 96% 8%, rgba(16,185,129,.20), transparent 24%),
      radial-gradient(circle at 92% 92%, rgba(245,158,11,.18), transparent 25%),
      linear-gradient(135deg,#eef6ff 0%,#f8fbff 42%,#f0fdf7 100%);
    min-height:100vh;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1240px !important; padding:1.4rem 2rem 3rem !important; }

/* Strong branded header */
.hm-top-brand {
    position:relative; overflow:hidden;
    background:linear-gradient(120deg,#0f2a68 0%,#1d4ed8 48%,#0f766e 100%);
    border:0; border-radius:28px; padding:28px 34px; margin:0 0 24px;
    box-shadow:0 20px 45px rgba(30,64,175,.22);
    color:white;
}
.hm-top-brand:after {
    content:""; position:absolute; width:240px; height:240px; right:-70px; top:-120px;
    border-radius:50%; background:rgba(255,255,255,.12);
    box-shadow:-120px 170px 0 25px rgba(255,255,255,.06);
}
.hm-top-title { font-size:2rem; font-weight:800; letter-spacing:-1px; color:white; position:relative; z-index:2; }
.hm-top-subtitle { font-size:.94rem; font-weight:500; color:#dbeafe; margin-top:4px; position:relative; z-index:2; }

/* Hero */
.hm-hero-card {
    position:relative; overflow:hidden;
    background:linear-gradient(135deg,rgba(255,255,255,.98),rgba(239,246,255,.96));
    border:1px solid #bfdbfe; border-radius:26px; padding:34px 36px; margin-bottom:22px;
    box-shadow:0 16px 38px rgba(15,23,42,.08);
}
.hm-hero-card:before { content:"🏡"; position:absolute; right:32px; bottom:-20px; font-size:8rem; opacity:.10; }
.hm-card {
    background:rgba(255,255,255,.94); border:1px solid #dbe5f0; border-radius:22px;
    padding:26px 28px; margin-bottom:20px; box-shadow:0 12px 30px rgba(15,23,42,.07);
}

/* Inputs */
[data-testid="stTextInput"], [data-testid="stNumberInput"], [data-testid="stSelectbox"], [data-testid="stMultiSelect"] { margin-bottom:8px; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
    border-radius:13px !important; border:1px solid #cbd5e1 !important; background:#fff !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    border-radius:13px !important; border-color:#cbd5e1 !important; background:#fff !important;
}

/* Buttons */
.stButton > button {
    border-radius:13px !important; min-height:46px; font-weight:700 !important;
    border:1px solid #bfdbfe !important; box-shadow:0 5px 14px rgba(15,23,42,.06);
    transition:.18s ease !important;
}
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 22px rgba(37,99,235,.16); }
button[kind="primary"] { background:linear-gradient(135deg,#2563eb,#1d4ed8) !important; color:white !important; border:0 !important; }

/* Property cards */
.property-card {
    background:rgba(255,255,255,.97); border:1px solid #dbe5f0; border-radius:22px;
    padding:24px; margin-bottom:20px; box-shadow:0 14px 32px rgba(15,23,42,.08);
    transition:.18s ease;
}
.property-card:hover { transform:translateY(-4px); box-shadow:0 20px 38px rgba(15,23,42,.12); border-color:#93c5fd; }
.prop-title { font-size:1.22rem; font-weight:800; color:#0f172a; }
.prop-location { color:#64748b; font-size:.9rem; margin:4px 0 12px; }
.prop-price { font-size:1.55rem; font-weight:800; color:#1d4ed8; }
.prop-meta { color:#334155; font-weight:600; }

/* Pills / info blocks */
.type-tag,.match-pill-high,.match-pill-medium,.match-pill-low {
    border-radius:999px !important; padding:6px 12px !important; font-weight:700 !important;
}
.contact-card { background:linear-gradient(135deg,#ecfdf5,#eff6ff); border:1px solid #a7f3d0; border-radius:18px; padding:18px 20px; }
.price-card { background:linear-gradient(135deg,#0f2a68,#2563eb,#0f766e); border:0; border-radius:24px; padding:30px; color:white; box-shadow:0 18px 40px rgba(37,99,235,.25); }
.price-value { font-size:3rem; font-weight:800; }

/* Alerts */
[data-testid="stAlert"] { border-radius:16px !important; }
hr { border-color:#dbe5f0 !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. Artifact & Dataset Loading (Cached for fast performance)
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads the saved preprocessing pipeline, trained models, and demo listings.
    Does NOT retrain anything.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")

    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    rf_model_path = os.path.join(models_dir, "random_forest.joblib")
    lr_model_path = os.path.join(models_dir, "linear_regression.joblib")
    dt_model_path = os.path.join(models_dir, "decision_tree.joblib")
    demo_prop_path = os.path.join(data_dir, "demo_properties.csv")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor artifact not found at: {preprocessor_path}")
    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"Model artifact not found at: {rf_model_path}")
    if not os.path.exists(demo_prop_path):
        raise FileNotFoundError(f"Demo property dataset not found at: {demo_prop_path}")

    preprocessor = joblib.load(preprocessor_path)
    models = {
        "Random Forest (Best: R² = 83.5%)": joblib.load(rf_model_path),
        "Linear Regression (R² = 82.1%)": joblib.load(lr_model_path),
        "Decision Tree (R² = 71.7%)": joblib.load(dt_model_path)
    }
    demo_properties = pd.read_csv(demo_prop_path)

    return preprocessor, models, demo_properties


try:
    preprocessor, models_dict, demo_df = load_artifacts()
except Exception as e:
    st.error(f"Error loading application artifacts: {e}")
    st.stop()


# -------------------------------------------------------------
# 3. Transparent Property Matching Algorithm (0 - 100 Score)
# -------------------------------------------------------------
def calculate_match_score(prop: pd.Series, req: dict):
    """
    Calculates a transparent rule-based matching score (0-100%) and generates
    verifiable matching reasons based on the user's specific requirements.
    NOTE: This is a property search matching metric, NOT predicted by Random Forest.
    """
    score = 0
    reasons = []

    # 1. Location Matching (City & Locality)
    if prop["city"].strip().lower() == req["city"].strip().lower():
        score += 25
        reasons.append(f"✓ Located in target city: **{prop['city']}** (+25%)")
        if req.get("locality") and req["locality"].strip().lower() in prop["locality"].strip().lower():
            score += 5
            reasons.append(f"✓ Prime match in preferred locality: **{prop['locality']}** (+5%)")
    else:
        reasons.append(f"✗ Located in {prop['city']} (Preferred: {req['city']})")

    # 2. Budget Matching
    price = float(prop["price_lakhs"])
    min_b, max_b = float(req["min_budget"]), float(req["max_budget"])
    if min_b <= price <= max_b:
        score += 20
        reasons.append(f"✓ Price (₹{price:.1f}L) is within your budget of ₹{min_b:.0f}L - ₹{max_b:.0f}L (+20%)")
    elif price < min_b:
        score += 15
        reasons.append(f"✓ Price (₹{price:.1f}L) is below your minimum budget (+15%)")
    elif price <= max_b * 1.15:
        score += 10
        reasons.append(f"~ Price (₹{price:.1f}L) is slightly (+15%) above budget ceiling (+10%)")
    else:
        reasons.append(f"✗ Price (₹{price:.1f}L) exceeds maximum budget of ₹{max_b:.0f}L")

    # 3. Specifications Matching
    if req["property_type"] == "🏠 House / Apartment" and prop["property_type"] == "House / Apartment":
        # Area matching
        area = float(prop["area_sqft"])
        if float(req["min_area"]) <= area <= float(req["max_area"]):
            score += 15
            reasons.append(f"✓ Living area ({area:,.0f} sq.ft.) matches preferred range (+15%)")
        elif area >= float(req["min_area"]) * 0.9:
            score += 8
            reasons.append(f"~ Living area ({area:,.0f} sq.ft.) close to target (+8%)")

        # Rooms / BHK matching
        if not pd.isna(prop.get("rooms")):
            p_rooms = float(prop["rooms"])
            req_rooms = float(req["rooms"])
            if p_rooms == req_rooms:
                score += 15
                reasons.append(f"✓ Exact configuration match: **{int(p_rooms)} BHK** (+15%)")
            elif abs(p_rooms - req_rooms) == 1:
                score += 8
                reasons.append(f"~ Nearby configuration: {int(p_rooms)} BHK (+8%)")

        # Property Age
        if not pd.isna(prop.get("age")):
            p_age = int(prop["age"])
            if p_age <= int(req["max_age"]):
                score += 10
                reasons.append(f"✓ Age ({p_age} yrs) is within your maximum limit of {req['max_age']} yrs (+10%)")

        # Amenities Matching
        p_amenities = [a.strip().lower() for a in str(prop.get("amenities", "")).split(";") if a.strip()]
        req_amenities = req.get("amenities", [])
        if req_amenities:
            matched_amenities = [a for a in req_amenities if a.lower() in p_amenities]
            match_pct = len(matched_amenities) / len(req_amenities)
            amen_pts = int(match_pct * 10)
            score += amen_pts
            if matched_amenities:
                amen_str = ", ".join(a.title() for a in matched_amenities)
                reasons.append(f"✓ Includes {len(matched_amenities)} of {len(req_amenities)} preferred amenities: {amen_str} (+{amen_pts}%)")

    elif req["property_type"] == "🌳 Land / Plot" and prop["property_type"] == "Land / Plot":
        # Plot Area
        area = float(prop["area_sqft"])
        if float(req["min_plot_area"]) <= area <= float(req["max_plot_area"]):
            score += 20
            reasons.append(f"✓ Plot area ({area:,.0f} sq.ft.) is within your desired dimensions (+20%)")
        
        # Land Type
        if str(prop.get("land_type")).strip().lower() == str(req.get("land_type")).strip().lower():
            score += 15
            reasons.append(f"✓ Land classification matches: **{prop['land_type']}** (+15%)")

        # Road Access & Corner
        if str(prop.get("road_access")).strip().lower() == "available" or str(req.get("road_access")).strip().lower() == "not required":
            score += 10
            reasons.append("✓ Road access available (+10%)")

    final_score = min(100, max(0, score))
    return final_score, reasons


# -------------------------------------------------------------
# 4. Rule-Based AI Property Assistant Response Engine (Step 7)
# -------------------------------------------------------------
def get_assistant_response(user_query: str, state: dict) -> str:
    """
    Generates intelligent, context-aware rule-based answers for HomeMetric.
    Runs 100% locally without external API keys.
    """
    q = user_query.strip().lower()

    # 1. Selected Property Context
    if any(k in q for k in ["selected property", "my property", "tell me about this property", "chosen property", "selected demo", "tell me about my selected property", "show me my selected property"]):
        prop = state.get("selected_property")
        if not prop:
            return "⚠️ **Please select a property from Matching Properties first.**"
        
        p_type = prop.get("property_type", "House / Apartment")
        price = float(prop.get("price_lakhs", 0))
        area = float(prop.get("area_sqft", 0))
        rate = (price * 1e5) / area if area > 0 else 0
        score = prop.get("match_score", 90)

        res = f"### 🏠 Information for Selected Demo Property:\n"
        res += f"- **Property Title**: {prop.get('title')}\n"
        res += f"- **City**: {prop.get('city')}\n"
        res += f"- **Locality**: {prop.get('locality')}\n"
        res += f"- **Property Type**: {p_type}\n"
        res += f"- **Price**: ₹{price:.1f} Lakhs (Approx. ₹{rate:,.0f} / sq.ft.)\n"
        res += f"- **Area**: {area:,.0f} sq.ft.\n"
        
        if p_type == "House / Apartment":
            rooms = int(prop.get('rooms', 3)) if not pd.isna(prop.get('rooms')) else 3
            age = int(prop.get('age', 0)) if not pd.isna(prop.get('age')) else 0
            amenities = str(prop.get('amenities', '')).replace(';', ', ').title()
            res += f"- **BHK/Rooms**: {rooms} BHK\n"
            res += f"- **Property Age**: {age} Years\n"
            res += f"- **Amenities**: {amenities if amenities else 'Standard society amenities'}\n"
        else:
            res += f"- **Land Type**: {prop.get('land_type', 'Residential')}\n"
            res += f"- **Road Access**: {prop.get('road_access', 'Available')}\n"
            res += f"- **Corner Plot**: {prop.get('corner_plot', 'No')}\n"

        res += f"- **Match Percentage**: 🎯 **{score}% Match**\n"
        res += f"- **Demo Owner / Contact**: {prop.get('owner_name', 'Demo Seller')} ({prop.get('owner_phone', '+91 99999 00000')})\n\n"
        res += f"> ⚠️ **Demo contact information only.** *Fictional data for demonstration purposes.*"
        return res

    # 2. User Requirements Context
    if any(k in q for k in ["my requirement", "my preference", "what are my requirements", "my search", "what did i search"]):
        req = state.get("property_requirements")
        if not req:
            return "📝 **Your property requirements have not been submitted yet.** Please visit **🔎 Property Requirements** to specify your preferences."
        
        res = "### 📋 Your Current Property Search Requirements:\n"
        res += f"- **Property Type**: {req.get('property_type')}\n"
        res += f"- **City**: {req.get('city')}\n"
        res += f"- **Locality**: {req.get('locality')}\n"
        res += f"- **Budget Range**: ₹{req.get('min_budget', 0):.1f} Lakhs — ₹{req.get('max_budget', 0):.1f} Lakhs\n"
        
        if req.get("property_type") == "🏠 House / Apartment":
            res += f"- **Rooms (BHK)**: {int(req.get('rooms', 3))} BHK\n"
            res += f"- **Area Range**: {req.get('min_area', 0):,.0f} — {req.get('max_area', 0):,.0f} sq.ft.\n"
            res += f"- **Max Property Age**: {req.get('max_age', 15)} Years\n"
            amen_list = ", ".join(a.title() for a in req.get('amenities', []))
            res += f"- **Desired Amenities**: {amen_list if amen_list else 'None specified'}\n"
            res += f"- **Furnishing**: {req.get('furnishing', 'Any')}\n"
            res += f"- **Purpose**: {req.get('purpose', 'Self Use')}\n"
        else:
            res += f"- **Plot Area Range**: {req.get('min_plot_area', 0):,.0f} — {req.get('max_plot_area', 0):,.0f} sq.ft.\n"
            res += f"- **Land Type**: {req.get('land_type', 'Residential')}\n"
            res += f"- **Road Access**: {req.get('road_access', 'Required')}\n"
            res += f"- **Corner Plot**: {req.get('corner_plot', 'Preferred')}\n"
        return res

    # 3. Matching Properties Context
    if any(k in q for k in ["what properties matched", "matching properties", "how many matches", "show matching", "matched my requirements"]):
        if not state.get("property_requirements"):
            return "⚠️ Please define your **🔎 Property Requirements** first to see matching results."
        return "🏘️ You can view your personalized matching property cards in the Matching Properties step. Each property card displays the compatibility percentage, price, area, and verified matching reasons."

    # 4. What is HomeMetric?
    if any(k in q for k in ["what is homemetric", "about homemetric", "what does homemetric do", "tell me about homemetric"]):
        return ("🏠 **HomeMetric** is a Smart Property Valuation & Discovery Platform developed as a Machine Learning project.\n\n"
                "Key capabilities include:\n"
                "1. **User Profiling & Requirements Gathering**: Capturing personalized housing criteria.\n"
                "2. **Rule-Based Property Matching (0–100%)**: Finding and ranking compatible properties across 8 Indian cities.\n"
                "3. **Machine Learning House Valuation**: Using a pre-trained **Random Forest Regressor** ($R^2 = 83.54\%$) to estimate real estate valuation instantly.\n"
                "4. **Interactive AI Assistant**: Providing instant guidance across property terms, search context, and ML predictions.")

    # 5. How does matching work? / What is matching score?
    if any(k in q for k in ["matching work", "matching score", "how does property matching work", "how is match calculated", "compatibility"]):
        return ("🎯 **How Property Matching Works:**\n\n"
                "HomeMetric calculates a transparent **0–100% compatibility score** by evaluating your submitted requirements against available listings:\n"
                "- **City & Locality Match**: Up to +30%\n"
                "- **Budget Range Match**: Up to +20%\n"
                "- **Area (sq.ft.) Match**: Up to +15%\n"
                "- **BHK / Rooms Match**: Up to +15%\n"
                "- **Building Age Limit**: Up to +10%\n"
                "- **Preferred Amenities**: Up to +10% proportional to matches.\n\n"
                "> 💡 **Important Distinction**: The **Matching Score** is a rule-based search compatibility metric based on your requirements. It is **NOT** the ML Price Prediction.")

    # 6. How is house price predicted? / Random Forest
    if any(k in q for k in ["how is the house price predicted", "how is house price predicted", "predict price", "random forest", "how does ai price prediction work", "ml model", "algorithm", "valuation work"]):
        return ("🌲 **Machine Learning Valuation Engine:**\n\n"
                "HomeMetric uses the trained **Random Forest model** and preprocessing pipeline to estimate house prices:\n"
                "1. **Preprocessing Pipeline (`preprocessor.joblib`)**: Property features (`rooms`, `area_sqft`, `location`, `age`, `amenities`) are handled with median imputation, `StandardScaler`, and `OneHotEncoder`.\n"
                "2. **Random Forest Regressor (`random_forest.joblib`)**: An ensemble of 100 decision trees outputs the average estimated valuation in ₹ Lakhs / Crores ($R^2 = 83.54\%$).\n\n"
                "> ℹ️ **The AI estimate is an ML-based estimate from this project and is not a professional property valuation.**")

    # 7. Difference between listing price & AI predicted price
    if any(k in q for k in ["difference between listing price and ai", "difference between listing price and ai predicted price", "listing price vs predicted", "listing vs ai", "difference between listing"]):
        return ("📊 **Difference Between Listing Price and AI Predicted Price:**\n\n"
                "- **Demo Listing Price**: The asking price set by the property seller/builder in the demo listing database.\n"
                "- **AI Predicted Price**: The statistical market valuation estimated by the trained **Random Forest Machine Learning model** based on square footage, location baseline, building age, and amenities.\n\n"
                "In the **💰 ML Valuation Tool**, HomeMetric displays a side-by-side comparison showing whether the model estimate is higher, lower, or close to the demo listing price.")

    # 8. Real Estate Terminology (BHK, Price per sqft, etc.)
    if "bhk" in q:
        return ("🛏️ **BHK stands for Bedroom, Hall, and Kitchen.**\n\n"
                "- **1 BHK**: 1 Bedroom, 1 Living Room/Hall, 1 Kitchen\n"
                "- **2 BHK**: 2 Bedrooms, 1 Living Room/Hall, 1 Kitchen\n"
                "- **3 BHK**: 3 Bedrooms, 1 Living Room/Hall, 1 Kitchen\n\n"
                "In HomeMetric, you can specify your required BHK configuration under Property Requirements.")

    if any(k in q for k in ["price per square foot", "sqft rate", "rate per sqft", "per square foot", "price per sqft"]):
        return ("📐 **Price per Square Foot (₹ / sq.ft.):**\n\n"
                "This is the standard unit rate in Indian real estate calculated as:\n"
                "$$\\text{Rate per sq.ft.} = \\frac{\\text{Total Property Price in ₹}}{\\text{Total Built-up Area in sq.ft.}}$$\n\n"
                "It allows buyers to directly compare property rates across different apartment sizes and localities.")

    # 9. How to view details / Get AI Price estimate
    if any(k in q for k in ["view property details", "how can i view details", "how can i view property details", "open details", "view details"]):
        return ("🏠 **How to View Property Details:**\n"
                "1. Continue through the guided flow to **🏘️ Matching Properties**.\n"
                "2. Click **'View Details →'** on any listing card.\n"
                "3. The dedicated **🏠 Property Details** page will open with complete specifications, amenities chips, and demo owner contact information.")

    if any(k in q for k in ["get ai price estimate", "how can i get an ai price estimate", "how to get ai price estimate", "how to estimate", "ai estimate", "ai price estimate"]):
        last_pred = state.get("last_prediction")
        pred_info = ""
        if last_pred:
            pred_info = f"\n\n💡 **Your Last AI Estimate**: **₹{last_pred['price_in_lakhs']:.2f} Lakhs** ({last_pred['location']}, {last_pred['area_sqft']:,.0f} sq.ft., {int(last_pred['rooms'])} BHK)."

        return ("💰 **How to Get an AI Price Estimate:**\n"
                "1. **From Selected Property**: Open any property on the **🏠 Property Details** page and click **'💰 Get AI Price Estimate'** to auto-fill its features.\n"
                "2. **Manual Valuation**: Continue to **💰 AI Property Valuation**, customize features, and click **'🔮 Predict House Price'**."
                f"{pred_info}\n\n"
                "> ℹ️ *The AI estimate is an ML-based estimate from this project and is not a professional property valuation.*")

    # 10. Information required to search
    if any(k in q for k in ["what information do i need", "information needed", "search for a property", "information do i need to search"]):
        return ("📝 **Information Needed for Property Search:**\n"
                "To search for properties in HomeMetric, you need:\n"
                "1. **Property Type**: House / Apartment or Land / Plot\n"
                "2. **Target City & Locality**: (e.g. Bangalore, Mumbai, Delhi, etc.)\n"
                "3. **Budget Range**: Minimum & Maximum budget in ₹ Lakhs\n"
                "4. **Specifications**: BHK, Area in sq.ft., Max building age, and Preferred Amenities.")

    # 11. Fallback for Unsupported Questions
    return ("🤖 I can help with HomeMetric features, property matching, property details, and ML price estimation. Please ask me something related to these topics.")


# -------------------------------------------------------------
# 5. Session State Initialization
# -------------------------------------------------------------
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "profile_submitted" not in st.session_state:
    st.session_state["profile_submitted"] = False

if "profile_completed" not in st.session_state:
    st.session_state["profile_completed"] = False

if "property_type" not in st.session_state:
    st.session_state["property_type"] = "🏠 House / Apartment"

if "property_requirements" not in st.session_state:
    st.session_state["property_requirements"] = None

if "search_submitted" not in st.session_state:
    st.session_state["search_submitted"] = False

if "current_view" not in st.session_state:
    st.session_state["current_view"] = "requirements"

if "selected_property" not in st.session_state:
    st.session_state["selected_property"] = None

if "prefill_valuation" not in st.session_state:
    st.session_state["prefill_valuation"] = None

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "pending_chat_prompt" not in st.session_state:
    st.session_state["pending_chat_prompt"] = None


# -------------------------------------------------------------
# 6. Global Top Header (Compact & Elegant)
# -------------------------------------------------------------
st.markdown("""
<div class="hm-top-brand">
    <div class="hm-top-title">🏠 HomeMetric</div>
    <div class="hm-top-subtitle">Smart Property Valuation & Discovery Platform &nbsp; • &nbsp; Find. Compare. Estimate.</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 7. Main App Content
# The application follows a guided page-to-page flow using in-page buttons.
# -------------------------------------------------------------

# -------------------------------------------------------------
# 8. Page Routing
# -------------------------------------------------------------

# =============================================================
# VIEW 1: WELCOME / USER PROFILE PAGE (Step 2 & 8B)
# =============================================================
if not st.session_state["profile_completed"]:
    # Polished Product Onboarding Hero Card
    st.markdown("""
    <div class="hm-hero-card">
        <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
            👋 Welcome to HomeMetric
        </div>
        <div style="font-size: 0.98rem; font-weight: 600; color: #2563EB; margin-bottom: 8px;">
            Smart Property Valuation & Discovery Platform
        </div>
        <div style="color: #475569; font-size: 0.90rem; line-height: 1.45; margin-bottom: 12px;">
            Find properties based on your requirements and explore AI-powered price estimates across 8 Indian metropolitan cities.
        </div>
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <span class="type-tag">🔍 Intelligent Discovery</span>
            <span class="type-tag">🎯 Transparent Matching</span>
            <span class="type-tag">🌲 ML Price Estimates</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["profile_submitted"] and st.session_state["user_profile"]:
        user_name = st.session_state["user_profile"]["name"]
        st.success(f"🎉 Welcome, **{user_name}**! Your session profile is ready.")
        st.info("Click the button below to proceed to the Property Search module.")
        
        if st.button("Continue to Property Search →", type="primary", use_container_width=True):
            st.session_state["profile_completed"] = True
            st.session_state["current_view"] = "requirements"
            st.rerun()
    else:
        st.markdown('<div class="hm-card">', unsafe_allow_html=True)
        st.markdown("#### 👤 User Profile Setup")
        st.caption("Please provide your details to personalize your property search session.")
        
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input(
                "Full Name *",
                placeholder="e.g. Anusha Garnepally",
                help="Required. Used only during this session."
            )
        with col2:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25,
                step=1,
                help="Enter your age (18-100)."
            )

        gender = st.selectbox(
            "Gender",
            options=["Female", "Male", "Prefer not to say"],
            index=0,
            help="Select your gender."
        )

        agree = st.checkbox(
            "I agree to use these details for this application session.",
            value=False
        )

        st.write("")
        if st.button("Continue →", type="primary", use_container_width=True):
            if not full_name.strip():
                st.warning("⚠️ Please enter your full name to proceed.")
            elif not agree:
                st.warning("⚠️ Please check the agreement box to continue.")
            else:
                st.session_state["user_profile"] = {
                    "name": full_name.strip(),
                    "age": int(age),
                    "gender": gender,
                    "agreed": True
                }
                st.session_state["profile_submitted"] = True
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# VIEW 2: PROPERTY REQUIREMENTS PAGE (Step 3 & 8B)
# =============================================================
elif st.session_state["current_view"] == "requirements":
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Profile", use_container_width=True):
            st.session_state["profile_completed"] = False
            st.session_state["current_view"] = "requirements"
            st.rerun()

    st.markdown("""
    <div style="margin-bottom: 14px;">
        <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A;">🔎 Find Your Ideal Property</div>
        <div style="font-size: 0.88rem; color: #64748B; margin-top: 1px;">
            Tell us what you're looking for, and we'll find properties that match your requirements.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Property Type Selection
    st.markdown('<div class="hm-section-title">🏠 1. Property Category</div>', unsafe_allow_html=True)
    prop_type = st.radio(
        "Select Property Type:",
        options=["🏠 House / Apartment", "🌳 Land / Plot"],
        index=0 if st.session_state["property_type"] == "🏠 House / Apartment" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state["property_type"] = prop_type

    # Form for Requirements
    with st.form("requirements_form"):
        # 2. Location Section
        st.markdown('<div class="hm-section-title">📍 Location Preferences</div>', unsafe_allow_html=True)
        loc_col1, loc_col2 = st.columns(2)
        
        with loc_col1:
            city_options = ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune']
            city = st.selectbox(
                "Preferred City / Location *",
                options=city_options,
                index=1,  # Default: Bangalore
                help="Select the metropolitan city from our dataset coverage."
            )
        with loc_col2:
            locality = st.text_input(
                "Preferred Locality / Area *",
                value="Koramangala" if city == "Bangalore" else "",
                placeholder="e.g. Koramangala, Indiranagar, Whitefield, Bandra...",
                help="Enter preferred neighborhood or locality."
            )

        # 3. Budget Section
        st.markdown('<div class="hm-section-title">💰 Budget Range (in ₹ Lakhs)</div>', unsafe_allow_html=True)
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            min_budget = st.number_input(
                "Minimum Budget (₹ Lakhs) *",
                min_value=1.0,
                max_value=2000.0,
                value=35.0,
                step=5.0,
                help="Minimum amount you are willing to spend."
            )
        with b_col2:
            max_budget = st.number_input(
                "Maximum Budget (₹ Lakhs) *",
                min_value=1.0,
                max_value=2000.0,
                value=120.0,
                step=5.0,
                help="Maximum ceiling budget for your property."
            )

        # 4. Property Specifications Section
        if prop_type == "🏠 House / Apartment":
            st.markdown('<div class="hm-section-title">🏠 Property Configuration</div>', unsafe_allow_html=True)
            h_col1, h_col2 = st.columns(2)
            
            with h_col1:
                rooms = st.number_input(
                    "Number of BHK / Rooms *",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=1.0,
                    help="Required number of bedrooms."
                )
                min_area = st.number_input(
                    "Minimum Area (sq.ft.) *",
                    min_value=100.0,
                    max_value=10000.0,
                    value=800.0,
                    step=50.0
                )
                furnishing = st.selectbox(
                    "Furnishing Status:",
                    options=["Furnished", "Semi-Furnished", "Unfurnished", "Any"],
                    index=3
                )

            with h_col2:
                max_age = st.slider(
                    "Maximum Property Age (Years):",
                    min_value=0,
                    max_value=60,
                    value=15,
                    step=1,
                    help="Upper limit on building age."
                )
                max_area = st.number_input(
                    "Maximum Area (sq.ft.) *",
                    min_value=100.0,
                    max_value=10000.0,
                    value=2000.0,
                    step=50.0
                )
                purpose = st.selectbox(
                    "Primary Purchase Purpose:",
                    options=["Self Use", "Investment", "Rental"],
                    index=0
                )

            # 5. Amenities Section
            st.markdown('<div class="hm-section-title">✨ Amenities & Preferences</div>', unsafe_allow_html=True)
            amenity_icon_options = [
                "🌳 Garden",
                "🏋️ Gym",
                "🛗 Lift",
                "🚗 Parking",
                "🏊 Pool",
                "🔐 Security"
            ]
            selected_amenity_icons = st.multiselect(
                "Select Amenities Needed:",
                options=amenity_icon_options,
                default=["🛗 Lift", "🚗 Parking", "🔐 Security"],
                help="Select all preferred amenities."
            )

        else:
            # Land / Plot Requirements
            st.markdown('<div class="hm-section-title">🌳 Land Specifications</div>', unsafe_allow_html=True)
            l_col1, l_col2 = st.columns(2)
            
            with l_col1:
                min_plot_area = st.number_input(
                    "Minimum Plot Area (sq.ft.) *",
                    min_value=100.0,
                    max_value=50000.0,
                    value=1200.0,
                    step=100.0
                )
                land_type = st.selectbox(
                    "Land Classification *",
                    options=["Residential", "Commercial", "Agricultural"],
                    index=0
                )
                corner_plot = st.selectbox(
                    "Corner Plot Preference:",
                    options=["Preferred", "Not Required"],
                    index=1
                )

            with l_col2:
                max_plot_area = st.number_input(
                    "Maximum Plot Area (sq.ft.) *",
                    min_value=100.0,
                    max_value=50000.0,
                    value=3000.0,
                    step=100.0
                )
                road_access = st.selectbox(
                    "Road Access Requirement:",
                    options=["Required", "Not Required"],
                    index=0
                )

        st.write("")
        submit_search = st.form_submit_button("🔎 Find Matching Properties", type="primary", use_container_width=True)

    # Validation & Submission handling
    if submit_search:
        if not locality.strip():
            st.error("⚠️ Please specify your preferred locality/area.")
        elif min_budget <= 0 or max_budget <= 0:
            st.error("⚠️ Budget values must be positive numbers.")
        elif min_budget > max_budget:
            st.error(f"⚠️ Minimum budget (₹{min_budget}L) cannot be greater than Maximum budget (₹{max_budget}L).")
        elif prop_type == "🏠 House / Apartment" and min_area > max_area:
            st.error(f"⚠️ Minimum area ({min_area} sq.ft.) cannot be greater than Maximum area ({max_area} sq.ft.).")
        elif prop_type == "🌳 Land / Plot" and min_plot_area > max_plot_area:
            st.error(f"⚠️ Minimum plot area ({min_plot_area} sq.ft.) cannot be greater than Maximum plot area ({max_plot_area} sq.ft.).")
        else:
            if prop_type == "🏠 House / Apartment":
                clean_amenities = [a.split(" ", 1)[1].lower() for a in selected_amenity_icons if " " in a]
                st.session_state["property_requirements"] = {
                    "property_type": prop_type,
                    "city": city,
                    "locality": locality.strip(),
                    "min_budget": float(min_budget),
                    "max_budget": float(max_budget),
                    "rooms": float(rooms),
                    "min_area": float(min_area),
                    "max_area": float(max_area),
                    "max_age": int(max_age),
                    "amenities": clean_amenities,
                    "furnishing": furnishing,
                    "purpose": purpose
                }
            else:
                st.session_state["property_requirements"] = {
                    "property_type": prop_type,
                    "city": city,
                    "locality": locality.strip(),
                    "min_budget": float(min_budget),
                    "max_budget": float(max_budget),
                    "min_plot_area": float(min_plot_area),
                    "max_plot_area": float(max_plot_area),
                    "land_type": land_type,
                    "road_access": road_access,
                    "corner_plot": corner_plot
                }

            st.session_state["search_submitted"] = True
            st.session_state["current_view"] = "matching"
            st.rerun()

    if st.session_state.get("search_submitted") and st.session_state.get("property_requirements"):
        st.markdown("---")
        st.success("✅ Your search requirements are active in this session.")
        if st.button("View Matching Properties →", type="primary", use_container_width=True):
            st.session_state["current_view"] = "matching"
            st.rerun()


# =============================================================
# VIEW 3: MATCHING PROPERTIES RESULTS PAGE (Step 4 & 8B)
# =============================================================
elif st.session_state["current_view"] == "matching":
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Requirements", use_container_width=True):
            st.session_state["current_view"] = "requirements"
            st.rerun()

    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A;">🏘️ Matching Properties</div>
        <div style="font-size: 0.88rem; color: #64748B; margin-top: 1px;">
            Ranked listings matching your location, budget, and specification preferences.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-banner">
        ⚠️ <b>Demo Property Listings</b>: Fictional demonstration listings created exclusively for property discovery demonstration.
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("property_requirements"):
        st.warning("⚠️ You haven't set any property requirements yet.")
        if st.button("← Set Property Requirements", type="primary"):
            st.session_state["current_view"] = "requirements"
            st.rerun()
    else:
        req = st.session_state["property_requirements"]

        scored_props = []
        for idx, row in demo_df.iterrows():
            score, reasons = calculate_match_score(row, req)
            row_dict = row.to_dict()
            row_dict["match_score"] = score
            row_dict["match_reasons"] = reasons
            scored_props.append(row_dict)

        results_df = pd.DataFrame(scored_props)

        # Filters & Sort Bar
        with st.expander("⚙️ Refine & Sort Matching Results", expanded=False):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=["Best Match", "Lowest Price", "Highest Price", "Largest Area"],
                    index=0
                )
            with f_col2:
                type_filter = st.selectbox(
                    "Property Type:",
                    options=["All Types", "House / Apartment", "Land / Plot"],
                    index=0
                )
            with f_col3:
                max_p_filter = st.number_input(
                    "Max Price (₹ Lakhs):",
                    min_value=10.0,
                    max_value=500.0,
                    value=float(req["max_budget"]) * 1.5,
                    step=10.0
                )
            with f_col4:
                min_a_filter = st.number_input(
                    "Min Area (sq.ft.):",
                    min_value=100.0,
                    max_value=10000.0,
                    value=500.0,
                    step=100.0
                )

        filtered_df = results_df.copy()

        if type_filter != "All Types":
            filtered_df = filtered_df[filtered_df["property_type"] == type_filter]

        filtered_df = filtered_df[
            (filtered_df["price_lakhs"] <= max_p_filter) &
            (filtered_df["area_sqft"] >= min_a_filter)
        ]

        if sort_by == "Best Match":
            filtered_df = filtered_df.sort_values(by="match_score", ascending=False)
        elif sort_by == "Lowest Price":
            filtered_df = filtered_df.sort_values(by="price_lakhs", ascending=True)
        elif sort_by == "Highest Price":
            filtered_df = filtered_df.sort_values(by="price_lakhs", ascending=False)
        elif sort_by == "Largest Area":
            filtered_df = filtered_df.sort_values(by="area_sqft", ascending=False)

        st.markdown(f"**Found {len(filtered_df)} matching properties:**")

        if len(filtered_df) == 0:
            st.error("😕 No properties found matching all your requirements.")
            st.info("💡 Try adjusting your budget, location, or area preferences in the filter bar above.")
            if st.button("← Modify Requirements"):
                st.session_state["current_view"] = "requirements"
                st.rerun()
        else:
            for idx, prop in filtered_df.iterrows():
                score = prop["match_score"]
                pill_class = "match-pill-high" if score >= 70 else "match-pill-med"
                rate_per_sqft = (prop["price_lakhs"] * 1e5) / prop["area_sqft"]
                
                # Clean structured property card layout (Step 8B standard)
                with st.container():
                    st.markdown(f"""
                    <div class="property-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div class="prop-title">{prop['title']}</div>
                                <div class="prop-location">📍 {prop['locality']}, {prop['city']} &nbsp;•&nbsp; <span class="type-tag">{prop['property_type']}</span></div>
                            </div>
                            <div class="{pill_class}">🎯 {score}% Match</div>
                        </div>
                        <div class="card-divider"></div>
                    """, unsafe_allow_html=True)

                    # 2x2 Grid: Price | Area | BHK | Status
                    grid_c1, grid_c2 = st.columns(2)
                    with grid_c1:
                        st.markdown(f'<div class="prop-price">₹{prop["price_lakhs"]:.1f} Lakhs</div>', unsafe_allow_html=True)
                        st.caption(f"Rate: ₹{rate_per_sqft:,.0f} / sq.ft.")
                    with grid_c2:
                        st.markdown(f'<div class="prop-meta">📐 {prop["area_sqft"]:,.0f} sq.ft.</div>', unsafe_allow_html=True)
                        if not pd.isna(prop.get("rooms")) and prop.get("rooms"):
                            st.caption(f"🛏️ {int(prop['rooms'])} BHK • Age: {int(prop.get('age', 0))} yrs")
                        elif not pd.isna(prop.get("land_type")) and prop.get("land_type"):
                            st.caption(f"🌳 {prop['land_type']} Classification")

                    st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)

                    # Facilities & Description
                    amenities_list = str(prop.get("amenities", "")).replace(";", ", ").title()
                    if amenities_list and amenities_list != "Nan":
                        st.markdown(f"<div style='font-size: 0.84rem; color: #334155; margin-bottom: 4px;'><b>✨ Facilities:</b> {amenities_list}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p style='color: #64748B; font-size: 0.86rem; margin-top: 4px; margin-bottom: 10px; line-height: 1.4;'>{prop['description']}</p>", unsafe_allow_html=True)

                    with st.expander("🎯 Why does this property match?"):
                        for r in prop["match_reasons"]:
                            st.write(r)
                        st.caption("Compatibility score calculated transparently based on your submitted search criteria.")

                    btn_key = f"view_btn_{prop['property_id']}"
                    if st.button(f"View Details for {prop['title'][:28]}... →", key=btn_key, type="primary", use_container_width=True):
                        st.session_state["selected_property"] = prop.to_dict()
                        st.session_state["current_view"] = "details"
                        st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# VIEW 4: DEDICATED PROPERTY DETAILS PAGE (Step 5 & 8B)
# =============================================================
elif st.session_state["current_view"] == "details":
    if not st.session_state.get("selected_property"):
        st.warning("⚠️ No property selected. Please choose a property from Matching Properties.")
        if st.button("← Go to Matching Properties", type="primary"):
            st.session_state["current_view"] = "matching"
            st.rerun()
    else:
        prop = st.session_state["selected_property"]
        score = prop.get("match_score", 90)
        pill_class = "match-pill-high" if score >= 70 else "match-pill-med"
        price_lakhs = float(prop["price_lakhs"])
        area_sqft = float(prop["area_sqft"])
        rate_per_sqft = (price_lakhs * 1e5) / area_sqft

        # Visually Strong Property Header
        st.markdown(f"""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">🏠 {prop["title"]}</div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="demo-tag">Demo Property</span>
                <span class="type-tag">{prop['property_type']}</span>
                <span class="{pill_class}">🎯 {score}% Match</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="demo-banner">
            ℹ️ <b>Demo Property Notice</b>: Fictional demonstration listing created for project evaluation.
        </div>
        """, unsafe_allow_html=True)

        # 1. Primary Metrics Card (Price & Area)
        st.markdown("##### 💰 Pricing & Key Metrics")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Listing Price", f"₹{price_lakhs:.1f} Lakhs", f"₹{price_lakhs*1e5:,.0f}")
        p_col2.metric("Total Area", f"{area_sqft:,.0f} sq.ft.")
        p_col3.metric("Rate / Sq.Ft.", f"₹{rate_per_sqft:,.0f}/sqft")
        p_col4.metric("Location", f"{prop['locality']}", f"{prop['city']}")

        st.markdown("---")

        # 2. Detailed Specifications Grid
        st.markdown("##### 📋 Property Specifications")
        spec_col1, spec_col2 = st.columns(2)

        with spec_col1:
            st.markdown(f"**City:** {prop['city']}")
            st.markdown(f"**Locality / Area:** {prop['locality']}")
            st.markdown(f"**Property Type:** {prop['property_type']}")
            if prop["property_type"] == "House / Apartment":
                rooms_val = int(prop['rooms']) if not pd.isna(prop.get('rooms')) else "N/A"
                st.markdown(f"**Configuration:** {rooms_val} BHK")
                st.markdown(f"**Furnishing:** {prop.get('furnishing', 'Semi-Furnished')}")
            else:
                st.markdown(f"**Land Classification:** {prop.get('land_type', 'Residential')}")
                st.markdown(f"**Road Access:** {prop.get('road_access', 'Available')}")

        with spec_col2:
            st.markdown(f"**Total Area:** {area_sqft:,.0f} sq.ft.")
            if prop["property_type"] == "House / Apartment":
                age_val = int(prop['age']) if not pd.isna(prop.get('age')) else 0
                st.markdown(f"**Building Age:** {age_val} Years")
                st.markdown(f"**Purchase Purpose:** {prop.get('purpose', 'Self Use')}")
            else:
                st.markdown(f"**Corner Plot:** {prop.get('corner_plot', 'No')}")
                st.markdown(f"**Intended Purpose:** {prop.get('purpose', 'Investment')}")

        # 3. Amenities Section (if House/Apartment)
        if prop["property_type"] == "House / Apartment":
            st.markdown("##### ✨ Available Facilities & Amenities")
            amenities_raw = str(prop.get("amenities", "")).split(";")
            amenity_icons_map = {
                "garden": "🌳 Garden",
                "gym": "🏋️ Gym",
                "lift": "🛗 Lift",
                "parking": "🚗 Parking",
                "pool": "🏊 Swimming Pool",
                "security": "🔐 24/7 Security"
            }
            amenity_chips = []
            for a in amenities_raw:
                a_clean = a.strip().lower()
                if a_clean in amenity_icons_map:
                    amenity_chips.append(amenity_icons_map[a_clean])
                elif a_clean:
                    amenity_chips.append(f"✓ {a_clean.title()}")

            if amenity_chips:
                st.markdown(" ".join([f"<span class='amenity-chip'>{chip}</span>" for chip in amenity_chips]), unsafe_allow_html=True)
            else:
                st.write("Standard society amenities.")

        # 4. Property Description
        st.markdown("##### 📝 Property Description")
        st.write(prop.get("description", "Premium property located in prime residential neighborhood."))

        # 5. Matching Criteria Breakdown
        st.markdown("##### 🎯 Why This Property Matches Your Requirements")
        reasons = prop.get("match_reasons", [])
        if reasons:
            for r in reasons:
                st.write(r)
        else:
            st.write(f"✓ High criteria alignment ({score}% overall match) with your preferred parameters.")

        st.markdown("---")

        # 6. Fictional Demo Owner / Agent Contact Section
        st.markdown("##### 📞 Demo Owner / Agent Contact")
        st.markdown(f"""
        <div class="contact-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #065F46; margin-bottom: 4px;">
                👤 Contact: {prop.get('owner_name', 'Demo Contact - Fictional Seller')}
            </div>
            <div style="font-size: 0.95rem; color: #047857; margin-bottom: 6px;">
                📱 Phone: <b>{prop.get('owner_phone', '+91 99999 00000 (Demo)')}</b> &nbsp;|&nbsp; 🕒 Available: 10:00 AM - 6:00 PM
            </div>
            <div style="font-size: 0.82rem; color: #64748B;">
                ⚠️ <i>This is fictional demo contact information for demonstration purposes only.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Page-to-page action buttons
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("← Back to Matching Properties", use_container_width=True):
                st.session_state["current_view"] = "matching"
                st.rerun()

        with btn_col2:
            if st.button("💰 Get AI Price Estimate", type="primary", use_container_width=True):
                p_amenities = [a.strip().lower() for a in str(prop.get("amenities", "")).split(";") if a.strip()]
                st.session_state["prefill_valuation"] = {
                    "title": prop["title"],
                    "city": prop["city"],
                    "locality": prop["locality"],
                    "price_lakhs": float(prop["price_lakhs"]),
                    "area_sqft": float(prop["area_sqft"]),
                    "rooms": float(prop["rooms"]) if not pd.isna(prop.get("rooms")) and prop.get("rooms") else 3.0,
                    "age": float(prop["age"]) if not pd.isna(prop.get("age")) and prop.get("age") else 5.0,
                    "amenities": p_amenities,
                    "match_score": prop.get("match_score", 90)
                }
                st.session_state["current_view"] = "valuation"
                st.rerun()

elif st.session_state["current_view"] == "valuation":
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Property Details", use_container_width=True):
            st.session_state["current_view"] = "details"
            st.rerun()

# =============================================================
# VIEW 5: ENHANCED ML PROPERTY VALUATION TOOL (Step 6 & 8B)
# ===========================================================#
        
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A;">💰 AI Property Valuation</div>
        <div style="font-size: 0.88rem; color: #64748B; margin-top: 1px;">
            Estimate property value using the trained HomeMetric prediction model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Check for pre-filled data from Property Details
    prefill = st.session_state.get("prefill_valuation")
    if prefill:
        st.markdown(f"""
        <div class="prefill-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #1E40AF;">
                    🏠 Estimating Selected Demo Property
                </div>
                <span class="match-pill-high">🎯 {prefill.get('match_score', 90)}% Match</span>
            </div>
            <div style="font-size: 1.0rem; font-weight: 600; color: #0F172A;">{prefill['title']}</div>
            <div style="color: #475569; font-size: 0.88rem; margin-top: 2px;">
                📍 <b>Location:</b> {prefill['locality']}, {prefill['city']} &nbsp;|&nbsp; 
                📐 <b>Area:</b> {prefill['area_sqft']:,.0f} sq.ft. &nbsp;|&nbsp; 
                💰 <b>Listed Price:</b> <span style="color: #1E40AF; font-weight: 700;">₹{prefill['price_lakhs']:.1f} Lakhs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Clear Selected Property (Switch to Manual Valuation Mode)", use_container_width=True):
            st.session_state["prefill_valuation"] = None
            st.rerun()
        st.write("")

    # 2. Form for ML valuation inputs
    with st.form("prediction_form"):
        st.markdown('<div class="hm-section-title">1. Property Characteristics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        location_options = ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune']
        default_loc_idx = 1
        if prefill and prefill.get("city") in location_options:
            default_loc_idx = location_options.index(prefill["city"])

        with col1:
            location = st.selectbox(
                "City / Location:",
                options=location_options,
                index=default_loc_idx,
                help="Select the metropolitan city where the property is located."
            )

            default_area = float(prefill["area_sqft"]) if prefill else 1200.0
            area_sqft = st.number_input(
                "Living Area (in Square Feet):",
                min_value=100.0,
                max_value=10000.0,
                value=default_area,
                step=50.0,
                help="Total built-up/carpet area of the house in sq.ft."
            )

        with col2:
            default_rooms = float(prefill["rooms"]) if prefill else 3.0
            rooms = st.number_input(
                "Number of Rooms (BHK):",
                min_value=1.0,
                max_value=10.0,
                value=default_rooms,
                step=1.0,
                help="Total number of bedrooms / rooms in the property."
            )
            default_age = float(prefill["age"]) if prefill else 5.0
            age = st.number_input(
                "Property Age (in Years):",
                min_value=0.0,
                max_value=80.0,
                value=default_age,
                step=1.0,
                help="Age of the building since construction (0 for new construction)."
            )
        st.markdown('<div class="hm-section-title">2. Available Amenities</div>', unsafe_allow_html=True)
        available_amenity_options = preprocessor["sorted_amenities"]
        
        default_amenities = ["lift", "parking", "security"]
        if prefill and prefill.get("amenities"):
            default_amenities = [a for a in prefill["amenities"] if a in available_amenity_options]

        selected_amenities = st.multiselect(
            "Select Amenities Available in Property / Society:",
            options=available_amenity_options,
            default=default_amenities,
            help="Select all facilities that apply."
        )

        submitted = st.form_submit_button("🔮 Predict House Price", type="primary", use_container_width=True)

    # 3. Prediction Pipeline Execution & Comparison Display
    if submitted:
        if area_sqft <= 0:
            st.error("⚠️ Living area must be greater than 0 sq.ft.")
        elif rooms <= 0:
            st.error("⚠️ Number of rooms must be at least 1.")
        else:
            with st.spinner("Processing input features and computing valuation with Random Forest model..."):
                num_input = np.array([[rooms, area_sqft, age]])
                num_imputed = preprocessor["num_imputer"].transform(num_input)
                num_scaled = preprocessor["scaler"].transform(num_imputed)

                loc_input = pd.DataFrame([[location]], columns=["location"])
                loc_encoded = preprocessor["ohe"].transform(loc_input)

                amenity_flags = np.array([[
                    1.0 if amen in selected_amenities else 0.0
                    for amen in preprocessor["sorted_amenities"]
                ]])

                feature_vector = np.hstack([num_scaled, loc_encoded, amenity_flags])
                input_df = pd.DataFrame(feature_vector, columns=preprocessor["feature_names"])

                predicted_price = float(models_dict["Random Forest (Best: R² = 83.5%)"].predict(input_df)[0])
                predicted_price = max(0.0, predicted_price)

                price_in_lakhs = predicted_price / 1e5
                price_in_crores = predicted_price / 1e7
                price_per_sqft = predicted_price / area_sqft
                formatted_price_inr = f"₹{predicted_price:,.2f}"

                if price_in_crores >= 1.0:
                    readable_price = f"₹{price_in_crores:.2f} Crore"
                else:
                    readable_price = f"₹{price_in_lakhs:.2f} Lakhs"

                # Store last prediction in session state for chatbot context
                st.session_state["last_prediction"] = {
                    "location": location,
                    "area_sqft": area_sqft,
                    "rooms": rooms,
                    "age": age,
                    "amenities": selected_amenities,
                    "predicted_price": predicted_price,
                    "price_in_lakhs": price_in_lakhs,
                    "price_per_sqft": price_per_sqft,
                    "model_name": "Random Forest"
                }

            # Display AI Prediction Hero Card
            st.markdown(f"""
            <div class="price-card">
                <div class="price-title">AI ESTIMATED MARKET VALUATION (RANDOM FOREST)</div>
                <div class="price-value">{readable_price}</div>
                <div class="price-sub">Exact Valuation: <b>{formatted_price_inr}</b> &nbsp;|&nbsp; Approx. <b>₹{price_per_sqft:,.0f} / sq.ft.</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Supporting Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Selected City", value=location)
            m2.metric(label="Unit Rate", value=f"₹{price_per_sqft:,.0f}/sqft")
            m3.metric(label="Total Amenities", value=f"{len(selected_amenities)} facilities")

            # Comparison Section (if prefilled from demo listing)
            if prefill and "price_lakhs" in prefill:
                listed_p = float(prefill["price_lakhs"])
                diff_lakhs = price_in_lakhs - listed_p
                diff_abs = abs(diff_lakhs)

                if price_in_lakhs > listed_p * 1.02:
                    interp_text = "The model estimate is higher than the demo listing price."
                    interp_color = "#1E40AF"
                elif price_in_lakhs < listed_p * 0.98:
                    interp_text = "The model estimate is lower than the demo listing price."
                    interp_color = "#92400E"
                else:
                    interp_text = "The model estimate is close to the demo listing price."
                    interp_color = "#065F46"

                st.markdown(f"""
                <div class="comparison-card">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;">
                        📊 Listing Price vs. AI Estimated Valuation Comparison
                    </div>
                    <div style="display: flex; justify-content: space-around; text-align: center; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 0.78rem; color: #64748B; font-weight: 700;">DEMO LISTED PRICE</div>
                            <div style="font-size: 1.35rem; font-weight: 800; color: #334155;">₹{listed_p:.1f} Lakhs</div>
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; color: #64748B; font-weight: 700;">AI PREDICTED PRICE</div>
                            <div style="font-size: 1.35rem; font-weight: 800; color: #1D4ED8;">₹{price_in_lakhs:.1f} Lakhs</div>
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; color: #64748B; font-weight: 700;">NUMERICAL DIFFERENCE</div>
                            <div style="font-size: 1.35rem; font-weight: 800; color: #475569;">₹{diff_abs:.1f} Lakhs</div>
                        </div>
                    </div>
                    <div style="background: white; border: 1px solid #E2E8F0; padding: 8px 12px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; color: {interp_color};">
                        📌 <b>Analysis</b>: {interp_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption("ℹ️ **Disclaimer**: AI estimate is generated by the trained Random Forest model using the project's housing dataset. It is an estimated value for demonstration and should not be treated as a professional property valuation.")

            with st.expander("🔍 View Internal Preprocessed Feature Vector (For ML Viva Demo)"):
                st.write("This shows the exact 17 numerical values passed into the ML model after imputation, scaling, and binary encoding:")
                st.dataframe(input_df.style.format(precision=3))

    st.markdown("---")
    with st.expander("ℹ️ How Does This Prediction System Work? (Project Architecture)"):
        st.markdown("""
        ### End-to-End Prediction Pipeline Architecture:
        1. **User Input**: The user inputs raw property features (`rooms`, `area_sqft`, `location`, `age`, `amenities`).
        2. **Preprocessing Pipeline (`preprocessor.joblib`)**:
           * **Numerical Pipeline**: Missing checks $\\rightarrow$ Median Imputer $\\rightarrow$ `StandardScaler` ($z = \\frac{x - \\mu}{\\sigma}$).
           * **Location Pipeline**: `OneHotEncoder` transforms city into 8 binary indicator columns.
           * **Amenities Pipeline**: Multi-label text splitter converts selected tags into 6 binary flags.
        3. **Trained Model (`random_forest.joblib`)**:
           * The processed 17-dimensional vector is passed to the ensemble of 100 decision trees.
           * Each tree outputs its price estimate, and the forest returns the mean prediction.
        4. **Output Display**: The raw numeric prediction is formatted into intuitive Indian currency units (₹ Lakhs / Crores).
        """)


# =============================================================
# VIEW 6: AI PROPERTY ASSISTANT CHATBOT (Step 7 & 8B)
# =============================================================
elif st.session_state["current_view"] == "assistant":
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="font-size: 1.45rem; font-weight: 800; color: #0F172A;">🤖 HomeMetric AI Property Assistant</div>
        <div style="font-size: 0.88rem; color: #64748B; margin-top: 1px;">
            Your intelligent, context-aware companion for property discovery & ML valuations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome Card
    st.markdown("""
    <div class="hm-hero-card" style="padding: 14px 18px; margin-bottom: 14px;">
        <div style="font-size: 1.02rem; font-weight: 700; color: #1E40AF; margin-bottom: 2px;">
            👋 Hi! I’m the HomeMetric Property Assistant.
        </div>
        <div style="color: #475569; font-size: 0.88rem;">
            I can help you understand your property requirements, matching results, property details, and AI price estimates.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clear Chat Action Controls Header
    chat_ctrl_col1, chat_ctrl_col2 = st.columns([3, 1])
    with chat_ctrl_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["pending_chat_prompt"] = None
            st.rerun()

    # Suggested Prompts (Quick Click Buttons)
    st.markdown("###### 💡 Suggested Questions:")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if st.button("❓ How does property matching work?", use_container_width=True):
            st.session_state["pending_chat_prompt"] = "How does property matching work?"
        if st.button("🏠 Tell me about my selected property", use_container_width=True):
            st.session_state["pending_chat_prompt"] = "Tell me about my selected property"
    with s_col2:
        if st.button("📋 What are my requirements?", use_container_width=True):
            st.session_state["pending_chat_prompt"] = "What are my requirements?"
        if st.button("🌲 How does AI price prediction work?", use_container_width=True):
            st.session_state["pending_chat_prompt"] = "How does AI price prediction work?"

    st.markdown("---")

    # Render Active Chat History
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending suggested question if triggered via button
    user_input = None
    if st.session_state.get("pending_chat_prompt"):
        user_input = st.session_state["pending_chat_prompt"]
        st.session_state["pending_chat_prompt"] = None

    # Chat Input Box
    typed_input = st.chat_input("Ask a question about HomeMetric, property search, or valuation...")
    if typed_input:
        user_input = typed_input

    # Process and Respond to User Input
    if user_input:
        # Append User Message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Contextual Assistant Response
        assistant_reply = get_assistant_response(user_input, st.session_state)
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

    # Educational Footer Disclaimer
    st.markdown("---")
    st.caption("ℹ️ **Disclaimer**: HomeMetric AI Property Assistant provides project-based information and should not be considered professional real-estate, financial, or valuation advice.")


