import os
import streamlit as st
import pandas as pd
import joblib
import pickle
import traceback
from pathlib import Path
from datetime import datetime

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛡️ Fraud Detection System</p>', unsafe_allow_html=True)

# Configuration
MODEL_BASENAME = "fraud_detector_final"
MODEL_EXT_CANDIDATES = [".pkl", ".joblib"]
DATA_BASENAME = "FRAUD DETECTION"
DATA_EXT_CANDIDATES = [".csv", ".xlsx", ".xls"]

def get_file_list():
    """Get list of files in current directory"""
    try:
        cwd = Path.cwd()
        return sorted([f.name for f in cwd.iterdir() if f.is_file()])
    except Exception as e:
        st.error(f"Failed to list directory: {e}")
        return []

def find_model_file():
    """Return filepath of the model if found"""
    cwd = Path.cwd()
    files = get_file_list()
    
    for ext in MODEL_EXT_CANDIDATES:
        model_file = MODEL_BASENAME + ext
        for f in files:
            if f.lower() == model_file.lower():
                return cwd / f
    
    for f in files:
        if MODEL_BASENAME.lower() in f.lower():
            return cwd / f
    return None

def find_data_file():
    """Return filepath and ext if found"""
    cwd = Path.cwd()
    files = get_file_list()
    
    for ext in DATA_EXT_CANDIDATES:
        data_file = DATA_BASENAME + ext
        for f in files:
            if f.lower() == data_file.lower():
                return cwd / f, ext
    
    for f in files:
        if DATA_BASENAME.lower() in f.lower():
            ext = Path(f).suffix.lower()
            return cwd / f, ext
    return None, None

@st.cache_resource
def load_model(path):
    """Load model with joblib or pickle with multiple encoding attempts"""
    import sys
    
    # Try joblib first
    try:
        model = joblib.load(path)
        return model, "joblib"
    except Exception as joblib_error:
        st.warning(f"Joblib loading failed: {str(joblib_error)[:100]}")
    
    # Try fixing pickle protocol issues
    try:
        import io
        import pickle5 as pickle_temp
        with open(path, "rb") as f:
            model = pickle_temp.load(f)
        return model, "pickle5"
    except:
        pass
    
    # Try pickle with fix_imports
    try:
        with open(path, "rb") as f:
            model = pickle.load(f, fix_imports=True, encoding='latin1')
        return model, "pickle (latin1 + fix_imports)"
    except Exception as fix_imports_error:
        st.warning(f"Pickle with fix_imports failed: {str(fix_imports_error)[:100]}")
    
    # Try pickle with latin1 encoding
    try:
        with open(path, "rb") as f:
            model = pickle.load(f, encoding='latin1')
        return model, "pickle (latin1)"
    except Exception as latin1_error:
        st.warning(f"Pickle with latin1 failed: {str(latin1_error)[:100]}")
    
    # Try pickle with bytes encoding
    try:
        with open(path, "rb") as f:
            model = pickle.load(f, encoding='bytes')
        return model, "pickle (bytes)"
    except Exception as bytes_error:
        st.warning(f"Pickle with bytes failed: {str(bytes_error)[:100]}")
    
    # Try pickle with ASCII encoding
    try:
        with open(path, "rb") as f:
            model = pickle.load(f, encoding='ASCII')
        return model, "pickle (ASCII)"
    except Exception as ascii_error:
        st.warning(f"Pickle with ASCII failed: {str(ascii_error)[:100]}")
    
    st.error("❌ All loading methods failed. The model file is incompatible.")
    st.error("**Solution Required:** You need to retrain and save the model.")
    
    return None, None

@st.cache_data
def load_dataset(path, ext):
    """Load CSV or Excel dataset"""
    try:
        if ext == ".csv" or str(path).lower().endswith(".csv"):
            df = pd.read_csv(path, low_memory=False)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def calculate_fraud_rates(df, column, target='is_fraud'):
    """Calculate fraud rate for categorical columns"""
    if column not in df.columns or target not in df.columns:
        return {}
    fraud_rates = df.groupby(column)[target].mean().to_dict()
    return fraud_rates

def engineer_features(input_data, df=None):
    """
    Create all engineered features that the model expects
    Based on your dataset structure
    """
    data = input_data.copy()
    
    # Extract datetime features if transaction_date exists
    if 'transaction_date' in data.columns:
        data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')
        data['tx_hour'] = data['transaction_date'].dt.hour
        data['tx_weekday'] = data['transaction_date'].dt.dayofweek
        data['is_weekend'] = (data['tx_weekday'] >= 5).astype(int)
        data['is_night'] = ((data['tx_hour'] >= 22) | (data['tx_hour'] <= 6)).astype(int)
    else:
        # Use defaults if no date column
        data['tx_hour'] = 12
        data['tx_weekday'] = 2
        data['is_weekend'] = 0
        data['is_night'] = 0
    
    # Create interaction features
    if 'amount' in data.columns and 'customer_age' in data.columns:
        data['amount_per_age'] = data['amount'] / (data['customer_age'] + 1)
    else:
        data['amount_per_age'] = 0
    
    # Calculate fraud rates from training data
    if df is not None:
        # Purchase category fraud rate
        if 'purchase_category' in data.columns and 'purchase_category' in df.columns:
            fraud_rates = calculate_fraud_rates(df, 'purchase_category')
            data['purchase_category_fraud_rate'] = data['purchase_category'].map(fraud_rates).fillna(0.5)
        else:
            data['purchase_category_fraud_rate'] = 0.5
        
        # Location fraud rate
        if 'location' in data.columns and 'location' in df.columns:
            fraud_rates = calculate_fraud_rates(df, 'location')
            data['location_fraud_rate'] = data['location'].map(fraud_rates).fillna(0.5)
        else:
            data['location_fraud_rate'] = 0.5
    else:
        # Use default values if no training data available
        data['purchase_category_fraud_rate'] = 0.5
        data['location_fraud_rate'] = 0.5
    
    return data

# Sidebar
with st.sidebar:
    st.header("📊 System Information")
    
    with st.expander("🔍 Debug Info", expanded=False):
        st.write("**Working Directory:**")
        st.code(str(Path.cwd()))
        
        st.write("**Files in Directory:**")
        files = get_file_list()
        if files:
            for f in files:
                st.text(f"• {f}")
        else:
            st.warning("No files found")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Ensure model file is in the directory
    2. Enter transaction details
    3. Click **Predict** to check for fraud
    """)

# Load model
model_path = find_model_file()
model = None
model_method = None

if model_path and model_path.exists():
    st.success(f"✅ Model found: `{model_path.name}`")
    model, model_method = load_model(model_path)
    if model:
        st.info(f"📦 Loaded using: **{model_method}**")
else:
    st.error(f"❌ Model not found. Please place `{MODEL_BASENAME}.pkl` or `.joblib` in: {Path.cwd()}")

# Load dataset (for fraud rate calculations and getting unique values)
data_path, data_ext = find_data_file()
df = None

if data_path and data_path.exists():
    st.success(f"✅ Dataset found: `{data_path.name}`")
    df = load_dataset(data_path, data_ext)
    
    if df is not None:
        with st.expander("📊 Dataset Preview", expanded=False):
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Columns:**", list(df.columns))
else:
    st.warning("⚠️ Dataset not found. Using default values.")

# Stop if no model
if model is None:
    st.error("🚫 Cannot proceed without a model.")
    
    st.markdown("---")
    st.markdown("### 🔧 How to Fix This Issue")
    st.markdown("""
    Your model file is incompatible with the current Python environment. You need to retrain and save the model properly.
    
    **Create a new Python script called `retrain_model.py` and run it:**
    """)
    
    st.code("""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('FRAUD DETECTION.csv')

# Check column names
print("Columns:", df.columns.tolist())

# Identify target column (adjust if needed)
target_col = 'is_fraud'  # Change this to match your actual target column name

# Prepare features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

# Save model using joblib (recommended)
joblib.dump(model, 'fraud_detector_final.pkl', compress=3)
print("Model saved as 'fraud_detector_final.pkl'")

# Verify the model can be loaded
loaded_model = joblib.load('fraud_detector_final.pkl')
print("✓ Model verified - can be loaded successfully!")
    """, language="python")
    
    st.info("📝 **Steps:**\n1. Create `retrain_model.py` with the code above\n2. Update the target column name if needed\n3. Run: `python retrain_model.py`\n4. Refresh this Streamlit app")
    
    st.stop()

# Get unique values from dataset for dropdowns
def get_unique_values(df, column, default_values):
    """Get unique values from dataset or use defaults"""
    if df is not None and column in df.columns:
        unique_vals = df[column].dropna().unique().tolist()
        return sorted([str(v) for v in unique_vals])
    return default_values

# Define options based on your dataset
card_type_options = get_unique_values(df, 'card_type', ['Rupay', 'MasterCard', 'Visa'])
location_options = get_unique_values(df, 'location', ['Bangalore', 'Surat', 'Hyderabad', 'Mumbai', 'Kolkata', 'Jaipur', 'Delhi', 'Chennai', 'Pune', 'Ahmedabad'])
purchase_category_options = get_unique_values(df, 'purchase_category', ['POS', 'Digital'])
fraud_type_options = get_unique_values(df, 'fraud_type', ['Identity theft', 'Malware', 'Payment card fraud', 'scam', 'phishing'])

# Prediction Interface
st.markdown("---")
st.header("🔮 Make Prediction")

with st.form("prediction_form"):
    st.subheader("Enter Transaction Details")
    
    input_values = {}
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Transaction Info**")
        
        input_values['transaction_id'] = st.number_input(
            "Transaction ID",
            min_value=1,
            value=100000,
            step=1
        )
        
        input_values['customer_id'] = st.number_input(
            "Customer ID",
            min_value=1000,
            value=2000,
            step=1
        )
        
        input_values['merchant_id'] = st.number_input(
            "Merchant ID",
            min_value=2000,
            value=2050,
            step=1
        )
        
        input_values['amount'] = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            step=10.0
        )
        
        input_values['transaction_date'] = st.text_input(
            "Transaction Date (MM/DD/YYYY or YYYY-MM-DD)",
            value=datetime.now().strftime("%m/%d/%Y")
        )
        
        input_values['card_type'] = st.selectbox(
            "Card Type",
            options=card_type_options
        )
    
    with col2:
        st.markdown("**👤 Customer & Location Info**")
        
        input_values['location'] = st.selectbox(
            "Transaction Location",
            options=location_options
        )
        
        input_values['purchase_category'] = st.selectbox(
            "Purchase Category",
            options=purchase_category_options
        )
        
        input_values['customer_age'] = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )
        
        input_values['fraud_type'] = st.selectbox(
            "Fraud Type (for reference)",
            options=fraud_type_options,
            help="This is typically unknown during prediction, but required by the model"
        )
    
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

if submitted:
    try:
        # Create base dataframe with exact column names from your dataset
        X_base = pd.DataFrame([{
            'transaction_id': input_values['transaction_id'],
            'customer_id': input_values['customer_id'],
            'merchant_id': input_values['merchant_id'],
            'amount': input_values['amount'],
            'transaction_date': input_values['transaction_date'],
            'card_type': input_values['card_type'],
            'location': input_values['location'],
            'purchase_category': input_values['purchase_category'],
            'customer_age': input_values['customer_age'],
            'fraud_type': input_values['fraud_type']
        }])
        
        with st.spinner("Engineering features and analyzing transaction..."):
            # Engineer features
            X_engineered = engineer_features(X_base, df)
            
            # Show all columns for debugging
            with st.expander("🔧 All Features (Debug)", expanded=False):
                st.write("**Available columns:**")
                st.write(list(X_engineered.columns))
                st.dataframe(X_engineered)
            
            # Make prediction
            prediction = model.predict(X_engineered)[0]
            
            # Get probability if available
            probability = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_engineered)[0]
                    probability = float(probs[1]) if len(probs) > 1 else None
                except:
                    pass
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if int(prediction) == 1:
                st.error("### ⚠️ FRAUD DETECTED")
                if probability is not None:
                    st.metric("Fraud Probability", f"{probability:.1%}")
                st.warning("🚨 This transaction shows signs of fraudulent activity. Please review carefully.")
            else:
                st.success("### ✅ TRANSACTION SAFE")
                if probability is not None:
                    st.metric("Fraud Probability", f"{probability:.1%}")
                st.info("✓ This transaction appears to be legitimate.")
        
        # Show input summary
        with st.expander("📝 Input Summary", expanded=False):
            st.json(input_values)
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        with st.expander("🔧 Error Details", expanded=True):
            st.code(traceback.format_exc())
            st.warning("""
            **Troubleshooting:**
            - Check if dataset has 'is_fraud' column for fraud rate calculations
            - Verify all column names match the training data
            - Ensure date format is correct (MM/DD/YYYY or YYYY-MM-DD)
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛡️ Fraud Detection System | Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>Includes automatic feature engineering</p>
</div>
""", unsafe_allow_html=True)
