import os
import tempfile
import traceback
from io import BytesIO
from pathlib import Path
from datetime import datetime
import importlib
import re

import pandas as pd
import streamlit as st
import joblib
import pickle

# Optional import for downloading model by URL
try:
    import requests
except Exception:
    requests = None

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
    """Get list of files in current directory (string names)."""
    try:
        cwd = Path.cwd()
        return sorted([f.name for f in cwd.iterdir() if f.is_file()])
    except Exception as e:
        st.error(f"Failed to list directory: {e}")
        return []

def find_model_file():
    """
    Search current dir and subfolders for matching model files.
    Returns Path or None.
    """
    cwd = Path.cwd()
    # Prefer exact basename + ext in cwd
    for ext in MODEL_EXT_CANDIDATES:
        candidate = cwd / (MODEL_BASENAME + ext)
        if candidate.exists():
            return candidate

    # Search recursively
    try:
        for p in cwd.rglob(f"{MODEL_BASENAME}*"):
            if p.is_file():
                return p
    except Exception:
        pass

    # fallback: any file with basename substring
    for p in cwd.iterdir():
        if p.is_file() and MODEL_BASENAME.lower() in p.name.lower():
            return p
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

def try_inject_stub_class(module_name: str, class_name: str) -> bool:
    """
    Try to import module_name and add a lightweight stub class/class alias
    with the name class_name so pickle can find it during unpickling.
    Returns True if injection succeeded.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False

    if hasattr(module, class_name):
        return True

    # create a minimal stub class
    try:
        # A very small class that accepts any args in __init__
        def _init(self, *args, **kwargs):
            pass

        Stub = type(class_name, (), {"__init__": _init})
        setattr(module, class_name, Stub)
        return True
    except Exception:
        return False

def analyze_unpickle_error_message(msg: str):
    """
    Try to extract missing global/class names from the pickle/unpickle error messages.
    Returns a list of (module_name, class_name) guesses to attempt injection.
    """
    results = []
    # common pattern from traceback: "Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ..."
    m = re.search(r"Can't get attribute '([^']+)' on <module '([^']+)'", msg)
    if m:
        cls = m.group(1)
        mod = m.group(2)
        results.append((mod, cls))
        return results

    # other pickle AttributeError formats, fallback: look for names starting with underscore from sklearn
    names = re.findall(r"_\w{3,}", msg)
    for name in set(names):
        # if it's sklearn related, try injecting into common sklearn modules
        candidates = [
            "sklearn.compose._column_transformer",
            "sklearn.compose._data",
            "sklearn.pipeline",
            "sklearn.preprocessing"
        ]
        for c in candidates:
            results.append((c, name))
    return results

@st.cache_resource
def load_model_from_path(path: Path):
    """
    Load model from a Path using joblib or pickle.
    Attempts to monkeypatch missing classes if unpickling fails with AttributeError.
    """
    # Try joblib.load first (fast and common)
    last_exc = None
    try:
        model = joblib.load(path)
        return model, "joblib"
    except Exception as e:
        last_exc = e
        # Fall through to more advanced handling below

    # Attempt to detect missing-class message and inject stubs
    msg = "".join(traceback.format_exception_only(type(last_exc), last_exc))
    candidates = analyze_unpickle_error_message(msg)

    injected_any = False
    for mod, cls in candidates:
        injected = try_inject_stub_class(mod, cls)
        injected_any = injected_any or injected

    # If we injected stubs, try loading again with joblib then pickle
    if injected_any:
        try:
            model = joblib.load(path)
            return model, f"joblib (loaded after injecting stubs: {', '.join([c for _, c in candidates])})"
        except Exception as e2:
            last_exc = e2

    # Last attempt: try pickle directly (some joblib files are pickles)
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, "pickle"
    except Exception as e3:
        # If we get here, everything failed. Provide helpful guidance.
        full_msg = "".join(traceback.format_exception(type(e3), e3, e3.__traceback__))
        hint = (
            "Model loading failed. Common cause: scikit-learn version mismatch between training and serving "
            "environments (private classes moved/renamed). Recommended fixes:\n\n"
            "1) Run the app with the same scikit-learn version used to save the model. Example (replace X.Y.Z with the version used to train):\n"
            "   pip install scikit-learn==X.Y.Z\n\n"
            "2) If you don't know the version, try common versions like 1.2.2 or 1.1.3.\n\n"
            "3) As a temporary fallback, the app attempted to inject stub classes for: "
            + ", ".join([f"{m}.{c}" for m, c in candidates]) + ".\n\n"
            "Full unpickle error:\n" + full_msg
        )
        raise RuntimeError(hint) from e3

def load_model_from_bytes(bytes_data: bytes, filename_hint: str = "uploaded_model"):
    """
    Load model from raw bytes by writing to a temporary file then loading.
    Returns (model, method) or raises.
    """
    suffix = Path(filename_hint).suffix if "." in filename_hint else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(bytes_data)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        return load_model_from_path(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

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
    Create engineered features that the model expects.
    This should match what model training used.
    """
    data = input_data.copy()

    # Extract datetime features if transaction_date exists
    if 'transaction_date' in data.columns:
        data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')
        data['tx_hour'] = data['transaction_date'].dt.hour.fillna(12).astype(int)
        data['tx_weekday'] = data['transaction_date'].dt.dayofweek.fillna(2).astype(int)
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
        if 'purchase_category' in data.columns and 'purchase_category' in df.columns:
            fraud_rates = calculate_fraud_rates(df, 'purchase_category')
            data['purchase_category_fraud_rate'] = data['purchase_category'].map(fraud_rates).fillna(0.5)
        else:
            data['purchase_category_fraud_rate'] = 0.5

        if 'location' in data.columns and 'location' in df.columns:
            fraud_rates = calculate_fraud_rates(df, 'location')
            data['location_fraud_rate'] = data['location'].map(fraud_rates).fillna(0.5)
        else:
            data['location_fraud_rate'] = 0.5
    else:
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
    1. Ensure model file is in the directory (or upload it below).
    2. Enter transaction details.
    3. Click **Predict** to check for fraud.
    """)

    # --- Model upload / URL fallback ---
    with st.expander("📦 Model (optional upload)", expanded=True):
        st.write("If your deployed environment doesn't already include the model file, upload it here or provide a URL.")
        uploaded_model = st.file_uploader(
            label="Upload model (.pkl / .joblib)",
            type=["pkl", "joblib"],
            accept_multiple_files=False
        )
        model_url = st.text_input("Or enter model URL (optional)", value="")

        if model_url and requests is None:
            st.warning("`requests` not available in environment. To enable URL download, add `requests` to requirements.txt.")

# --- Load model logic: prefer uploaded file, then URL, then repo/workdir ---
model = None
model_method = None
model_path = None

if uploaded_model is not None:
    try:
        st.info("Loading model from uploaded file...")
        bytes_data = uploaded_model.read()
        model, model_method = load_model_from_bytes(bytes_data, filename_hint=uploaded_model.name)
        st.success(f"✅ Model loaded from uploaded file: {uploaded_model.name}")
    except Exception as e:
        st.error("Failed to load uploaded model.")
        with st.expander("Upload error details", expanded=True):
            st.code(traceback.format_exc())
elif model_url:
    if requests is None:
        st.error("Cannot download model because `requests` is not installed in this environment.")
    else:
        try:
            st.info("Downloading model from URL...")
            resp = requests.get(model_url, timeout=30)
            resp.raise_for_status()
            model, model_method = load_model_from_bytes(resp.content, filename_hint=model_url.split("/")[-1])
            st.success("✅ Model downloaded and loaded from URL")
        except Exception as e:
            st.error("Failed to download/load model from URL.")
            with st.expander("Download error", expanded=True):
                st.code(traceback.format_exc())
else:
    model_path = find_model_file()
    if model_path is None:
        st.error(f"❌ Model not found. Please place `{MODEL_BASENAME}.pkl` or `.joblib` in: {Path.cwd()}")
    else:
        try:
            st.success(f"✅ Model found: `{model_path.name}`")
            model, model_method = load_model_from_path(model_path)
            st.info(f"📦 Loaded using: **{model_method}**")
        except Exception as e:
            # load_model_from_path raises helpful RuntimeError with guidance
            st.error("Failed to load model from path: see details below.")
            with st.expander("Model load error", expanded=True):
                st.text(str(e))

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
    st.stop()

# Utility to get unique values from dataset for dropdowns
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
            max_value=1000000.0,
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

# ======= REPLACE existing `if submitted:` block with this =======
FRAUD_THRESHOLD = 0.5  # change to 0.3 or 0.4 to be more sensitive

if submitted:
    try:
        # Build input dataframe
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

        with st.spinner("Engineering features and preparing input..."):
            X_engineered = engineer_features(X_base, df)

            # Debug: show engineered features
            with st.expander("🔧 All Features (Debug)", expanded=True):
                st.write("**Available columns in engineered input:**")
                st.write(list(X_engineered.columns))
                st.dataframe(X_engineered)

            # Diagnostics: model expectations
            model_expected = None
            model_feature_source = None
            try:
                # 1) Common sklearn attribute
                if hasattr(model, "feature_names_in_"):
                    model_expected = list(model.feature_names_in_)
                    model_feature_source = "model.feature_names_in_"
                else:
                    # 2) If model is a Pipeline, attempt to find final estimator's feature_names_in_
                    if hasattr(model, "named_steps"):
                        for name, step in model.named_steps.items():
                            if hasattr(step, "feature_names_in_"):
                                model_expected = list(step.feature_names_in_)
                                model_feature_source = f"pipeline.named_steps['{name}'].feature_names_in_"
                                break
                # 3) If we still don't have expected feature names, try attribute 'columns' for DataFrame transformers
            except Exception:
                model_expected = None
                model_feature_source = None

            # Show model diagnostics in expander
            with st.expander("🧾 Model Diagnostics", expanded=True):
                try:
                    st.write("Model type:", type(model))
                    if hasattr(model, "classes_"):
                        st.write("Model classes_:", getattr(model, "classes_"))
                    if model_expected:
                        st.write(f"Model expected features (source: {model_feature_source}):")
                        st.write(model_expected)
                        # Attempt to align columns
                        missing = [c for c in model_expected if c not in X_engineered.columns]
                        extra = [c for c in X_engineered.columns if c not in model_expected]
                        st.write("Missing from input (will be filled with 0):", missing)
                        st.write("Extra columns in input (kept):", extra)
                    else:
                        st.write("Model does not expose `feature_names_in_` — cannot auto-align.")
                except Exception as ex_diag:
                    st.write("Diagnostics error:", str(ex_diag))

            # If we have expected feature names, reindex X_engineered to match (fill missing with 0)
            if model_expected:
                # preserve original X_engineered copy for display
                X_for_model = X_engineered.reindex(columns=model_expected, fill_value=0)
            else:
                # no guidance — send the engineered table as-is
                X_for_model = X_engineered.copy()

            # show the final DataFrame passed to model (for debugging)
            with st.expander("🧪 Final input passed to model", expanded=False):
                st.write("Columns passed to model (in order):")
                st.write(list(X_for_model.columns))
                st.dataframe(X_for_model)

            # Attempt prediction: prefer predict, fall back to predict_proba or decision_function
            label = None
            prob = None
            pred_exception = None

            # 1) Try model.predict
            if hasattr(model, "predict"):
                try:
                    pred = model.predict(X_for_model)
                    label = int(pred[0])
                except Exception as e:
                    pred_exception = e
                    label = None

            # 2) If predict failed or not available, try predict_proba
            if label is None and hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_for_model)[0]
                    # assume index 1 corresponds to fraud if classes_ = [0,1]
                    if hasattr(model, "classes_"):
                        classes = list(getattr(model, "classes_"))
                        # try to find index of class '1' or 'fraud' - fallback to index 1
                        if 1 in classes:
                            idx = classes.index(1)
                        elif "fraud" in classes:
                            idx = classes.index("fraud")
                        else:
                            idx = 1 if len(probs) > 1 else 0
                    else:
                        idx = 1 if len(probs) > 1 else 0
                    prob = float(probs[idx])
                    label = 1 if prob >= FRAUD_THRESHOLD else 0
                except Exception as e:
                    pred_exception = e
                    label = None

            # 3) If still not determined, try decision_function
            if label is None and hasattr(model, "decision_function"):
                try:
                    score = model.decision_function(X_for_model)[0]
                    # convert score to label with zero threshold (adjust if you want)
                    label = 1 if score >= 0 else 0
                except Exception as e:
                    pred_exception = e
                    label = None

            if label is None:
                raise RuntimeError(f"Model prediction failed. Last error: {pred_exception}")

        # Display results (binary decision only if you prefer)
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Show probability only for debugging; remove later if you want pure label
            if prob is not None:
                st.write(f"Fraud probability (used threshold {FRAUD_THRESHOLD}): **{prob:.2%}**")
            if int(label) == 1:
                st.error("### ⚠️ FRAUD DETECTED")
                st.warning("🚨 This transaction shows signs of fraudulent activity. Please review carefully.")
            else:
                st.success("### ✅ TRANSACTION SAFE")
                st.info("✓ This transaction appears to be legitimate.")

        # Show input summary
        with st.expander("📝 Input Summary", expanded=False):
            st.json(input_values)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        with st.expander("🔧 Error Details", expanded=True):
            st.code(traceback.format_exc())
            st.warning("""
            Troubleshooting tips:
            - If model expects preprocessed features (scaled/encoded), you must apply the *same* preprocessing before prediction.
            - If the model exposes `feature_names_in_`, ensure the engineered features match those names and dtypes.
            - If `predict_proba` shows low probability even for extreme inputs, try lowering FRAUD_THRESHOLD (e.g., 0.3).
            - If your model uses a sklearn Pipeline with internal ColumnTransformer, prefer saving/loading the entire pipeline so preprocessing is included.
            """)
# ======= end replacement block =======


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛡️ Fraud Detection System | Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>Includes automatic feature engineering and model upload fallback</p>
</div>
""", unsafe_allow_html=True)
