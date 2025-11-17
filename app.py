import streamlit as st
import pandas as pd
import pickle
import os

# Page config
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️")

# Title
st.title("🛡️ Fraud Detection System")

# System Information Sidebar
with st.sidebar:
    st.header("📊 System Information")
    
    with st.expander("🔍 Debug Info"):
        st.write("Model file exists:", os.path.exists('fraud_detector_final.pkl'))
        st.write("Dataset file exists:", os.path.exists('FRAUD_DETECTION.csv'))

# Instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.write("1. Ensure model file is in the directory")
    st.write("2. Enter transaction details")
    st.write("3. Click **Predict** to check for fraud")

# Load model with proper error handling
@st.cache_resource
def load_model():
    try:
        # Try loading with encoding parameter
        with open('fraud_detector_final.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        return model, None
    except Exception as e:
        return None, str(e)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('FRAUD_DETECTION.csv')
        return df, None
    except Exception as e:
        return None, str(e)

# Status messages
model, model_error = load_model()
dataset, data_error = load_data()

if model is not None:
    st.success("✅ Model found: fraud_detector_final.pkl")
else:
    st.error(f"❌ Failed to load model: {model_error}")
    st.info("💡 **Solution**: Retrain and save the model using the same Python version")

if dataset is not None:
    st.success("✅ Dataset found: FRAUD_DETECTION.csv")
else:
    st.error(f"❌ Failed to load dataset: {data_error}")

# Dataset Preview
if dataset is not None:
    with st.expander("📊 Dataset Preview"):
        st.dataframe(dataset.head(10))
        st.write(f"Total records: {len(dataset)}")
        st.write(f"Features: {list(dataset.columns)}")

# Prediction Section
st.header("🔮 Make Prediction")

if model is None:
    st.error("⚠️ Cannot proceed without a model.")
    st.markdown("""
    ### How to fix this error:
    
    **Option 1: Retrain the model** (Recommended)
    ```python
    # Run this code to retrain and save the model properly
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    # Load your data
    df = pd.read_csv('FRAUD_DETECTION.csv')
    
    # Prepare features and target
    X = df.drop('isFraud', axis=1)  # Adjust column name as needed
    y = df['isFraud']
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Save model with protocol 4 for better compatibility
    with open('fraud_detector_final.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    ```
    
    **Option 2: Use joblib instead of pickle**
    ```python
    import joblib
    # Save: joblib.dump(model, 'fraud_detector_final.pkl')
    # Load: model = joblib.load('fraud_detector_final.pkl')
    ```
    """)
else:
    # Get feature names from the model or dataset
    if dataset is not None:
        feature_cols = [col for col in dataset.columns if col != 'isFraud']
    else:
        # Default features - adjust based on your actual features
        feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest']
    
    st.write("Enter transaction details:")
    
    # Create input fields dynamically
    input_data = {}
    cols = st.columns(2)
    for idx, feature in enumerate(feature_cols[:10]):  # Limit to first 10 features for display
        with cols[idx % 2]:
            input_data[feature] = st.number_input(
                f"{feature}", 
                value=0.0, 
                format="%.2f",
                key=feature
            )
    
    if st.button("🔍 Predict", type="primary"):
        try:
            # Create DataFrame with input
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            if prediction == 1:
                st.error("🚨 **FRAUD DETECTED**")
                st.write(f"Fraud Probability: {probability[1]:.2%}")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")
                st.write(f"Legitimate Probability: {probability[0]:.2%}")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
