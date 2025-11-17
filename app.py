# Add these enhancements to your existing app.py

# 1. ADD THIS AFTER THE PREDICTION RESULTS SECTION (around line 350)
# Add visualization of prediction confidence

        # Enhanced Results Display with Gauge Chart
        if probability is not None:
            import plotly.graph_objects as go
            
            # Create gauge chart for fraud probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Risk Score", 'font': {'size': 24}},
                delta = {'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "red" if probability > 0.5 else "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#90EE90'},
                        {'range': [30, 70], 'color': '#FFD700'},
                        {'range': [70, 100], 'color': '#FF6B6B'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)

# 2. ADD TRANSACTION HISTORY TRACKING
# Add this before the prediction form (around line 235)

# Initialize session state for transaction history
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

# Add this inside the prediction success block (around line 355)
        # Save to history
        st.session_state.transaction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_id': input_values['transaction_id'],
            'amount': input_values['amount'],
            'location': input_values['location'],
            'prediction': 'FRAUD' if int(prediction) == 1 else 'SAFE',
            'probability': f"{probability:.2%}" if probability else "N/A"
        })

# Add this in the sidebar (around line 210)
    if st.session_state.transaction_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Predictions")
        history_df = pd.DataFrame(st.session_state.transaction_history[-5:])  # Last 5
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.transaction_history = []
            st.rerun()

# 3. ADD BATCH PREDICTION FEATURE
# Add this after the single prediction form (around line 350)

st.markdown("---")
st.header("📊 Batch Prediction")

with st.expander("Upload CSV for Batch Processing"):
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} transactions")
        st.dataframe(batch_df.head())
        
        if st.button("🔍 Run Batch Prediction"):
            with st.spinner("Processing batch predictions..."):
                try:
                    # Engineer features for batch
                    X_batch = engineer_features(batch_df, df)
                    
                    # Make predictions
                    predictions = model.predict(X_batch)
                    probabilities = model.predict_proba(X_batch)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    # Add results to dataframe
                    batch_df['prediction'] = ['FRAUD' if p == 1 else 'SAFE' for p in predictions]
                    if probabilities is not None:
                        batch_df['fraud_probability'] = probabilities
                    
                    st.success(f"✅ Processed {len(batch_df)} transactions")
                    
                    # Show results
                    fraud_count = sum(predictions)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", len(batch_df))
                    with col2:
                        st.metric("Fraud Detected", fraud_count)
                    with col3:
                        st.metric("Fraud Rate", f"{fraud_count/len(batch_df):.1%}")
                    
                    # Display results
                    st.dataframe(batch_df)
                    
                    # Download results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "fraud_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")

# 4. ADD MODEL PERFORMANCE METRICS
# Add this in the sidebar after model is loaded (around line 220)

    if model is not None and df is not None:
        st.markdown("---")
        st.markdown("### 📈 Model Info")
        
        with st.expander("Model Details"):
            st.write(f"**Model Type:** {type(model).__name__}")
            
            if hasattr(model, 'n_estimators'):
                st.write(f"**Trees:** {model.n_estimators}")
            
            if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                st.write("**Top 5 Important Features:**")
                importances = pd.DataFrame({
                    'feature': model.feature_names_in_,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(5)
                
                for idx, row in importances.iterrows():
                    st.text(f"• {row['feature']}: {row['importance']:.4f}")

# 5. ADD EXPORT FUNCTIONALITY
# Add this after the prediction results (around line 365)

        # Export Report
        st.markdown("---")
        if st.button("📄 Generate PDF Report"):
            st.info("PDF generation would require additional libraries like reportlab or fpdf")
            st.code("""
# To enable PDF reports, install:
pip install reportlab

# Then use reportlab to generate PDF with prediction details
            """)

# 6. ADD REAL-TIME STATISTICS
# Add this at the top of the main area (around line 225)

if df is not None:
    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        if 'is_fraud' in df.columns:
            fraud_rate = df['is_fraud'].mean()
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")
    
    with col3:
        if 'amount' in df.columns:
            avg_amount = df['amount'].mean()
            st.metric("Avg Amount", f"${avg_amount:,.2f}")
    
    with col4:
        if 'location' in df.columns:
            unique_locations = df['location'].nunique()
            st.metric("Locations", unique_locations)
    
    st.markdown("---")
