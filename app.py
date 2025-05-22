import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from io import StringIO
import joblib
# Set page configuration
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🔍",
    layout="wide"
)

def load_model(model_path):
    """Load the XGBoost model from file with version compatibility."""
    try:
        # Try loading with the standard method first
        try:
            model = joblib.load(model_path)
            return model
        except (TypeError, AttributeError) as e:
            # If there's a compatibility issue with use_label_encoder
            if 'use_label_encoder' in str(e):
                import xgboost as xgb
                # Load the model with a custom unpickler
                model = joblib.load(model_path, custom_objects={'XGBClassifier': xgb.XGBClassifier})
                return model
            raise e
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("This might be due to version incompatibility. Please ensure you have the correct versions of xgboost and scikit-learn installed.")
        return None


def main():
    st.title("🔍 Fraud Detection System")
    st.write("Upload a CSV file containing transaction data for fraud detection.")
    
    # Load the model
    model_path = "ARTIFACTS/xgboost_model.joblib"
    model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load the model. Please check the model file.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display the uploaded data
            st.subheader("Uploaded Data")
            st.write(f"Number of records: {len(df)}")
            st.dataframe(df.head())
            
            # Preprocess the data
            processed_df = df
            
            # Make predictions
            if st.button("Predict Fraud"):
                with st.spinner("Making predictions..."):
                    try:
                        # Get predictions (probabilities)
                        probabilities = model.predict_proba(processed_df)[:, 1]
                        
                        # Add predictions to the dataframe
                        df['Fraud_Probability'] = probabilities
                        df['Prediction'] = (probabilities >= 0.5).astype(int)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Summary statistics
                        fraud_count = df['Prediction'].sum()
                        total = len(df)
                        st.metric("Potential Fraud Cases", f"{fraud_count} out of {total} ({fraud_count/total*100:.2f}%)")
                        
                        # Show only records with confidence > 50%
                        high_confidence_df = df[df['Fraud_Probability'] > 0.5]
                        
                        # Show detailed results for high confidence predictions
                        if len(high_confidence_df) > 0:
                            st.subheader("Confidence Predictions")
                            st.dataframe(high_confidence_df[['Prediction', 'Fraud_Probability']].sort_values('Fraud_Probability', ascending=False))
                        else:
                            st.info("No records with confidence > 50% were found.")
                        
                        # Download results
                        # Filter dataframe for high confidence predictions before downloading
                        high_confidence_df = df[df['Fraud_Probability'] > 0.5]
                        csv = high_confidence_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name='fraud_predictions.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
                        st.error("Please ensure your data matches the expected format.")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.error("Please upload a valid CSV file.")

if __name__ == "__main__":
    main()
