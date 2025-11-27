# ==============================================================================
# 0. Setup and Imports
# Run "pip install streamlit pandas numpy scikit-learn imbalanced-learn" in your terminal
# Then run the app using: "streamlit run app.py"
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import io
from collections import Counter
import matplotlib.pyplot as plt

# Scikit-learn and imbalanced-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Set global random seed for reproducibility
np.random.seed(42)

# Column names for the KDD Cup 1999 dataset (41 features + 1 target)
COL_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","attack_type"
]
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# ==============================================================================
# 1. Preprocessing and Feature Engineering Function
# ==============================================================================

@st.cache_data
def preprocess_data(df):
    """Performs KDD-specific preprocessing steps."""
    
    # 1. Remove Duplicates
    df.drop_duplicates(inplace=True)
    
    # 2. Binary Classification: Map all attacks to 1, 'normal' to 0
    # NOTE: KDDTest+ uses 'normal' (no dot)
    df['attack_label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    y = df['attack_label']
    X = df.drop(columns=['attack_type', 'attack_label'])

    # 3. Handle Categorical Features using One-Hot Encoding
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    
    # 4. Explicitly coerce potential numerical columns to a numeric type
    numerical_cols_pre_scaling = [col for col in X.columns if col not in CATEGORICAL_COLS]
    
    for col in numerical_cols_pre_scaling:
        # Force conversion, turning non-numeric values into NaN
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    # Replace any resulting NaN values (if any occurred during coerce)
    X.fillna(0, inplace=True) 

    return X, y

# ==============================================================================
# 2. Model Training and Optimization Function
# ==============================================================================

def train_and_optimize(X, y):
    """Trains the SMOTE/RF model and finds the optimal decision threshold."""
    
    # 1. Split data (stratify ensures correct class ratios in splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 2. Define Model Pipeline
    # StandardScaler must be the first step in the pipeline after SMOTE!
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Note: StandardScaler is applied to ALL features after OHE but before SMOTE/RF,
    # but SMOTE is applied only to the training subset within the pipeline.
    pipeline = Pipeline(steps=[('smote', smote), ('scaler', scaler), ('classifier', model)])
    
    # 3. Train
    pipeline.fit(X_train, y_train)
    
    # 4. Predict probabilities on the held-out TEST set
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 5. Optimize Decision Threshold for F1-Score
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    
    ix = np.argmax(fscore)
    optimal_threshold = thresholds[ix]
    max_f1 = fscore[ix]
    
    # 6. Final Evaluation with Optimal Threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    report = classification_report(y_test, y_pred_optimal, output_dict=True, zero_division=0)
    
    # Store results in session state
    st.session_state.trained_model = pipeline
    st.session_state.optimal_threshold = optimal_threshold
    st.session_state.report = report
    st.session_state.test_data = (y_test, y_prob)
    st.session_state.feature_names = list(X.columns)
    
    st.success("Model Training Complete! Optimized F1-Score: {:.4f}".format(max_f1))
    
    return optimal_threshold, report, precision, recall, max_f1, ix

# ==============================================================================
# 3. Streamlit Interface Functions
# ==============================================================================

def display_results(optimal_threshold, report, precision, recall, max_f1, ix):
    """Displays the classification report and P-R curve."""
    st.header("4. Model Evaluation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimal Threshold")
        st.info(f"Threshold set for maximum F1-score: **{optimal_threshold:.4f}**")
        st.markdown(f"**Max F1-Score Achieved:** **{max_f1:.4f}**")

    with col2:
        st.subheader("Classification Report (Optimal Threshold)")
        report_df = pd.DataFrame(report).transpose()
        report_df.index = ['Normal (0)', 'Attack (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg']
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
        
    st.subheader("Precision-Recall Curve Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    ax.scatter(recall[ix], precision[ix], marker='o', color='red', 
                label=f'Optimal Point (Max F1={max_f1:.2f})')
    ax.set_xlabel('Recall (True Positive Rate)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.set_title('Precision-Recall Curve for Attack Detection')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig) # [Image of Precision-Recall Curve]

def prediction_interface():
    """Allows user to input one sample and get a real-time prediction."""
    st.header("5. Real-time Prediction Interface")
    
    if 'trained_model' not in st.session_state:
        st.warning("Please upload a dataset and train the model first.")
        return

    st.markdown("Enter values for a new connection to predict if it is an attack (1) or normal (0).")
    
    model = st.session_state.trained_model
    optimal_threshold = st.session_state.optimal_threshold
    feature_names = st.session_state.feature_names
    
    # Create input fields (only for original non-categorical numerical columns for simplicity)
    original_features = COL_NAMES[:-1] # All columns except attack_type

    # Use a dictionary to collect user inputs
    user_input = {}
    
    # Layout using two columns
    cols = st.columns(3)
    
    for i, col_name in enumerate(original_features):
        with cols[i % 3]:
            # Use appropriate input widget based on data type
            if col_name in CATEGORICAL_COLS:
                # Use selectbox for categorical features
                if col_name == 'protocol_type':
                    options = ['icmp', 'tcp', 'udp']
                elif col_name == 'flag':
                    options = ['SF', 'S0', 'REJ', 'RSTO', 'SH', 'S1', 'RSTOS0', 'OTH', 'S2', 'RSTR', 'S3']
                else: # service is too high-cardinality for a simple select box, just use text input
                    options = ['http', 'ftp', 'smtp', 'domain_udp', 'private', 'ecr_i', 'other', 'telnet', 'pop_3', 'finger', 'auth', 'urp_i', 'eco_i', 'Z39_50', 'bgp', 'hostnames', 'ctf', 'supdup', 'link', 'imap4', 'daytime', 'uucp_path', 'gopher', 'netstat', 'klogin', 'kshell', 'echo', 'discard', 'systat', 'login', 'exec', 'ntp_tcp', 'shell', 'efs', 'ssh', 'name', 'whois', 'tim_i', 'domain', 'pop_2', 'sunrpc', 'courier', 'remote_job', 'iso_tsap', 'vmtp', 'rdp', 'netbios_ns', 'netbios_dgm', 'netbios_ssn', 'mtp', 'printer', 'pm_dump', 'sntp', 'nnsp', 'irc', 'nntp', 'bftp', 'ldap', 'netmon', 'urh_i', 'tftp_u', 'http_443', 'aol', 'hftp', 'harvest']
                
                # Check if it's a high-cardinality field, use text input as fallback for 'service'
                if col_name == 'service':
                    user_input[col_name] = st.text_input(f"Enter {col_name}", value="http")
                else:
                    user_input[col_name] = st.selectbox(f"Select {col_name}", options=options)

            else:
                # Use number input for numerical features
                default_value = 0.0 # Placeholder default
                if col_name in ['src_bytes', 'dst_bytes']:
                    default_value = 100 
                user_input[col_name] = st.number_input(f"Enter {col_name}", value=default_value)

    if st.button("Predict Attack Status"):
        # 1. Convert user input to a DataFrame row
        input_df = pd.DataFrame([user_input])
        
        # 2. Re-create the feature set exactly as it was during training
        # a. One-Hot Encode (critical step for categorical features)
        input_df_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=True)
        
        # b. Reindex and fill missing OHE columns with 0
        missing_cols = set(feature_names) - set(input_df_encoded.columns)
        for col in missing_cols:
            input_df_encoded[col] = 0
            
        # Ensure the order is exactly the same as training
        final_input = input_df_encoded[feature_names].values

        # 3. Predict Probability
        try:
            # The pipeline handles scaling internally
            prob = model.predict_proba(final_input)[:, 1][0]
            
            # 4. Apply Optimal Threshold
            prediction = 1 if prob >= optimal_threshold else 0

            st.subheader("Prediction Result:")
            
            # Display results
            if prediction == 1:
                st.error(f"ATTACK DETECTED (Probability: {prob:.4f} vs. Threshold: {optimal_threshold:.4f})")
                st.markdown("⚠️ **Action Recommended:** This connection is highly likely to be malicious.")
            else:
                st.success(f"NORMAL CONNECTION (Probability: {prob:.4f} vs. Threshold: {optimal_threshold:.4f})")
                st.markdown("✅ **Status:** Connection classified as safe.")
                
        except Exception as e:
            st.error(f"Prediction failed. Ensure all input fields are correctly filled. Error: {e}")

# ==============================================================================
# 4. Main Streamlit App
# ==============================================================================
def main():
    st.set_page_config(layout="wide", page_title="KDD Intrusion Detection Optimizer")
    st.title("🛡️ KDD Cup '99 Intrusion Detection Model Optimizer")
    st.markdown("---")
    
    # Initialize session state variables
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None

    # 1. File Upload
    st.header("1. Upload KDD Dataset")
    uploaded_file = st.file_uploader("Upload your KDDTest+.csv file here", type="csv")

    if uploaded_file is not None and st.session_state.df is None:
        try:
            # Read the file from the buffer
            df_raw = pd.read_csv(uploaded_file, header=None, names=COL_NAMES, low_memory=False)
            st.session_state.df_raw = df_raw
            st.session_state.X, st.session_state.y = preprocess_data(df_raw.copy())
            st.success("Dataset loaded and preprocessed successfully!")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # 2. Data Preview
    if st.session_state.df_raw is not None:
        st.header("2. Dataset Preview (First 5 Rows)")
        st.dataframe(st.session_state.df_raw.head(), use_container_width=True)
        
        X_df = pd.DataFrame(st.session_state.X, columns=st.session_state.feature_names)
        col_dist, col_size = st.columns(2)
        with col_dist:
            st.subheader("Class Distribution (Before SMOTE)")
            st.bar_chart(st.session_state.y.value_counts(), use_container_width=True)
        with col_size:
            st.subheader("Data Summary")
            st.write(f"Total Records: {len(st.session_state.df_raw)}")
            st.write(f"Features (After OHE): {st.session_state.X.shape[1]}")
            st.write(f"Class 1 (Attack) Count: {st.session_state.y.sum()}")
            st.write(f"Class 0 (Normal) Count: {len(st.session_state.y) - st.session_state.y.sum()}")
            st.info("The preprocessing steps included dropping duplicates and converting all attacks to a binary '1'.")

    # 3. Model Training
    st.header("3. Train Model and Optimize Threshold")
    if st.session_state.df_raw is not None:
        if st.button("Start Training (SMOTE + Random Forest)", type="primary"):
            with st.spinner('Training model, applying SMOTE, and optimizing F1-Score...'):
                X = st.session_state.X
                y = st.session_state.y
                optimal_threshold, report, precision, recall, max_f1, ix = train_and_optimize(X, y)
                display_results(optimal_threshold, report, precision, recall, max_f1, ix)
        
        # Display results if already trained
        if 'trained_model' in st.session_state:
            optimal_threshold = st.session_state.optimal_threshold
            report = st.session_state.report
            y_test, y_prob = st.session_state.test_data
            
            # Recalculate P-R curve for display consistency
            precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
            fscore = (2 * precision * recall) / (precision + recall)
            fscore[np.isnan(fscore)] = 0
            ix = np.argmax(fscore)
            max_f1 = fscore[ix]
            
            display_results(optimal_threshold, report, precision, recall, max_f1, ix)

    st.markdown("---")

    # 5. Prediction Interface (runs only if model is trained)
    if 'trained_model' in st.session_state:
        prediction_interface()

if __name__ == '__main__':
    main()