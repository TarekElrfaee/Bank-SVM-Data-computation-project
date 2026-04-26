import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Bank Marketing Predictor Pro", page_icon="🏦", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure this matches your exported file name
    return joblib.load('bank_svm_pipeline.pkl')

model = load_model()

# --- HELPER FUNCTIONS ---
def create_gauge_chart(probability):
    """Creates a visual gauge for the prediction probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40, 'color': "white"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Subscription Probability", 'font': {'size': 20, 'color': "white"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00cc96" if probability > 0.5 else "#ef553b"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 85, 59, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(0, 204, 150, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --- UI LAYOUT ---
st.title("🏦 Bank Term Deposit Prediction System")
st.markdown("Assess client subscription likelihood using a calibrated Support Vector Machine.")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["🎯 Single Client Scoring", "📂 Batch Processing"])

# ==========================================
# TAB 1: SINGLE CLIENT SCORING
# ==========================================
with tab1:
    with st.form("prediction_form"):
        # Use expanders to keep the UI clean
        with st.expander("👤 Client Demographics", expanded=True):
            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", 17, 100, 38)
            job = col2.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
            marital = col3.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
            education = col1.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
            
        with st.expander("💳 Financial Status", expanded=True):
            col4, col5, col6 = st.columns(3)
            default = col4.selectbox("Credit in Default?", ['no', 'yes', 'unknown'])
            housing = col5.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
            loan = col6.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])

        with st.expander("📞 Campaign Interactions", expanded=False):
            col7, col8, col9 = st.columns(3)
            contact = col7.selectbox("Communication Type", ['cellular', 'telephone'])
            month = col8.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            day_of_week = col9.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
            duration = col7.number_input("Last Call Duration (s)", 0, 5000, 180)
            campaign = col8.number_input("Contacts this Campaign", 1, 50, 2)
            pdays = col9.number_input("Days since Previous Contact (999=never)", 0, 999, 999)
            previous = col7.number_input("Contacts before Campaign", 0, 10, 0)
            poutcome = col8.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])

        with st.expander("🌍 Macro-Economic Indicators", expanded=False):
            col10, col11, col12 = st.columns(3)
            emp_var_rate = col10.number_input("Employment Variation Rate", value=1.1)
            cons_price_idx = col11.number_input("Consumer Price Index", value=93.994)
            cons_conf_idx = col12.number_input("Consumer Confidence Index", value=-36.4)
            euribor3m = col10.number_input("Euribor 3-month Rate", value=4.857)
            nr_employed = col11.number_input("Number of Employees", value=5191.0)

        submit_single = st.form_submit_button(label="Analyze Client", use_container_width=True)

    if submit_single:
        # Construct exact DataFrame for the pipeline
        input_data = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan, 'contact': contact, 
            'month': month, 'day_of_week': day_of_week, 'duration': duration, 
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 
            'poutcome': poutcome, 'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 
            'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Yes)

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.subheader("Final Verdict")
            if pred == 1:
                st.success("### ✅ HIGH POTENTIAL\nThis client exhibits patterns consistent with subscribing.")
            else:
                st.error("### ❌ LOW POTENTIAL\nThis client is unlikely to subscribe at this time.")
                
        with res_col2:
            st.plotly_chart(create_gauge_chart(prob), use_container_width=True)

# ==========================================
# TAB 2: BATCH PROCESSING
# ==========================================
with tab2:
    st.subheader("Bulk Scoring")
    st.markdown("Upload a CSV containing multiple client records. Ensure the file contains the same 20 columns used during model training.")
    
    uploaded_file = st.file_uploader("Upload Client Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_data)} records. Generating predictions...")
            
            # Predict
            batch_preds = model.predict(batch_data)
            batch_probs = model.predict_proba(batch_data)[:, 1]
            
            # Append results
            results_df = batch_data.copy()
            results_df['Predicted_Subscription'] = ['Yes' if p == 1 else 'No' for p in batch_preds]
            results_df['Confidence_Score'] = [f"{p:.2%}" for p in batch_probs]
            
            # Display interactive dataframe
            st.dataframe(
                results_df.style.applymap(
                    lambda x: 'background-color: rgba(0, 204, 150, 0.2)' if x == 'Yes' else 'background-color: rgba(239, 85, 59, 0.2)', 
                    subset=['Predicted_Subscription']
                ), 
                use_container_width=True
            )
            
            # Allow downloading the scored data
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Dataset",
                data=csv,
                file_name='scored_bank_clients.csv',
                mime='text/csv',
                type="primary"
            )
        except Exception as e:
            st.error(f"Error processing file: Ensure column names match the training set perfectly. Details: {e}")