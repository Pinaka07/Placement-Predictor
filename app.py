import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Set page configuration
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .success {
        color: #28a745;
        font-weight: bold;
    }
    .warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and transformers
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("placement_prediction.pkl", 'rb'))
        transformer = pickle.load(open("transformer.pkl", 'rb'))
        scaler = pickle.load(open("scaler.pkl", 'rb'))
        return model, transformer, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

# Load and process data
@st.cache_data
def load_data():
    try:
        # Load the data - the CSV is already properly formatted
        plc_data = pd.read_csv("placementdata .csv")
        
        # Convert target variable to numerical
        plc_data['PlacementStatus'] = plc_data['PlacementStatus'].map({'NotPlaced': 0, 'Placed': 1})
        
        # Convert numeric columns
        numeric_columns = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                          'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
        for col in numeric_columns:
            plc_data[col] = pd.to_numeric(plc_data[col], errors='coerce')
        
        return plc_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def predict_placement(input_data, model, transformer, scaler):
    """Make prediction using the loaded model"""
    try:
        # Transform input data
        input_transformed = transformer.transform(input_data)
        input_scaled = scaler.transform(input_transformed)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Convert to categorical result
        if prediction >= 0.5:
            status = "Placed"
            confidence = prediction
        else:
            status = "Not Placed"
            confidence = 1 - prediction
        
        return status, prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Placement Predictor</h1>', unsafe_allow_html=True)
    
    # Load model and data
    model, transformer, scaler = load_model()
    data = load_data()
    
    if model is None or data is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Analysis", "🎯 Predict Placement", "📈 Model Performance"]
    )
    
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📊 Data Analysis":
        show_data_analysis(data)
    elif page == "🎯 Predict Placement":
        show_prediction_page(model, transformer, scaler)
    elif page == "📈 Model Performance":
        show_model_performance(data, model, transformer, scaler)

def show_home_page(data):
    """Display home page with overview"""
    st.markdown("## Welcome to Placement Predictor!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", len(data))
    
    with col2:
        placed_count = data['PlacementStatus'].sum()
        st.metric("Students Placed", placed_count)
    
    with col3:
        placement_rate = (placed_count / len(data)) * 100
        st.metric("Placement Rate", f"{placement_rate:.1f}%")
    
    st.markdown("---")
    
    # Key insights
    st.markdown("### Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_cgpa = data['CGPA'].mean()
        st.metric("Average CGPA", f"{avg_cgpa:.2f}")
        
        avg_projects = data['Projects'].mean()
        st.metric("Average Projects", f"{avg_projects:.1f}")
    
    with col2:
        avg_internships = data['Internships'].mean()
        st.metric("Average Internships", f"{avg_internships:.1f}")
        
        avg_aptitude = data['AptitudeTestScore'].mean()
        st.metric("Average Aptitude Score", f"{avg_aptitude:.1f}")

def show_data_analysis(data):
    """Display data analysis and visualizations"""
    st.markdown("## 📊 Data Analysis")
    
    # Data overview
    st.markdown("### Dataset Overview")
    st.dataframe(data.head(), use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CGPA Distribution")
        fig = px.histogram(data, x='CGPA', nbins=20, color_discrete_sequence=['#1f77b4'])
        fig.update_layout(title="Distribution of CGPA Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Placement Status")
        placement_counts = data['PlacementStatus'].value_counts()
        fig = px.pie(values=placement_counts.values, names=['Not Placed', 'Placed'], 
                     color_discrete_sequence=['#ff7f0e', '#2ca02c'])
        fig.update_layout(title="Placement Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### Feature Correlation with Placement")
    numeric_cols = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                   'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
    
    correlations = data[numeric_cols + ['PlacementStatus']].corr()['PlacementStatus'].sort_values(ascending=False)
    
    fig = px.bar(x=correlations.index[:-1], y=correlations.values[:-1], 
                 color=correlations.values[:-1], color_continuous_scale='RdBu')
    fig.update_layout(title="Feature Correlation with Placement Status", 
                     xaxis_title="Features", yaxis_title="Correlation")
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model, transformer, scaler):
    """Display prediction interface"""
    st.markdown("## 🎯 Predict Placement")
    
    st.markdown("Enter student details to predict placement status:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cgpa = st.slider("CGPA", 6.0, 10.0, 8.0, 0.1)
            internships = st.slider("Number of Internships", 0, 5, 1)
            projects = st.slider("Number of Projects", 0, 10, 2)
            workshops = st.slider("Workshops/Certifications", 0, 10, 2)
            aptitude_score = st.slider("Aptitude Test Score", 0, 100, 75)
        
        with col2:
            soft_skills = st.slider("Soft Skills Rating", 1.0, 5.0, 3.0, 0.1)
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
            placement_training = st.selectbox("Placement Training", ["Yes", "No"])
            ssc_marks = st.slider("SSC Marks", 0, 100, 80)
            hsc_marks = st.slider("HSC Marks", 0, 100, 80)
        
        submitted = st.form_submit_button("Predict Placement")
    
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            "StudentID": [9999],
            "CGPA": [cgpa],
            "Internships": [internships],
            "Projects": [projects],
            "Workshops/Certifications": [workshops],
            "AptitudeTestScore": [aptitude_score],
            "SoftSkillsRating": [soft_skills],
            "ExtracurricularActivities": [extracurricular],
            "PlacementTraining": [placement_training],
            "SSC_Marks": [ssc_marks],
            "HSC_Marks": [hsc_marks]
        })
        
        # Make prediction
        status, prediction, confidence = predict_placement(input_data, model, transformer, scaler)
        
        if status:
            # Display results
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if status == "Placed":
                    st.markdown('<div class="prediction-card success">', unsafe_allow_html=True)
                    st.markdown(f"<h2>✅ {status}</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-card warning">', unsafe_allow_html=True)
                    st.markdown(f"<h2>❌ {status}</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence Score", f"{confidence:.2%}")
            
            with col3:
                st.metric("Raw Prediction", f"{prediction:.3f}")
            
            # Show input summary
            st.markdown("### Input Summary")
            input_summary = pd.DataFrame({
                "Feature": ["CGPA", "Internships", "Projects", "Workshops", "Aptitude Score", 
                           "Soft Skills", "Extracurricular", "Training", "SSC Marks", "HSC Marks"],
                "Value": [cgpa, internships, projects, workshops, aptitude_score, 
                         soft_skills, extracurricular, placement_training, ssc_marks, hsc_marks]
            })
            st.dataframe(input_summary, use_container_width=True)

def show_model_performance(data, model, transformer, scaler):
    """Display model performance metrics"""
    st.markdown("## 📈 Model Performance")
    
    # Prepare data for evaluation
    X = data.drop("PlacementStatus", axis=1)
    y = data["PlacementStatus"]
    
    # Transform and scale data
    X_transformed = transformer.transform(X)
    X_scaled = scaler.transform(X_transformed)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.2%}")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred_binary)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Placed', 'Placed'],
                    y=['Not Placed', 'Placed'],
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 