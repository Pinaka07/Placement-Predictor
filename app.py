import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

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

# Load the trained pipeline (preprocessing + model bundled as one object)
@st.cache_resource
def load_pipeline():
    try:
        with open("pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except FileNotFoundError:
        st.error("pipeline.pkl not found. Please run placement_predictor.py first.")
        return None

@st.cache_resource
def load_test_split():
    try:
        with open("test_split.pkl", "rb") as f:
            split = pickle.load(f)
        return split["X_test"], split["y_test"]
    except FileNotFoundError:
        st.error("test_split.pkl not found. Please run placement_predictor.py first.")
        return None, None

@st.cache_data
def load_data():
    try:
        plc_data = pd.read_csv("placementdata .csv")
        plc_data['PlacementStatus'] = plc_data['PlacementStatus'].map({'NotPlaced': 0, 'Placed': 1})

        numeric_columns = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
                          'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
        for col in numeric_columns:
            plc_data[col] = pd.to_numeric(plc_data[col], errors='coerce')

        return plc_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def predict_placement(input_data, pipeline):
    """Make a prediction using the loaded pipeline (preprocessing + model)."""
    try:
        proba_placed = pipeline.predict_proba(input_data)[0][1]  # P(Placed)

        if proba_placed >= 0.5:
            status = "Placed"
            confidence = proba_placed
        else:
            status = "Not Placed"
            confidence = 1 - proba_placed

        return status, proba_placed, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def main():
    st.markdown('<h1 class="main-header">🎓 Placement Predictor</h1>', unsafe_allow_html=True)

    pipeline = load_pipeline()
    data = load_data()

    if pipeline is None or data is None:
        st.stop()

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
        show_prediction_page(pipeline)
    elif page == "📈 Model Performance":
        X_test, y_test = load_test_split()
        if X_test is None:
            st.stop()
        show_model_performance(pipeline, X_test, y_test)

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

    st.markdown("### Dataset Overview")
    st.dataframe(data.head(), use_container_width=True)

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

    st.markdown("### Feature Correlation with Placement")
    numeric_cols = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
                   'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']

    correlations = data[numeric_cols + ['PlacementStatus']].corr()['PlacementStatus'].sort_values(ascending=False)

    fig = px.bar(x=correlations.index[:-1], y=correlations.values[:-1],
                 color=correlations.values[:-1], color_continuous_scale='RdBu')
    fig.update_layout(title="Feature Correlation with Placement Status",
                     xaxis_title="Features", yaxis_title="Correlation")
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(pipeline):
    """Display prediction interface"""
    st.markdown("## 🎯 Predict Placement")

    st.markdown("Enter student details to predict placement status:")

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
        # No StudentID here — it's a row identifier, not a model feature.
        input_data = pd.DataFrame({
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

        status, proba_placed, confidence = predict_placement(input_data, pipeline)

        if status:
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
                st.metric("P(Placed)", f"{proba_placed:.3f}")

            st.markdown("### Input Summary")
            input_summary = pd.DataFrame({
                "Feature": ["CGPA", "Internships", "Projects", "Workshops", "Aptitude Score",
                           "Soft Skills", "Extracurricular", "Training", "SSC Marks", "HSC Marks"],
                "Value": [cgpa, internships, projects, workshops, aptitude_score,
                         soft_skills, extracurricular, placement_training, ssc_marks, hsc_marks]
            })
            st.dataframe(input_summary, use_container_width=True)

def show_model_performance(pipeline, X_test, y_test):
    st.markdown("## 📈 Model Performance")
    st.caption(f"Evaluated on {len(X_test)} held-out rows the model never trained on.")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")

    with col2:
        st.metric("Precision", f"{precision:.2%}")

    with col3:
        st.metric("Recall", f"{recall:.2%}")

    with col4:
        st.metric("F1-Score", f"{f1:.2%}")

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Placed', 'Placed'],
                    y=['Not Placed', 'Placed'],
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix (held-out test set)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()