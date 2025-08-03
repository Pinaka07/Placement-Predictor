# 🎓 Placement Predictor

A comprehensive machine learning application that predicts student placement status based on various academic and personal factors.

## Features

- **Interactive Web Interface**: Beautiful Streamlit-based UI with multiple pages
- **Data Analysis**: Comprehensive visualizations and insights
- **Real-time Predictions**: Instant placement predictions with confidence scores
- **Model Performance**: Detailed metrics and confusion matrix
- **Responsive Design**: Modern, user-friendly interface

## Pages

1. **🏠 Home**: Overview with key metrics and insights
2. **📊 Data Analysis**: Interactive visualizations and data exploration
3. **🎯 Predict Placement**: Input form for making predictions
4. **📈 Model Performance**: Model evaluation metrics and confusion matrix

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script first** (if not already done):
   ```bash
   python placement_predictor.py
   ```

4. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Making Predictions

1. Navigate to the "🎯 Predict Placement" page
2. Fill in the student details:
   - **CGPA**: Cumulative Grade Point Average (6.0-10.0)
   - **Internships**: Number of internships completed
   - **Projects**: Number of projects completed
   - **Workshops/Certifications**: Number of workshops/certifications
   - **Aptitude Test Score**: Score in aptitude tests (0-100)
   - **Soft Skills Rating**: Rating of soft skills (1.0-5.0)
   - **Extracurricular Activities**: Yes/No
   - **Placement Training**: Yes/No
   - **SSC Marks**: Secondary School Certificate marks (0-100)
   - **HSC Marks**: Higher Secondary Certificate marks (0-100)

3. Click "Predict Placement" to get results

### Understanding Results

- **Placed**: Student is predicted to get placed
- **Not Placed**: Student is predicted to not get placed
- **Confidence Score**: How confident the model is in its prediction
- **Raw Prediction**: The actual numerical prediction value

## Data Analysis Features

- **Distribution Plots**: Visualize CGPA distribution and placement status
- **Correlation Analysis**: See which features most influence placement
- **Key Metrics**: Average scores and placement rates
- **Model Performance**: Accuracy, precision, recall, and F1-score

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: 11 student characteristics
- **Target**: Binary placement status (Placed/Not Placed)
- **Preprocessing**: One-hot encoding for categorical variables, standardization for numerical features

## File Structure

```
placement/
├── app.py                    # Streamlit web application
├── placement_predictor.py    # Training script
├── placementdata .csv        # Dataset
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── placement_prediction.pkl  # Trained model (generated)
├── transformer.pkl          # Data transformer (generated)
└── scaler.pkl              # Data scaler (generated)
```

## Technical Details

- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Machine Learning**: Scikit-learn for model training
- **Data Processing**: Pandas for data manipulation
- **Styling**: Custom CSS for enhanced UI

## Troubleshooting

1. **Model files not found**: Run `python placement_predictor.py` first
2. **Port already in use**: Use `streamlit run app.py --server.port 8502`
3. **Dependencies issues**: Update pip and reinstall requirements

## Contributing

Feel free to contribute by:
- Adding new features
- Improving the UI/UX
- Enhancing the model performance
- Adding more visualizations

## License

This project is open source and available under the MIT License. 