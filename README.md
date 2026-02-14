# Iris Classifier Streamlit App 

## Files created
1. training_model.py        : Training the RandomForest model and save rf_model.joblib
2. iris_model.joblib       : Trained model and metadata (features, target_names)
3. Iris_app.py                : Streamlit app for prediction and exploration
4. data_exploration.py   : To compute descriptive stats and save histograms
5. requirements.txt      : Python packages required
6. README.md             : This file

## How to run
1. First we need to create and activate a virtual environment (using latest python version as 3.11.9)

2. Install required modules:
   pip install -r requirements.txt

3. To retrain the model:
   python train_model.py

   This will create or update the 'rf_model.joblib' in the current folder.

4. To run the Streamlit app:
   streamlit run Iris_app.py

5. Open the displayed local URL in browser (usually http://localhost:8501)

## Notes
1. The app contains two modes: Prediction and Data Exploration ( using histograms and scatter plots).
2. The trained model file 'rf_model.joblib' is included.
