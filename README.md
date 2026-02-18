# Breast Tumor Assessment (Machine Learning Prototype)

This project is a small, focused machine learning application built to demonstrate how a classification model can be presented responsibly in a healthcare-related context. The app estimates whether a tumor’s numeric diagnostic features more closely resemble benign or malignant cases.

The goal of this project is not to build a medical tool. It is to show how machine learning, user experience, and ethical framing can work together in a health informatics setting.


## What this app does

- Accepts mean diagnostic feature values (e.g., radius, texture, area).
- Runs those inputs through a trained classification model.
- Returns a model assessment: closer to benign patterns or malignant patterns.
- Displays model confidence (when available).
- Includes input validation to catch obvious data entry mistakes.
- Provides performance transparency using a confusion matrix and metrics.

This is designed as a decision-support style prototype, not a prediction machine.


## Important medical disclaimer

This tool is for educational and demonstration purposes only. It is not a medical device and it does not provide a diagnosis.

The model output is a statistical estimate based on patterns learned from historical training data. It does not incorporate:

- patient medical history  
- imaging review (mammogram, ultrasound, MRI)  
- pathology interpretation  
- laboratory confirmation  
- clinical judgment  

If you have concerns about your health, consult a licensed healthcare professional. This project is meant to demonstrate responsible machine learning design in a healthcare-adjacent space, not to replace medical expertise.


## Why this is more than just a classifier

In health-related applications, how a model is presented matters just as much as its accuracy. This project intentionally avoids authoritative or alarmist language. Instead, it:

- Frames results as “model assessments,” not diagnoses.
- Displays probability as confidence, not certainty.
- Warns users when inputs look unrealistic or possibly mistyped.
- Surfaces limitations clearly.
- Emphasizes transparency over performance marketing.

The interface is designed to reflect principles from health informatics and human-centered AI.


## Tech stack

- Python  
- Streamlit  
- scikit-learn  
- NumPy  
- joblib  
- Matplotlib  


## Project structure
```
breast-tumor-classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/
│   ├── classifier.pkl
│   ├── conf_matrix.pkl
│   ├── metrics.txt

```


## How to run locally

1. Clone the repository:
git clone https://github.com/vaishuchr/tumor-assessment-prototype.git
cd tumor-assessment-prototype

3. Install dependencies:
    pip install -r requirements.txt

4. Run the Streamlit app:
    streamlit run app.py






