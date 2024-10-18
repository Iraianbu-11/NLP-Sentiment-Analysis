import streamlit as st
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Load models and other resources
predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
STOPWORDS = set(stopwords.words("english"))

# Define the sentiment mapping
def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"

# Single prediction function
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return sentiment_mapping(y_predictions)

# Create pie chart for single prediction result
def get_distribution_graph(prediction):
    fig, ax = plt.subplots(figsize=(3, 3))
    labels = ['Positive', 'Negative']
    sizes = [1 if prediction == 'Positive' else 0, 1 if prediction == 'Negative' else 0]
    colors = ['green', 'red']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    
    return fig

# Streamlit app interface
st.title("Text Sentiment Predictor")

# Text input for single prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction button logic
if st.button("Predict"):
    if user_input:
        # Single text prediction
        prediction = single_prediction(predictor, scaler, cv, user_input)
        st.write(f"Predicted sentiment: {prediction}")
        
        # Display the distribution graph
        fig = get_distribution_graph(prediction)
        #st.pyplot(fig)
    else:
        st.error("Please enter text for prediction.")
