from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

app = Flask(__name__)

# Function to download NLTK data
def download_nltk_data():
    nltk.download('stopwords')

# Load the model and vectorizer
def load_or_create_model():
    global model, vectorizer
    
    try:
        model = pickle.load(open('spam_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    except FileNotFoundError:
        # If model files don't exist, we'll train a simple model
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        
        # Sample spam/ham dataset
        messages = [
            ("Free entry in 2 a wkly comp to win FA Cup final tkts", "spam"),
            ("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot", "spam"),
            ("Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles", "spam"),
            ("SIX chances to win CASH! Reply to this message", "spam"),
            ("WINNER!! As a valued network customer you have been selected to receive a £900 prize", "spam"),
            ("Hi, how are you doing today?", "ham"),
            ("What time is the meeting scheduled for tomorrow?", "ham"),
            ("I'll be home by 7. We can watch a movie.", "ham"),
            ("Please let me know when you submit the report", "ham"),
            ("I'm going to the grocery store, need anything?", "ham"),
            ("Call me when you get a chance", "ham"),
            ("The project deadline has been extended to next week", "ham"),
            ("Don't forget to bring your laptop to the meeting", "ham"),
            ("Your ticket has been confirmed. Reference: ABC123", "ham"),
            ("Sorry, I can't talk right now. I'll call you back later", "ham")
        ]
        
        X = [msg for msg, label in messages]
        y = [1 if label == "spam" else 0 for msg, label in messages]
        
        # Create and train a simple model
        vectorizer = TfidfVectorizer(max_features=1000)
        X_vectorized = vectorizer.fit_transform(X)
        model = MultinomialNB()
        model.fit(X_vectorized, y)
        
        # Save the model and vectorizer
        pickle.dump(model, open('spam_model.pkl', 'wb'))
        pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Initialize NLTK and model at startup
download_nltk_data()
load_or_create_model()

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and stem
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    tokens = [ps.stem(word) for word in tokens if word not in stopwords_set]
    
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if not message:
            return jsonify({'error': 'No message provided'})
        
        # Preprocess the message
        processed_message = preprocess_text(message)
        
        # Vectorize the message
        message_vectorized = vectorizer.transform([processed_message])
        
        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        
        # Get probability
        probability = round(max(model.predict_proba(message_vectorized)[0]) * 100, 2)
        
        # Return result
        result = {
            'message': message,
            'is_spam': bool(prediction),
            'probability': probability,
            'classification': 'Spam' if prediction == 1 else 'Not Spam'
        }
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

