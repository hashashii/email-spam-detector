from flask import Flask, request, render_template, jsonify
import pickle
import os
import sys

app = Flask(__name__)

# Model එක load කිරීමට පෙර folder එක තිබේදැයි පරීක්ෂා කිරීම
MODEL_PATH = 'models/spam_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("ERROR: Model files not found in 'models/' folder!")
        return None, None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Models loaded successfully!")
        return model, vectorizer
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_email():
    if model is None or vectorizer is None:
        return jsonify({'success': False, 'error': 'Model not loaded on server'})

    try:
        data = request.get_json()
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({'success': False, 'error': 'No email text provided'})

        # Predict
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)[0]
        probability = model.predict_proba(email_vec)[0]
        
        is_spam = bool(prediction)
        confidence = float(probability[prediction] * 100)
        
        return jsonify({
            'success': True,
            'is_spam': is_spam,
            'confidence': round(confidence, 2),
            'message': '🚨 SPAM EMAIL' if is_spam else '✅ SAFE EMAIL'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Debug mode එකෙන් වැරැද්ද හරියටම බලාගන්න පුළුවන්
    app.run(debug=True, port=5000)