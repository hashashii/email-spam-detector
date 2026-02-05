# test_email.py
import pickle

def load_model():
    """Model & Vectorizer load කරනවා"""
    with open('models/spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def check_spam(email_text):
    """Email එකක් check කරනවා"""
    model, vectorizer = load_model()
    
    # Transform & Predict
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]
    probability = model.predict_proba(email_vec)[0]
    
    # Results
    if prediction == 1:
        result = "🚨 SPAM"
        confidence = probability[1] * 100
    else:
        result = "✅ SAFE"
        confidence = probability[0] * 100
    
    print(f"\n{'='*60}")
    print(f"Email: {email_text[:100]}...")
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"{'='*60}\n")
    
    return prediction

# Interactive Mode
if __name__ == "__main__":
    print("📧 Email Spam Checker - Interactive Mode")
    print("Type 'quit' to exit\n")
    
    while True:
        email = input("Enter email text: ")
        if email.lower() == 'quit':
            break
        check_spam(email)