# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

print("="*70)
print("🚀 EMAIL SPAM DETECTOR - AI MODEL TRAINING")
print("="*70)

# ============================================
# STEP 1: Load Dataset
# ============================================
print("\n📂 STEP 1: Loading dataset...")
df = pd.read_csv('spam_dataset.csv')

# Clean data
df = df.dropna()  # null values ඉවත් කරනවා
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert to numbers

print(f"✅ Dataset loaded successfully!")
print(f"   📧 Total Emails: {len(df)}")
print(f"   ✅ Safe Emails: {len(df[df['label']==0])}")
print(f"   ⚠️ Spam Emails: {len(df[df['label']==1])}")

# ============================================
# STEP 2: Prepare Data
# ============================================
print("\n🔧 STEP 2: Preparing data for training...")
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split complete!")
print(f"   📚 Training set: {len(X_train)} emails ({len(X_train)/len(df)*100:.1f}%)")
print(f"   🧪 Test set: {len(X_test)} emails ({len(X_test)/len(df)*100:.1f}%)")

# ============================================
# STEP 3: Text Vectorization
# ============================================
print("\n🔢 STEP 3: Converting text to numbers (Vectorization)...")
vectorizer = TfidfVectorizer(
    max_features=3000,        # වැදගත්ම වචන 3000 විතරක්
    stop_words='english',     # "the", "is" වගේ අනවශ්‍ය වචන remove
    ngram_range=(1, 2),       # single words + word pairs
    min_df=2                  # අවම වශයෙන් 2 emails වලවත් තියෙන වචන
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Vectorization complete!")
print(f"   📊 Feature count: {X_train_vec.shape[1]} unique features")
print(f"   💾 Training data shape: {X_train_vec.shape}")

# ============================================
# STEP 4: Train AI Model
# ============================================
print("\n🧠 STEP 4: Training AI model...")
print("   ⏳ Please wait...")

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

print("✅ Training complete!")

# ============================================
# STEP 5: Evaluate Model Performance
# ============================================
print("\n📊 STEP 5: Evaluating model performance...")
print("   ⏳ Testing on unseen data...")

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*70}")
print(f"🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"{'='*70}")

print("\n📈 Detailed Classification Report:")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Spam Email']))

print("\n📉 Confusion Matrix:")
print("-"*70)
cm = confusion_matrix(y_test, y_pred)
print(f"✅ True Negatives (Correctly identified Safe):  {cm[0][0]}")
print(f"❌ False Positives (Safe marked as Spam):       {cm[0][1]}")
print(f"❌ False Negatives (Spam marked as Safe):       {cm[1][0]}")
print(f"✅ True Positives (Correctly identified Spam):  {cm[1][1]}")

# ============================================
# STEP 6: Save Model
# ============================================
print(f"\n💾 STEP 6: Saving trained model...")

# Create models directory
os.makedirs('models', exist_ok=True)

with open('models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved successfully!")
print(f"   📁 Model file: models/spam_model.pkl")
print(f"   📁 Vectorizer file: models/vectorizer.pkl")

# ============================================
# STEP 7: Live Testing
# ============================================
print(f"\n{'='*70}")
print("🧪 STEP 7: LIVE TESTING WITH SAMPLE EMAILS")
print(f"{'='*70}\n")

test_samples = [
    "Congratulations! You've won a $1000 gift card. Click here now!",
    "Hi, can we schedule our meeting for tomorrow at 3 PM?",
    "URGENT: Your account will be closed. Verify your identity now!",
    "The project report has been submitted. Please review when you can.",
    "Get rich quick! Make $5000 per week working from home!",
    "Thanks for your email. I'll get back to you soon.",
    "FREE FREE FREE! Win an iPhone 15 Pro Max today!!!",
    "Meeting reminder: Team standup at 10 AM tomorrow"
]

for i, email in enumerate(test_samples, 1):
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]
    probability = model.predict_proba(email_vec)[0]
    
    if prediction == 1:
        result = "🚨 SPAM"
        confidence = probability[1] * 100
        emoji = "⚠️"
    else:
        result = "✅ SAFE"
        confidence = probability[0] * 100
        emoji = "✅"
    
    print(f"{i}. {emoji} Email: {email[:55]}...")
    print(f"   Result: {result} | Confidence: {confidence:.1f}%")
    print("-"*70)

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE! MODEL IS READY TO USE!")
print(f"{'='*70}\n")

print("📌 Next Steps:")
print("   1. Run 'python test_email.py' to test with your own emails")
print("   2. Run 'python app.py' to start the web interface")
print("\n")