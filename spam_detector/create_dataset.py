# create_dataset.py
import pandas as pd
import urllib.request

print("📥 Downloading dataset...")

# UCI SMS Spam Dataset download කරනවා
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

try:
    # Download & Load
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    
    # Save locally
    df.to_csv('spam_dataset.csv', index=False)
    
    print("\n✅ Dataset Downloaded Successfully!")
    print(f"📊 Total Emails: {len(df)}")
    print(f"✅ Ham (Safe): {len(df[df['label']=='ham'])}")
    print(f"⚠️ Spam: {len(df[df['label']=='spam'])}")
    print(f"\n📁 Saved as: spam_dataset.csv")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n⚠️ Backup method: Creating sample dataset...")
    
    # Backup: Sample dataset හදනවා
    sample_data = {
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 100,
        'text': [
            'WINNER!! Free entry to win £1000 cash prize',
            'Hi, how are you doing today?',
            'Congratulations! Claim your free iPhone now',
            'Can we meet for lunch tomorrow?',
            'URGENT! Your account needs verification',
            'The meeting has been rescheduled to 3pm'
        ] * 100
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('spam_dataset.csv', index=False)
    
    print("✅ Sample dataset created!")
    print(f"📊 Total Emails: {len(df)}")