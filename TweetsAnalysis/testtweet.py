import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        text = text.lower().strip()
        return text
    return ""

class TweetAnalyzer:
    def __init__(self):
        try:
            # Load the original dataset to fit the tokenizers
            print("Loading dataset...")
            df = pd.read_csv('text.csv')
            df['cleaned_text'] = df['text'].apply(clean_text)
            
            # Custom load function with optimizer handling
            print("Loading models...")
            self.sentiment_model = self._load_model_safely('analisis_tweets.h5')
            self.emotion_model = self._load_model_safely('emotion_classifier_v3.h5')
            
            # Constants
            self.MAX_WORDS = 10000
            self.MAX_LEN = 100
            self.MAX_WORDS_EMOTION = 10000
            self.MAX_LEN_EMOTION = 100
            
            # Initialize and fit tokenizers
            print("Preparing tokenizers...")
            self.sentiment_tokenizer = Tokenizer(num_words=self.MAX_WORDS, oov_token='<OOV>')
            self.emotion_tokenizer = Tokenizer(num_words=self.MAX_WORDS_EMOTION, oov_token='<OOV>')
            
            self.sentiment_tokenizer.fit_on_texts(df['cleaned_text'])
            self.emotion_tokenizer.fit_on_texts(df['cleaned_text'])
            
            # Labels
            self.sentiment_labels = ['negative', 'neutral', 'positive']
            self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
            
            print("Initialization complete!")
            
        except Exception as e:
            raise Exception(f"Initialization failed: {str(e)}")

    def _load_model_safely(self, model_path):
        """Safely load model with custom optimizer configuration"""
        try:
            # Custom objects to handle optimizer compatibility
            custom_objects = {
                'Adam': tf.keras.optimizers.legacy.Adam,
                'Adamax': tf.keras.optimizers.legacy.Adamax,
                'Nadam': tf.keras.optimizers.legacy.Nadam,
                'RMSprop': tf.keras.optimizers.legacy.RMSprop,
                'SGD': tf.keras.optimizers.legacy.SGD,
                'Adadelta': tf.keras.optimizers.legacy.Adadelta,
                'Adagrad': tf.keras.optimizers.legacy.Adagrad,
                'Ftrl': tf.keras.optimizers.legacy.Ftrl
            }
            
            return tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False  # Load without compilation
            )
        except Exception as e:
            raise Exception(f"Error loading model {model_path}: {str(e)}")
    
    def analyze_tweet(self, text):
        try:
            # Clean the text
            cleaned_text = clean_text(text)
            
            # Prepare text for sentiment analysis
            sentiment_sequence = self.sentiment_tokenizer.texts_to_sequences([cleaned_text])
            sentiment_padded = pad_sequences(sentiment_sequence, maxlen=self.MAX_LEN)
            
            # Prepare text for emotion analysis
            emotion_sequence = self.emotion_tokenizer.texts_to_sequences([cleaned_text])
            emotion_padded = pad_sequences(emotion_sequence, maxlen=self.MAX_LEN_EMOTION)
            
            # Get predictions
            sentiment_pred = self.sentiment_model.predict(sentiment_padded, verbose=0)[0]
            emotion_pred = self.emotion_model.predict(emotion_padded, verbose=0)[0]
            
            # Get sentiment (no change needed here as it's working)
            sentiment_idx = np.argmax(sentiment_pred)
            sentiment = self.sentiment_labels[sentiment_idx]
            sentiment_conf = sentiment_pred[sentiment_idx]
            
            # Get emotion safely
            emotion_idx = np.argmax(emotion_pred[:len(self.emotion_labels)])  # Only consider valid indices
            emotion = self.emotion_labels[emotion_idx]
            emotion_conf = emotion_pred[emotion_idx]
            
            # Get top 3 emotions safely
            # First, get indices only for valid emotions
            valid_pred = emotion_pred[:len(self.emotion_labels)]
            top_indices = np.argsort(valid_pred)[-3:][::-1]
            
            top_3_emotions = {
                self.emotion_labels[idx]: float(emotion_pred[idx])
                for idx in top_indices
            }
            
            return {
                'sentiment': sentiment,
                'sentiment_confidence': float(sentiment_conf),
                'emotion': emotion,
                'emotion_confidence': float(emotion_conf),
                'top_3_emotions': top_3_emotions
            }
        except Exception as e:
            raise Exception(f"Error analyzing tweet: {str(e)}")

def main():
    # Initialize analyzer
    print("\nInitializing Tweet Analyzer...")
    try:
        analyzer = TweetAnalyzer()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return
    
    while True:
        print("\n" + "="*50)
        print("Tweet Sentiment and Emotion Analyzer")
        print("="*50)
        print("\nEnter your tweet (or 'quit' to exit):")
        
        # Get user input
        tweet = input("> ").strip()
        
        if tweet.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if not tweet:
            print("\nPlease enter a valid tweet!")
            continue
        
        try:
            # Analyze tweet
            print("\nAnalyzing...")
            results = analyzer.analyze_tweet(tweet)
            
            # Print results
            print("\nResults:")
            print("-"*20)
            print(f"Sentiment: {results['sentiment']} ({results['sentiment_confidence']:.2%} confidence)")
            print(f"\nPrimary Emotion: {results['emotion']} ({results['emotion_confidence']:.2%} confidence)")
            
            print("\nTop 3 Emotions:")
            for emotion, confidence in results['top_3_emotions'].items():
                print(f"- {emotion}: {confidence:.2%}")
                
        except Exception as e:
            print(f"\nError analyzing tweet: {str(e)}")
            print("Please try again with a different input.")

if __name__ == "__main__":
    main()