import numpy as np
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
import uvicorn
from fastapi import FastAPI
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json


# --Preparing ML Preprocessing--
html_pattern = re.compile(r'<.*?>')
url_pattern = re.compile(r'(https?://\S+)|(www\.\S+)|(\S+\.\S+/\S+)')
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"  # other miscellaneous symbols
                           u"\U000024C2-\U0001F251"  # enclosed characters
                           "]+", flags=re.UNICODE)
punkt_pattern = re.compile(r"[^\w\s]")
STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stemmer = PorterStemmer()


def make_new_predictions_ml(inputText):

    # --Tokenization--
    tokens = nltk.word_tokenize(inputText)

    # --Cleaning Tokens--
    htmlTokensRm = ([html_pattern.sub(r'', word)
                    for word in tokens])  # Remove tokens containing html tags

    urlTokensRm = ([url_pattern.sub(r'', word)
                   for word in htmlTokensRm])  # Remove tokens containing urls

    emojiTokensRm = ([emoji_pattern.sub(r'', word)
                     for word in urlTokensRm])  # Remove tokens containing emojis

    # Remove tokens containing Punctuations
    punktTokensRm = ([punkt_pattern.sub(r'', word) for word in emojiTokensRm])

    # Remove tokens containing stopwords
    stopWordsRm = ([word for word in punktTokensRm if word not in STOPWORDS])

    lemmtizedTokensRm = ([lemmatizer.lemmatize(word)
                         for word in stopWordsRm])  # Lemmatize Tokens

    stemmedTokensRm = ([stemmer.stem(word)
                       for word in lemmtizedTokensRm])  # Stem Tokens

    # Remove repeating characters from tokens
    ReapeatTokensRm = (
        [re.sub(r'(\w)\1{2,}', r'\1', word) for word in stemmedTokensRm])

    digitTokensRm = ([word for word in ReapeatTokensRm if not re.search(
        r'\d', word)])  # Remove tokens containing digits

    underscoreTokensRm = ([word for word in digitTokensRm if not re.search(
        r'_|\w*_\w*', word)])  # Remove tokens containing underscore

    specialTokensRm = ([word for word in underscoreTokensRm if not re.search(
        r'[^a-zA-Z0-9\s]', word)])  # Remove tokens containing Special Characters

    # Remove tokens less than 2 characters
    cleanTokens = ([word for word in specialTokensRm if len(word) > 2])

    # --Vectorizing Tokens--
    Tfidf_Vectorizer = joblib.load(open('Tfidf_Vectorizer.pkl', 'rb'))

    tokensVectorized = Tfidf_Vectorizer.transform(
        [" ".join(cleanTokens)]).toarray()

    # --Predicting Review Class--
    model = joblib.load(open('ml_model.pkl', 'rb'))

    prediction = model.predict(tokensVectorized)

    categories = ['Negative', 'Positive']

    return categories[prediction[0]]


# --Preparing DL Preprocessing--
max_length = 8496
padding_type='post'
truncating_type="pre"

def make_new_dl_predictions(inputText):

    # --Preparing Preprocessing--
    html_pattern = re.compile(r'<.*?>')
    url_pattern = re.compile(r'(https?://\S+)|(www\.\S+)|(\S+\.\S+/\S+)')
    emoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"  # other miscellaneous symbols
                                  u"\U000024C2-\U0001F251"  # enclosed characters
                                "]+", flags=re.UNICODE)
    punkt_pattern = re.compile(r"[^\w\s]")
    STOPWORDS = set(stopwords.words('english'))
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    stemmer = PorterStemmer()

    # --Tokenization--
    tokens = nltk.word_tokenize(inputText)

    # --Cleaning Tokens--
    htmlTokensRm = ( [ html_pattern.sub(r'', word) for word in tokens ] ) # Remove tokens containing html tags

    urlTokensRm = ( [ url_pattern.sub(r'', word) for word in htmlTokensRm ] ) # Remove tokens containing urls

    emojiTokensRm = ( [ emoji_pattern.sub(r'', word) for word in urlTokensRm ] ) # Remove tokens containing emojis

    punktTokensRm = ( [ punkt_pattern.sub(r'', word) for word in emojiTokensRm ] ) # Remove tokens containing Punctuations

    stopWordsRm = ( [ word for word in punktTokensRm if word not in STOPWORDS ] ) # Remove tokens containing stopwords

    lemmtizedTokensRm = ( [ lemmatizer.lemmatize(word) for word in stopWordsRm ] ) # Lemmatize Tokens

    stemmedTokensRm = ( [ stemmer.stem(word) for word in lemmtizedTokensRm ] ) # Stem Tokens

    ReapeatTokensRm = ( [ re.sub(r'(\w)\1{2,}', r'\1', word) for word in stemmedTokensRm] )  # Remove repeating characters from tokens

    digitTokensRm = ( [ word for word in ReapeatTokensRm if not re.search(r'\d', word) ] ) # Remove tokens containing digits

    underscoreTokensRm = ( [ word for word in digitTokensRm if not re.search(r'_|\w*_\w*', word) ] ) # Remove tokens containing underscore

    specialTokensRm = ( [ word for word in underscoreTokensRm if not re.search(r'[^a-zA-Z0-9\s]', word) ] ) # Remove tokens containing Special Characters
    
    cleanTokens = ( [ word for word in specialTokensRm if len(word) > 2 ] )  # Remove tokens less than 2 characters

    # --Vectorizing Tokens--
    cleanSentence = [" ".join(cleanTokens)]

    # Load the Tokenizer dictionary
    tokenizer_dict = joblib.load(open('keras_tokenizer.pkl', 'rb'))

    # Convert the dictionary back to a Tokenizer object
    tokenizer = tokenizer_from_json(tokenizer_dict)

    sentence_sequences = tokenizer.texts_to_sequences(cleanSentence)

    padded_sentence_sentences = pad_sequences(sentence_sequences, maxlen = max_length, padding = padding_type, truncating = truncating_type)

    # --Predicting Review Class--
    dl_model = keras.models.load_model('dl_model.h5')

    predictionsProbs = dl_model.predict(padded_sentence_sentences)

    prediction = [1 if pred > 0.50 else 0 for pred in predictionsProbs]

    categories = ['Negative', 'Positive']

    return categories[prediction[0]]



app = FastAPI(
    title="Sentiment Model API",
    description="A Simple API that use Ml and Dl Models to predict the sentiment of the movie's reviews",
    version="0.1",
)



@app.get("/predict-review-ml")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict with ml model the sentiment of the content.
    :param review:
    :return: prediction
    """

    reviewPrediction = make_new_predictions_ml(review)

    return reviewPrediction


@app.get("/predict-review-dl")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict with dl model the sentiment of the content.
    :param review:
    :return: prediction
    """

    reviewPrediction = make_new_predictions_ml(review)

    return reviewPrediction


# Run Command:  ------------------------------- `uvicorn fastapi_deployment:app --reload` --------------------------#