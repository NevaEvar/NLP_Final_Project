import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm')


def PreprocessText(text):
    """
    Cleans text from non alphabet characters and extra spaces
    """
    text = ' '.join(text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text


def RemoveStopWords(text):
    """
    Tokenize and remove stopwords
    """
    words = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    
    tokens = []
    for word in words:
            if word.lower() not in stopWords:
                tokens.append(word.lower())
    
    return tokens


#geeks for geeks implementation
def Lemmatize(tokens):
    """
    Returns the lemmatized version of the text back
    """
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

