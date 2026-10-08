
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd


def StopWordCountDict(text):
    """
    Takes in a act text and returns a dictionary of stop words and their counts, as well as the ratio of stop words to non-stop words
    """
    # Tokenize the text into words
    text = ' '.join(text)
    words = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    
    # Create a list to store found stopwords
    stopWordsFound = []
    stopWordsFoundCount = 0
    nonStopWordsFoundCount = 0
    for word in words:
        if word.lower() in stopWords:
            stopWordsFound.append(word.lower())
            stopWordsFoundCount += 1
        else:
            nonStopWordsFoundCount += 1

    fDist = nltk.FreqDist(stopWordsFound)
    return fDist.most_common(), stopWordsFoundCount/(nonStopWordsFoundCount + stopWordsFoundCount)



def ProportionalityMatrix(dialogueGroups):
    """
    Takes in a dictionary of dialogue acts
    Returns a DataFrame where each cell contains the proportion of a specific stop word in that dialogue act
    """
    # create a DataFrame to store the results for further comparison
    # rows contain the dialogue acts, columns contain the stop words found in the dialogue acts
    stopWords = set(stopwords.words('english'))
    dialogueStopWords = {}
    for act in dialogueGroups.keys():
        text = ' '.join(dialogueGroups[act])
        words = nltk.word_tokenize(text)
        totalTokens = len(words)

        perActStopWords = {}
        for word in words:
            word = word.lower()
            if word in stopWords:
                if word in perActStopWords:
                    perActStopWords[word] += 1
                else:
                    perActStopWords[word] = 1

        wordFrequencies = {}
        for word, count in perActStopWords.items():
            wordFrequencies[word] = count / totalTokens
        dialogueStopWords[act] = wordFrequencies

    return pd.DataFrame.from_dict(dialogueStopWords, orient='index').fillna(0)



def StopWordPropToFreqTokens(text):
    """
    calculates the precence of stop words in the most frequent tokens in dialogue acts
    uses the head/tail break method to determine the most frequent tokens
    """
#     
    fDistNorm = nltk.FreqDist(text)
    stopWords = set(stopwords.words('english'))

    totalCount = 0
    for word, count in fDistNorm.most_common():
        totalCount += count
    mean = totalCount / len(fDistNorm)

    stopCount = 0
    outSideCount = 0
    for word, count in fDistNorm.most_common():
        if count < mean:
            break
        if word in stopWords:
            stopCount += 1
        else:
            outSideCount += 1

    return stopCount / (stopCount + outSideCount), mean