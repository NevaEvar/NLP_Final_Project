# word precense feature and code structure inspired by co-pilot approach suggestion on classification 
# simple binary word precense implementation
import re
import nltk
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from nltk.corpus import nps_chat


def DialogueActFeatureNames(text):
    """
    Return binary word presence and bigram precense features for an utterance
    """
    tokens = re.findall(r"[a-z0-9']+", str(text).lower())
    features = set()
    for token in tokens:
        features.add(f'word = {token}')

    for i in range(len(tokens) - 1): 
        left = tokens[i]
        right = tokens[i + 1]
        features.add(f'bigram = {left}_{right}')

    return features


def DialogueActFeatures(text, vocabulary):
    """
    Extract in vocabulary word and adjacent word presence features
    """
    allFeatures = DialogueActFeatureNames(text)
    features = {}
    for feature in allFeatures: 
        if feature in vocabulary:
            features[feature] = True
            
    return features


def TrainClassifier(examples, maxFeatures=3000):
    """
    Fit an NLTK Naive Bayes model from (utterance, dialogue act) pairs
    """
    featureCounts = Counter()
    for text, label in examples:
        features = DialogueActFeatureNames(text)
        
        for feature in features: 
            featureCounts[feature] += 1

    vocabulary = set()
    mostCommonFeatures = featureCounts.most_common(maxFeatures)
    for feature, count in mostCommonFeatures:
        vocabulary.add(feature)

    featureSets = []
    for text, label in examples:
        features = DialogueActFeatures(text, vocabulary)
        featureSets.append((features, label))

    classifier = nltk.NaiveBayesClassifier.train(featureSets)
    return vocabulary, classifier


def ExamplesFromFiles(fileIds):
    """
    Returns a (utterance, dialogue act) pairs
    """
    examples = []
    for fileId in fileIds:
        posts = nps_chat.xml_posts(fileId)

        for post in posts:
            if post.text and post.text.strip():
                text = post.text.strip()
                label = post.attrib['class']
                examples.append((text, label))

    return examples


def PredictDialogueActs(texts, dialogueActVocabulary, dialogueActClassifier):
    """
    Predict NPS Chat dialogue act labels for utterances from any corpus
    """
    featureSets = []
    for text in texts: 
        featureSets.append(DialogueActFeatures(text, dialogueActVocabulary))

    return dialogueActClassifier.classify_many(featureSets)
