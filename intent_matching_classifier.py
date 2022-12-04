import json

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

analyzer = TfidfVectorizer().build_analyzer()
sb_stemmer = SnowballStemmer('english')


def stemmed_words(doc):
    return (sb_stemmer.stem(w) for w in analyzer(doc))


def build_intent_matching_classifier():
    # Load all the required dataset
    with open('./data/intent_matching/small_talk_intent.json') as f:
        small_talk_intent = json.load(f)

    with open('./data/intent_matching/information_retrieval_game_intent.json') as f:
        information_retrieval_game_intent = json.load(f)

    with open('./data/intent_matching/information_retrieval_recommendation_intent.json') as f:
        information_retrieval_recommendation_intent = json.load(f)

    with open('./data/intent_matching/identity_management_intent.json') as f:
        identity_management_intent = json.load(f)

    with open('./data/intent_matching/question_answer_intent.json') as f:
        question_answer_intent = json.load(f)

    intent_label = {
        'small talk': small_talk_intent,
        'game': information_retrieval_game_intent,
        'movie recommendation': information_retrieval_recommendation_intent,
        'identity management': identity_management_intent,
        'question answer': question_answer_intent
    }

    data_inputs = []
    data_intents = []

    # Append the data inputs & intent into a list
    for label in intent_label.keys():
        for intent in intent_label[label]['intents']:
            for text in intent['text']:
                data_inputs.append(text)
                data_intents.append(label)

    intent_tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), ngram_range=(1, 3),
                                              analyzer=stemmed_words)
    x_train_tf = intent_tfidf_vectorizer.fit_transform(data_inputs)

    # Train the decision tree classifier
    intent_classifier = SVC(probability=True).fit(x_train_tf, data_intents)

    return [intent_label, intent_classifier, intent_tfidf_vectorizer]
