import json

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


def train_intent_classifier():
    with open('./data/small_talk_intent.json') as f:
        small_talk_intent = json.load(f)

    with open('./data/information_retrieval_intent.json') as f:
        information_retrieval_intent = json.load(f)

    with open('./data/identity_management_intent.json') as f:
        identity_management_intent = json.load(f)

    intent_label = {
        'small talk': small_talk_intent,
        'information retrieval': information_retrieval_intent,
        'identity management': identity_management_intent
    }

    # Store the types of intent: small_talk, information_retrieval or identity management
    classes = []
    data_inputs = []
    data_intents = []
    # Store the sub category of the intent
    # intent_category = []

    for label in intent_label.keys():
        for intent in intent_label[label]['intents']:
            if intent['intent'] not in classes:
                classes.append(intent['intent'])

            for text in intent['text']:
                data_inputs.append(text)
                data_intents.append(label)
                # intent_category.append(intent['intent'])

    analyzer = TfidfVectorizer().build_analyzer()
    sb_stemmer = SnowballStemmer('english')

    def stemmed_words(doc):
        return (sb_stemmer.stem(w) for w in analyzer(doc))

    intent_tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), ngram_range=(1, 2),
                                              analyzer=stemmed_words)
    x_train_tf = intent_tfidf_vectorizer.fit_transform(data_inputs)

    # Train the decision tree classifier
    intent_classifier = DecisionTreeClassifier().fit(x_train_tf, data_intents)

    return [classes, intent_classifier, intent_tfidf_vectorizer]

# new_data = intent_tfidf_vectorizer.transform([user_input])
# user_intent = clf.predict(new_data)
