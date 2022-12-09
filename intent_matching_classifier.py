import json

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Evaluation visualisation library
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

    intent_tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), ngram_range=(1, 2),
                                              analyzer=stemmed_words)

    X_train_input, X_test_input, y_train_intent, y_test_intent = train_test_split(data_inputs, data_intents, stratify=data_intents, test_size=0.2, random_state=42)

    # x_train_tf = intent_tfidf_vectorizer.fit_transform(data_inputs)
    X_train_tf = intent_tfidf_vectorizer.fit_transform(X_train_input)

    # Train the classifier
    intent_classifier = SVC(random_state=42, probability=True).fit(X_train_tf, y_train_intent)

    # Classifier evaluation section
    # References: Code adapted from COMP 3009 Machine Learning module
    # X_test_tf = intent_tfidf_vectorizer.transform(X_test_input)
    # evaluation_accuracy = intent_classifier.score(X_test_tf, y_test_intent)
    # print("SVM intent classification accuracy: {:.2f}%".format(evaluation_accuracy * 100))
    #
    # # Plot confusion matrix,
    # y_pred = intent_classifier.predict(X_test_tf)
    # print(y_pred)
    # cm = confusion_matrix(y_test_intent, y_pred)
    # fig, ax = plt.subplots(figsize=(1, 1))
    # fig.set_size_inches(7, 4)
    # ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    #
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(x=j, y=i,
    #                 s=cm[i, j],
    #                 va='center', ha='center')
    #
    # classes = intent_classifier.classes_
    #
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # plt.xlabel('Predicted Values')
    # plt.ylabel('Actual Values')
    # plt.show()
    #
    # # Save the figure to evaluation folder
    # fig.savefig('./evaluation/intent_matching_confusion_matrix.png', dpi=100)
    #
    # print(classification_report(y_test_intent, y_pred))

    return [intent_label, intent_classifier, intent_tfidf_vectorizer]


# Evaluation of intent classifier
# [intent_label, intent_classifier, intent_tfidf_vectorizer] = build_intent_matching_classifier()

