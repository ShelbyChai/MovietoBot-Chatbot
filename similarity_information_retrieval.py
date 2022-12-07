from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------
# Label for intent
GAME_LABEL = 'game'
RECOMMENDATION_LABEL = 'movie recommendation'
SMALL_TALK_LABEL = 'small talk'
IDENTITY_MANAGEMENT_LABEL = 'identity management'
QUESTION_ANSWER_LABEL = 'question answer'

analyzer = TfidfVectorizer().build_analyzer()
sb_stemmer = SnowballStemmer('english')


def stemmed_words(doc):
    return (sb_stemmer.stem(w) for w in analyzer(doc))


def build_movie_recommendation_vectorizer(genre_corpus):
    # Genres column text pre-processing
    genres_list = [gen.split('|') for gen in genre_corpus if '|' in gen]
    genres_list.extend(gen for gen in genre_corpus if '|' not in gen)
    genres_list = [genre for genres in genres_list for genre in genres]

    # Build a vectorizer with text tokenization, convert lowercase, remove stop words and word stemming
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'), analyzer=stemmed_words)

    # Fit the formatted Genres information_retrieval into the vectorizer
    tfidf_vectorizer.fit_transform(genres_list)

    return tfidf_vectorizer


def calculate_it_similarity(query, document, intent, vectorizer):
    similarity = 0

    # Compute the cosine similarity between the user input with the correct Movie Title
    if intent == GAME_LABEL:
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, analyzer=stemmed_words)

        movie_titles_matrix = tfidf_vectorizer.fit_transform([document])
        user_answer_matrix = tfidf_vectorizer.transform([query])

        similarity = cosine_similarity(user_answer_matrix, movie_titles_matrix)
        # print(similarity)

    # Genre Searching
    if intent == RECOMMENDATION_LABEL:

        if '|' in document:
            document = document.split('|')
            document = ' '.join(document)

        query_tfidf_vector = vectorizer.transform([query])
        document_tfidf_vector = vectorizer.transform([document])

        # Compute the cosine similarity between the user_query and genre row document
        similarity = cosine_similarity(query_tfidf_vector, document_tfidf_vector)

    return similarity


def build_vectorizer_and_matrix(corpus):
    # Build a vectorizer with text tokenization, convert lowercase, remove stop words, word stemming
    # and bi-grams
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'),
                                       analyzer=stemmed_words, ngram_range=(1, 2))

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    return [tfidf_vectorizer, tfidf_matrix]


def rank_similar_documents(input_data, tfidf_vectorizer, corpus_matrix):
    # Calculate the tfidf vector of the selected movie to be guessed
    document_tfidf_vector = tfidf_vectorizer.transform([input_data])
    # Find the cosine similarity of the summary corpus with the selected movie summary
    similarity = cosine_similarity(corpus_matrix, document_tfidf_vector)
    similarity = list(similarity)
    # Add an index column to the list
    for index, item in enumerate(similarity):
        similarity[index] = [index, item]
    # Sort the summary similarity in descending order
    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
    top_similarity = similarity[0:3]

    return top_similarity


def build_tfidf_vectorizer(corpus):
    data_inputs = []

    for intent in corpus['intents']:
        for text in intent['text']:
            data_inputs.append(text)

    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english'),
                                       analyzer=stemmed_words, ngram_range=(1, 2))

    tfidf_vectorizer.fit_transform(data_inputs)

    # print(im_matrix)
    # print(tfidf_vectorizer.get_feature_names_out())

    return tfidf_vectorizer


def calculate_similarity(query, document, vectorizer):
    query_tfidf_vector = vectorizer.transform([query])
    document_tfidf_vector = vectorizer.transform([document])

    # Compute the cosine similarity between the user_query and genre row document
    similarity = cosine_similarity(query_tfidf_vector, document_tfidf_vector)

    return similarity
