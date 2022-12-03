import random

import pandas as pd

# Import bot response functions
from response_generator import genre_response, similar_genre_response, fallback_genre_response, display_movie_summary, \
    correct_answer_response, incorrect_answer_response
from similarity_information_retrieval import build_genre_tfidf_vectorizer, build_summary_vectorizer, get_similar_movies, \
    calculate_similarity
from similarity_information_retrieval import RECOMMENDATION_LABEL, GAME_LABEL, SMALL_TALK_LABEL, IDENTITY_MANAGEMENT_LABEL
from intent_matching import build_intent_matching_classifier

# --------------------------------------------------------------------------
movie_df = pd.read_csv(r"data/information_retrieval/movie_dataset.csv")


def movie_recommendation(user_input, vectorizer):
    max_similarity = 0
    response = ''
    movie_list = []
    similar_movies = []

    for index, row in movie_df.iterrows():
        similarity = calculate_similarity(user_input, row['Genres'], RECOMMENDATION_LABEL, vectorizer)

        if similarity > max_similarity:
            max_similarity = similarity

        # Keep a list of maximum similarity response
        if similarity == 1:
            movie_list.append(genre_response(row))

        # Keep a list of similar movies response for suggestion
        if 0.7 <= similarity < 0.9:
            similar_movies.append([similarity, row])

        if similarity >= 0.9:
            response = genre_response(row)

    # Depends on the maximum similarity value, assign the corresponding response
    if max_similarity == 1:
        response = random.choice(movie_list)
    elif max_similarity >= 0.9:
        response = response
    elif similar_movies:
        # Sort and ranks the similar movies in decreasing relevance
        similar_movies = sorted(similar_movies, key=lambda x: x[0], reverse=True)
        num_suggestion = len(similar_movies)

        # Only select 3 highest relevance movies
        if num_suggestion >= 3:
            similar_movies = similar_movies[:3]

        response = similar_genre_response(similar_movies)
    else:
        response = fallback_genre_response()

    print(max_similarity)
    print('Chatbot: ' + response)


# Randomize a movie as the answer and find 2 most similar movies to it base on its summary.
# The user have to guess the correct answer from the 3 option and user will be awarded with
# point with each correct attempt.
def movie_guessing_game(game_point):
    response = ''
    # Randomize a movie as the answer
    random_movie_index = random.randint(0, len(movie_df))
    # Get 2 similar summary movies based on the random_movie_index
    index = get_similar_movies(movie_df.loc[random_movie_index], summary_tfidf_vectorizer, summary_matrix)
    # Keep track of the correct answer
    answer = [index[0][0], movie_df['Title'][index[0][0]], movie_df['Summary'][index[0][0]]]
    # Shuffle the movie title
    random.shuffle(index)

    title_dict = {
        index[0][0]: movie_df['Title'][index[0][0]],
        index[1][0]: movie_df['Title'][index[1][0]],
        index[2][0]: movie_df['Title'][index[2][0]],
    }

    display_movie_summary(title_dict, answer[2])

    user_answer = input('Your answer >>> ')

    similarity = calculate_similarity(user_answer, answer[1], intent, summary_tfidf_vectorizer)

    if similarity > 0.8:
        game_point += 1
        response = correct_answer_response(game_point)
    else:
        response = incorrect_answer_response(answer[1], game_point)

    print('Chatbot: ' + response)

    return game_point


"""
Chatbot
"""

print("Setting up movie database ....")
genre_tfidf_vectorizer = build_genre_tfidf_vectorizer(movie_df['Genres'])
[summary_tfidf_vectorizer, summary_matrix] = build_summary_vectorizer(movie_df['Summary'])
[intent_label, intent_classifier, intent_tfidf_vectorizer] = build_intent_matching_classifier()
print("Hiya, my name is Filmtobot, what can I do for you?")

# ---------------------------------
intent = RECOMMENDATION_LABEL
stop_list = ['Bye', 'Goodbye']
stop = False

# Mini Game variables
user_mini_game_point = 0

print(genre_tfidf_vectorizer.get_feature_names_out())

while not stop:
    similarity = 0
    maximum_similarity = 0

    user_query = input('User: ')

    # Intent classification
    vectorized_user_query = intent_tfidf_vectorizer.transform([user_query])
    user_intent = intent_classifier.predict(vectorized_user_query)

    class_keys = intent_classifier.classes_.tolist()
    probability_values = intent_classifier.predict_proba(vectorized_user_query).tolist()[0]
    # similarity_values = [value * 100 for value in similarity_values]

    # Construct an intent probability distribution dictionary
    # References: https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
    class_probability_dict = {class_keys[i]: probability_values[i] for i in range(len(class_keys))}

    print(class_probability_dict)
    print(user_intent)

    if user_query not in stop_list:
        intent_prediction_probability = class_probability_dict[user_intent[0]]
        # Only proceed with the intent if the classifier have high probability
        if intent_prediction_probability >= 0.9:
            if user_intent == RECOMMENDATION_LABEL:
                movie_recommendation(user_query, genre_tfidf_vectorizer)

            if user_intent == GAME_LABEL:
                user_mini_game_point = movie_guessing_game(user_mini_game_point)

            if user_intent == SMALL_TALK_LABEL:
                print('Doing small talk')

            if user_intent == IDENTITY_MANAGEMENT_LABEL:
                print('Doing identity management')

        elif 0.9 > intent_prediction_probability > 0.8:
            print("Do you mean XXX?")

        # Fallback mechanism
        else:
            print("Chatbot: I don't understand you :(.")

    else:
        print("Chatbot: Bye")
        stop = True

# for intent in intent_label['information retrieval']['intents']:
#     for text in intent['text']:
#         calculate_similarity()


# TODO: Hyperparameter tuning