import random

import pandas as pd

from similarity_information_retrieval import build_genre_tfidf_vectorizer
from similarity_information_retrieval import build_summary_vectorizer
from similarity_information_retrieval import get_similar_movies
from similarity_information_retrieval import calculate_similarity
from similarity_information_retrieval import GENRES_LABEL
from similarity_information_retrieval import GAME_LABEL

# Import bot response function
from response_generator import genre_response
from response_generator import similar_genre_response
from response_generator import fallback_genre_response
from response_generator import display_movie_summary
from response_generator import correct_answer_response
from response_generator import incorrect_answer_response

# --------------------------------------------------------------------------
movie_df = pd.read_csv(r"./data/movie_dataset.csv")


def movie_recommendation(user_input, vectorizer):
    max_similarity = 0
    response = ''
    movie_list = []
    similar_movies = []

    for index, row in movie_df.iterrows():
        similarity = calculate_similarity(user_input, row[GENRES_LABEL], GENRES_LABEL, vectorizer)

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
Main Chatbot Loop 
"""

genre_tfidf_vectorizer = build_genre_tfidf_vectorizer(movie_df['Genres'])
[summary_tfidf_vectorizer, summary_matrix] = build_summary_vectorizer(movie_df['Summary'])

intent = GAME_LABEL
stop_list = ['Bye', 'Goodbye']
stop = False

# Mini Game variables
user_mini_game_point = 0

print(genre_tfidf_vectorizer.get_feature_names_out())

while not stop:
    query = input('Chatbot: What can I do for you?\nUser: ')
    if query not in stop_list:
        if intent == GENRES_LABEL:
            movie_recommendation(query, genre_tfidf_vectorizer)

        if intent == GAME_LABEL:
            user_mini_game_point = movie_guessing_game(user_mini_game_point)

    else:
        print("Chatbot: Bye")
        stop = True
