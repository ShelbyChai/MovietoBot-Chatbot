import random

import pandas as pd

from similarity_information_retrieval import build_genre_tfidf_vectorizer
from similarity_information_retrieval import build_summary_vectorizer
from similarity_information_retrieval import get_similar_movies
from similarity_information_retrieval import calculate_similarity
from similarity_information_retrieval import GENRES_LABEL
from similarity_information_retrieval import compute_query_title_similarity

from response_generator import genre_response
from response_generator import similar_genre_response
from response_generator import fallback_genre_response

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


"""
Main Chatbot Loop 
"""

genre_tfidf_vectorizer = build_genre_tfidf_vectorizer(movie_df['Genres'])
[summary_tfidf_vectorizer, summary_matrix] = build_summary_vectorizer(movie_df['Summary'])

intent = 'Game'
stop_list = ['Bye', 'Goodbye']
stop = False

# Mini Game variables
user_mini_game_point = 0


print(genre_tfidf_vectorizer.get_feature_names_out())

while not stop:
    query = input('Chatbot: What can I do for you?\nUser: ')
    if query not in stop_list:
        if intent == 'Genre':
            movie_recommendation(query, genre_tfidf_vectorizer)
        if intent == 'Game':
            random_movie_index = random.randint(0, len(movie_df))
            # Get 2 similar summary movies based on the random_movie_index
            index = get_similar_movies(movie_df.loc[random_movie_index], summary_tfidf_vectorizer, summary_matrix)
            # Keep track of the correct answer
            answer = [index[0][0], movie_df['Title'][index[0][0]]]
            # Shuffle the movie title
            random.shuffle(index)

            index_title_dict = {
                index[0][0]: movie_df['Title'][index[0][0]],
                index[1][0]: movie_df['Title'][index[1][0]],
                index[2][0]: movie_df['Title'][index[2][0]],
            }

            print("Chatbot: Let's get started! For this mini game, I scrambled 2 similar movie plot with the movie "
                  "you have to guess. \n" "Which movie below do you think correspond to this given summary -> " +
                  movie_df['Summary'].loc[random_movie_index] + "\n")

            print("1. " + movie_df['Title'][index[0][0]])
            print("2. " + movie_df['Title'][index[1][0]])
            print("3. " + movie_df['Title'][index[2][0]])

            user_answer = input('Your answer >>> ')

            if compute_query_title_similarity(user_answer, answer[1]) > 0.8:
                user_mini_game_point += 1
                print("Damn you are good! You now have " + str(user_mini_game_point) +
                      " point.")
            else:
                print(f"Oops, the answer is '" + answer[1] + "', your current score remains: " +
                      str(user_mini_game_point) + ".")

    else:
        print("Chatbot: Bye")
        stop = True
