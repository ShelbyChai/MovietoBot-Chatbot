import random

import pandas as pd

# Import bot response functions
from response_generator import genre_response, similar_genre_response, fallback_genre_response, display_movie_summary, \
    correct_answer_response, incorrect_answer_response
from similarity_information_retrieval import build_genre_tfidf_vectorizer, build_summary_vectorizer, get_similar_movies, \
    calculate_it_similarity
from similarity_information_retrieval import RECOMMENDATION_LABEL, GAME_LABEL, SMALL_TALK_LABEL, IDENTITY_MANAGEMENT_LABEL
from intent_matching import build_intent_matching_classifier, build_tag_classifier
from identity_management import get_user_name

# --------------------------------------------------------------------------
movie_df = pd.read_csv(r"data/information_retrieval/movie_dataset.csv")


def movie_recommendation(user_input, vectorizer):
    max_similarity = 0
    response = ''
    movie_list = []
    similar_movies = []

    for index, row in movie_df.iterrows():
        similarity = calculate_it_similarity(user_input, row['Genres'], RECOMMENDATION_LABEL, vectorizer)

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

    # print(max_similarity)
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

    similarity_answer = calculate_it_similarity(user_answer, answer[1], GAME_LABEL, summary_tfidf_vectorizer)

    # print(similarity_answer)

    if similarity_answer > 0.8:
        game_point += 1
        response = correct_answer_response(game_point)
    else:
        response = incorrect_answer_response(answer[1], game_point)

    print('Chatbot: ' + response)

    return game_point


def identity_management(name):
    response = ''

    vectorized_identity_query = identity_tfidf_vectorizer.transform([user_query])
    identity_intent = identity_classifier.predict(vectorized_identity_query)

    identity_keys = identity_classifier.classes_.tolist()
    identity_probability_values = identity_classifier.predict_proba(vectorized_identity_query).tolist()[0]

    # Construct an identity intent probability distribution dictionary
    identity_dict = {identity_keys[i]: identity_probability_values[i] for i in range(len(identity_keys))}

    # print(identity_dict)
    # print(identity_intent)

    if identity_dict[identity_intent[0]] > 0.66:
        for intent in intent_corpus[IDENTITY_MANAGEMENT_LABEL]['intents']:
            if intent['intent'] == identity_intent[0]:
                response = random.choice(intent['responses'])

        # Store the username
        if identity_intent[0] == "InitialUserName":
            print('Chatbot: ' + response)
            name_query = input("User: ")
            name = get_user_name(name_query)
            print('Chatbot: Great! Hi ' + name + ", nice to meet you! What can I do for you today?")

        # Change the username
        if identity_intent[0] == "ChangeUserName":
            name = get_user_name(user_query)
            response = response.replace("<HUMAN>", name)
            print('Chatbot: ' + response)

        # Explicit name output
        if identity_intent[0] == "UserNameQuery":
            if name == "":
                print("Chatbot: I still don't know your name.")
            else:
                response = response.replace("<HUMAN>", name)
                print("Chatbot: " + response)
    else:
        print("Chatbot: Sorry, I don't understand.")

    return name


"""
Chatbot
"""
print("Setting up popcorn ....")
# Build intent matching classifier and tfidf_vectorizer
genre_tfidf_vectorizer = build_genre_tfidf_vectorizer(movie_df['Genres'])
[summary_tfidf_vectorizer, summary_matrix] = build_summary_vectorizer(movie_df['Summary'])
[intent_corpus, intent_classifier, intent_tfidf_vectorizer] = build_intent_matching_classifier()
[identity_classifier, identity_tfidf_vectorizer] = build_tag_classifier(intent_corpus[IDENTITY_MANAGEMENT_LABEL])
print("Chatbot: Hiya, my name is Filmtobot.")

# ---------------------------------
stop_list = ['Bye', 'Goodbye']
stop = False

user_name = ""
# Mini Game variables
user_mini_game_point = 0

# print(genre_tfidf_vectorizer.get_feature_names_out())

while not stop:
    # Display user's name if we got it
    if user_name != '':
        user_query = input(user_name + ": ")
    else:
        user_query = input('User: ')

    # Intent classification
    vectorized_user_query = intent_tfidf_vectorizer.transform([user_query])
    user_intent = intent_classifier.predict(vectorized_user_query)

    class_keys = intent_classifier.classes_.tolist()
    probability_values = intent_classifier.predict_proba(vectorized_user_query).tolist()[0]

    # Construct an intent probability distribution dictionary
    class_probability_dict = {class_keys[i]: probability_values[i] for i in range(len(class_keys))}

    print(class_probability_dict)
    # print(user_intent)

    if user_query not in stop_list:
        intent_prediction_probability = class_probability_dict[user_intent[0]]
        print(intent_prediction_probability)

        # Only proceed with the intent if the classifier have confidence score on the class
        if intent_prediction_probability >= 0.8:
            if user_intent == RECOMMENDATION_LABEL:
                movie_recommendation(user_query, genre_tfidf_vectorizer)

            if user_intent == GAME_LABEL:
                user_mini_game_point = movie_guessing_game(user_mini_game_point)

            if user_intent == SMALL_TALK_LABEL:
                print('Doing small talk')

            if user_intent == IDENTITY_MANAGEMENT_LABEL:
                user_name = identity_management(user_name)

        elif 0.8 > intent_prediction_probability > 0.7:
            print("Chatbot: Can you please reformulate your query?")

        else:
            print("Chatbot: Sorry, I don't understand.")

    else:
        print("Chatbot: Bye")
        stop = True


# TODO: Hyperparameter tuning
# TODO: Save the model using pickle
# TODO: Maybe 1 intent_prediction_probability for all and provide fallback mechanism if sim 0.7-0.8

# Functionality
# TODO: Small Talk
# TODO: Question and Answering (Bot ask question or user ask question)
# TODO: More functionality on Information retrieval (Retrieve Summary? Generate story from keyword?)
# Generate movie plot: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots