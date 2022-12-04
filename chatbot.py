import ast
import random
import pandas as pd

from csv import writer
from identity_management import get_user_name
from intent_matching_classifier import build_intent_matching_classifier
# Import bot response functions
from response_generator import genre_response, similar_genre_response, fallback_genre_response, display_movie_summary, \
    correct_answer_response, incorrect_answer_response, get_datetime_response
from similarity_information_retrieval import RECOMMENDATION_LABEL, GAME_LABEL, SMALL_TALK_LABEL, \
    IDENTITY_MANAGEMENT_LABEL, QUESTION_ANSWER_LABEL
from similarity_information_retrieval import build_genre_tfidf_vectorizer, build_tfidf_vectorizer_with_matrix, \
    rank_similar_documents, \
    calculate_it_similarity, calculate_similarity, build_tfidf_vectorizer

# --------------------------------------------------------------------------
# Load the data
movie_df = pd.read_csv(r"data/information_retrieval/movie_dataset.csv")
question_answer_df = pd.read_csv(r"data/information_retrieval/movie_question_answer.csv")


def movie_recommendation(user_input, vectorizer):
    max_similarity = 0
    recommendation_response = ''
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
            recommendation_response = genre_response(row)

    # Depends on the maximum similarity value, assign the corresponding response
    if max_similarity == 1:
        recommendation_response = random.choice(movie_list)
    elif max_similarity >= 0.9:
        recommendation_response = recommendation_response
    elif similar_movies:
        # Sort and ranks the similar movies in decreasing relevance
        similar_movies = sorted(similar_movies, key=lambda x: x[0], reverse=True)
        num_suggestion = len(similar_movies)

        # Only select 3 highest relevance movies
        if num_suggestion >= 3:
            similar_movies = similar_movies[:3]

        recommendation_response = similar_genre_response(similar_movies)
    else:
        recommendation_response = fallback_genre_response()

    # print(max_similarity)
    print('Chatbot: ' + recommendation_response)


# Randomize a movie as the answer and find 2 most similar movies to it base on its summary.
# The user have to guess the correct answer from the 3 option and user will be awarded with
# point with each correct attempt.
def movie_guessing_game(game_point):
    game_response = ''
    # Randomize a movie as the answer
    random_movie_index = random.randint(0, len(movie_df))
    # Get 2 similar summary movies based on the random_movie_index
    index = rank_similar_documents(movie_df.loc[random_movie_index]['Summary'], summary_tfidf_vectorizer,
                                   summary_matrix)
    # Keep track of the correct answer
    answer = [index[0][0], movie_df['Title'][index[0][0]], movie_df['Summary'][index[0][0]]]
    # Shuffle the movie title
    random.shuffle(index)

    title_dict = {
        index[0][0]: movie_df['Title'][index[0][0]],
        index[1][0]: movie_df['Title'][index[1][0]],
        index[2][0]: movie_df['Title'][index[2][0]],
    }

    display_movie_summary(title_dict, answer[2], user_name)

    user_answer = input('Your answer >>> ')

    similarity_answer = calculate_it_similarity(user_answer, answer[1], GAME_LABEL, summary_tfidf_vectorizer)

    # print(similarity_answer)

    if similarity_answer > 0.8:
        game_point += 1
        game_response = correct_answer_response(game_point)
    else:
        game_response = incorrect_answer_response(answer[1], game_point)

    print('Chatbot: ' + game_response)

    return game_point


def small_talk_and_identity_management(name, user_intent, stop):
    similarity = 0
    maximum_similarity = 0
    tag = ''
    bot_response = ''

    # Select the subcategory with the highest similarity between the user query and the text in the identity_management_intent corpus
    for intent in intent_corpus[user_intent]['intents']:
        for text in intent['text']:
            # Assign the vectorizer depends on the user intent
            if user_intent == IDENTITY_MANAGEMENT_LABEL:
                similarity = calculate_similarity(user_query, text, im_tfidf_vectorizer)
            elif user_intent == SMALL_TALK_LABEL:
                similarity = calculate_similarity(user_query, text, st_tfidf_vectorizer)

            if similarity > maximum_similarity:
                maximum_similarity = similarity
                tag = intent['intent']
                bot_response = random.choice(intent['responses'])

    if maximum_similarity > 0.7:
        if user_intent == IDENTITY_MANAGEMENT_LABEL:
            # Store the username
            if tag == "InitialUserName":
                print('Chatbot: ' + bot_response)
                if user_name != "":
                    name_query = input(user_name + ": ")
                else:
                    name_query = input("User: ")

                name = get_user_name(name_query, im_tfidf_vectorizer.get_feature_names_out())
                print('Chatbot: Great! Hi ' + name + ", nice to meet you! What can I do for you today?")

            # Change the username
            if tag == "ChangeUserName":
                name = get_user_name(user_query, im_tfidf_vectorizer.get_feature_names_out())
                bot_response = bot_response.replace("<HUMAN>", name)
                print("Chatbot: " + bot_response)

            # Explicit name output
            if tag == "UserNameQuery":
                if name == "":
                    print("Chatbot: I still don't know your name.")
                else:
                    bot_response = bot_response.replace("<HUMAN>", name)
                    print("Chatbot: " + bot_response)

        if user_intent == SMALL_TALK_LABEL:
            if tag == 'GoodBye':
                stop = True
                print("Chatbot: " + bot_response)

            elif tag == 'TimeQuery':
                bot_response = get_datetime_response(bot_response)
                print(bot_response)

            elif tag == 'ScoreQuery':
                bot_response = "Chatbot: " + bot_response + str(user_mini_game_point)
                print(bot_response)

            elif tag == 'RealNameQuery' or tag == 'NameQuery':
                bot_response = bot_response.replace("<CHATBOT>", bot_name)
                print("Chatbot: " + bot_response)

            else:
                print("Chatbot: " + bot_response)

    else:
        print("Chatbot: Sorry, I don't understand.")
        print(maximum_similarity)

    return [name, stop]


"""
Chatbot
"""
print("Setting up movie database ....")
# Build intent matching classifier and tfidf_vectorizer
genre_tfidf_vectorizer = build_genre_tfidf_vectorizer(movie_df['Genres'])
[summary_tfidf_vectorizer, summary_matrix] = build_tfidf_vectorizer_with_matrix(movie_df['Summary'])
[intent_corpus, intent_classifier, intent_tfidf_vectorizer] = build_intent_matching_classifier()
im_tfidf_vectorizer = build_tfidf_vectorizer(intent_corpus[IDENTITY_MANAGEMENT_LABEL])
st_tfidf_vectorizer = build_tfidf_vectorizer(intent_corpus[SMALL_TALK_LABEL])
question_list = question_answer_df.question.values.tolist()
[question_vectorizer, question_matrix] = build_tfidf_vectorizer_with_matrix(question_list)

# ---------------------------------
stop_list = ['Bye', 'Goodbye']
stop = False

user_name = ""
bot_name = "CineBot"
# Mini Game variables
user_mini_game_point = 0

# print(genre_tfidf_vectorizer.get_feature_names_out())

print("Chatbot: Hiya, my name is " + bot_name + ".")
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
    # The probability distribution of the class
    probability_values = intent_classifier.predict_proba(vectorized_user_query).tolist()[0]

    # Construct an intent probability distribution dictionary
    class_probability_dict = {class_keys[i]: probability_values[i] for i in range(len(class_keys))}

    print(class_probability_dict)
    print(user_intent)

    if user_query not in stop_list:
        # The confidence level of the chosen class
        intent_prediction_probability = class_probability_dict[user_intent[0]]
        print(intent_prediction_probability)

        # Only proceed with the intent if the classifier have confidence score on the class
        if intent_prediction_probability >= 0.8:
            if user_intent == RECOMMENDATION_LABEL:
                movie_recommendation(user_query, genre_tfidf_vectorizer)

            if user_intent == GAME_LABEL:
                user_mini_game_point = movie_guessing_game(user_mini_game_point)

            if user_intent == IDENTITY_MANAGEMENT_LABEL or user_intent == SMALL_TALK_LABEL:
                [user_name, stop] = small_talk_and_identity_management(user_name, user_intent[0], stop)

            if user_intent == QUESTION_ANSWER_LABEL:
                # Get a random response from the intent json file
                response = random.choice(intent_corpus[QUESTION_ANSWER_LABEL]['intents'][0]['responses'])
                question_query = input("Chatbot: " + response + "\nQuestion >>> ")

                # Contains the top 3 most similar questions' index and probability
                top_similarity_questions = rank_similar_documents(question_query, question_vectorizer, question_matrix)
                top_similarity_questions = [[item_list[0], item_list[1].item()] for item_list in
                                            top_similarity_questions]

                print(top_similarity_questions)
                top_similarity = top_similarity_questions[0][1]

                answers = question_answer_df.iloc[top_similarity_questions[0][0]]['answers']
                answer_list = ast.literal_eval(answers)
                print(top_similarity)

                # Provide the highest matched similarity answer correspond to the question
                if top_similarity >= 0.7:
                    print("Chatbot: The answer to this is " + random.choice(answer_list) + ".")

                # Suggest if the user is asking the specific question if low similarity is returned
                elif 0.7 > top_similarity > 0.5:
                    question = question_answer_df.iloc[top_similarity_questions[0][0]]['question']

                    if user_name != "":
                        user_re_prompt = input("Chatbot: Are you suggesting -> " + question + "\n" + user_name + ": ")
                    else:
                        user_re_prompt = input("Chatbot: Are you suggesting -> " + question + "\nUser: ")

                    # If the user agrees to the suggestion then provide the answer and append the new question and
                    # answer pair to the movie question answer dataset
                    if user_re_prompt.strip().lower() == 'yes':
                        print("Chatbot: The answer to this is " + random.choice(answer_list) + ".")

                        new_question_answer_pair = [len(question_answer_df.index), question_query,
                                                    question_answer_df.iloc[top_similarity_questions[0][0]]['answers']]

                        with open('./data/information_retrieval/movie_question_answer.csv', 'a') as qna_csv:
                            writer_instance = writer(qna_csv)
                            writer_instance.writerow(new_question_answer_pair)
                            qna_csv.close()
                    else:
                        print("Chatbot: Sorry, I don't know the answer to your question.")

                else:
                    print("Chatbot: Sorry, I don't know the answer to your question.")

        elif 0.8 > intent_prediction_probability > 0.6:
            print("Chatbot: Can you please reformulate your query?")

        else:
            print("Chatbot: Sorry, I don't understand.")

    else:
        print("Chatbot: Bye! See you soon.")
        stop = True

# TODO: Save the model using pickle

# Functionality
# TODO: Conversation fallback Disambiguation
