import random
from datetime import datetime
"""
Genres
"""


# Generate random style of genre response
def genre_response(row):
    response_title_section = [
        "Hiya, I found '" + row['Title'] + "', released on " + str(row['Year']) + ". ",
        "I suggest '" + row['Title'] + " (" + str(row['Year']) + "). ",
        "I recommend '" + row['Title'] + "', released on " + str(row['Year']) + ". "
    ]

    response_genre_section = [
        "It's a " + genres_parsing(row['Genres']) + " movie. ",
        "This movie is categorized as " + genres_parsing(row['Genres']) + ". "
    ]

    response_uri_section = [
        "If you are interested, here's the YouTube trailer link https://www.youtube.com/watch?v=" + row[
            'YouTube Trailer'] + ".",
        "Do give https://www.youtube.com/watch?v=" + row['YouTube Trailer'] + " a visit for its official trailer.",
        "Here's a little sneak peak https://www.youtube.com/watch?v=" + row['YouTube Trailer'] + " for its trailer."
    ]

    random_title = random.choice(response_title_section)
    random_genre = random.choice(response_genre_section)
    random_uri = random.choice(response_uri_section)

    return random_title + random_genre + random_uri


def similar_genre_response(similar_movies):
    response_header = [
        "I can't find the exact movie category you are looking for, but here is some similar movie "
        "you may be interested.\n",
        "There isn't a total match of what you requested, but I have got something similar.\n",
    ]

    response_body = []

    for index, movie in enumerate(similar_movies):
        response_body.append(str(index + 1) + ". " + movie[1]['Title'] + " (" + str(movie[1]['Year']) + "), a " +
                             genres_parsing(movie[1]['Genres']) + " movies. ")

    random_header = random.choice(response_header)

    return random_header + "\n".join(response_body)


def fallback_genre_response():
    response = [
        "Sorry I can't find any movie that match the requirement of yours TAT.",
        "Ouch, there isn't any movies requested in my database. ",
        "My apologies, currently we don't have any movies that match your requirement. "
    ]

    random_response = random.choice(response)

    return random_response


# Format the genres string etc. 'Action|Horror|Thriller' -> 'Action, Horror and Thriller'
def genres_parsing(genres):
    # Get the number of '|' in the genres
    count = genres.count('|')
    # Replace all '|' by ', ' except the last '|'
    if count != 1:
        genres = genres.replace('|', ', ', count - 1)

    # Replace the last '|' by ' and '
    genres = genres.replace('|', ' and ')

    return genres


"""
Games
"""


def display_movie_summary(title_dict, movie_summary, user_name):
    response = "For this mini game, I will scramble 2 similar movie plot with the movie and you have to guess it. \n\n" "Which movie below do you think correspond to this given summary -> " + movie_summary + "\n"

    if user_name != "":
        response = "Chatbot: Let's get started " + user_name + "! " + response
        print(response)
    else:
        print("Chatbot: " + response)

    for index, title in enumerate(title_dict.values()):
        print(str(index + 1) + ". " + title)


def correct_answer_response(game_point):
    response = [
        "Damn you're good! You now have " + str(game_point) + " point.",
        "We've got a cinephile! Your currently own " + str(game_point) + " point.",
        "You got it! You now owns " + str(game_point) + " point."
    ]

    random_response = random.choice(response)

    return random_response


def incorrect_answer_response(answer, game_point):
    response = [
        "Oops, the answer is '" + answer + "', your current point remains: " + str(game_point) + ".",
        "Not quite, it is '" + answer + "', your point remains: " + str(game_point) + ".",
        "Ouch, the correct answer for this is '" + answer + "', your point still remains: " + str(game_point) + "."
    ]

    random_response = random.choice(response)

    return random_response


"""
Date Time
"""


def get_datetime_response(response):
    present = datetime.now()
    day_of_week = datetime.today().strftime('%A')
    date_time = present.strftime("%d/%B/%Y %H:%M")

    return response + str(day_of_week) + ", " + str(date_time)
