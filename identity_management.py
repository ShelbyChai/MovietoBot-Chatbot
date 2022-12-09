from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_user_name(user_input, vocabulary):
    vocabulary = vocabulary.tolist()
    common_word = ["name", "people", "please", "call", "me", "my", "change"]
    # Extend the common word with the query vocabulary
    common_word.extend(vocabulary)
    common_word.remove("adam")
    common_word.remove("bella")

    # print(common_word)

    tokens = word_tokenize(user_input)

    # Remove text that is present in the COMMON_WORD , stopword list and not alphabetic letters
    user_name = [token for token in tokens
                 if not token.lower() in common_word and not token.lower() in stopwords.words(
            'english') and token.isalpha()]

    user_name = " ".join(user_name)

    return user_name.title()
