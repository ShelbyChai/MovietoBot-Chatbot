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

    post = pos_tag(user_name, tagset="universal")
    # Only conserve text which its p-o-s tag is 'NOUN'
    user_name = ' '.join([tup[0] for tup in post if tup[1] == 'NOUN'])

    return user_name.title()
