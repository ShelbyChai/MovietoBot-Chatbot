import pandas as pd
import random

movie_df = pd.read_csv(r"../data/information_retrieval/movie_dataset.csv")
question_answer_df = pd.read_csv(r"../data/information_retrieval/movie_question_answer.csv")

random.seed(42)

# Evaluation index for question and answering system
randomized_mq_index_list = []
for index in range(8):
    randomized_mq_index_list.append(random.randint(0, len(question_answer_df.index)))

print("Movie qna index: " + str(randomized_mq_index_list))

# Evaluation index for movie recommendation system
randomized_mr_index_list = []
for index in range(4):
    randomized_mr_index_list.append(random.randint(0, len(movie_df.index)))

print("Movie recommendation index: " + str(randomized_mr_index_list))
