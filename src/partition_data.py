import pandas as pd
import json
df = pd.read_csv("../data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv")
dict = {}
for i in range(len(df)):
    raw_data = df.iloc[i][0]
    temp_list = raw_data.split("\t")
    text_id = temp_list[0]
    question_num = temp_list[1]
    prob_distribution = [float(data) for data in temp_list[3:7]]
    if text_id not in dict.keys():
        dict[text_id] = {}
    dict[text_id][question_num] = prob_distribution
with open("../data/gpt-3.5-turbo-data.json", "w") as json_file:
    json.dump(dict, json_file, indent=4)
df = pd.read_csv("../data/real_convs_random_baseline.tsv")
dict = {}
for i in range(len(df)):
    raw_data = df.iloc[i][0]
    temp_list = raw_data.split("\t")
    text_id = temp_list[0]
    judge_id = temp_list[1]
    question_num = temp_list[-2]
    answer = int(float(temp_list[-1]))
    if text_id not in dict.keys():
        dict[text_id] = {}
    dict[text_id][question_num] = {}
    if "judge" in dict[text_id][question_num].keys():
        print("True")
    dict[text_id][question_num]["judge"] = judge_id
    dict[text_id][question_num]["distribution"] = [0.0, 0.0, 0.0, 0.0]
    dict[text_id][question_num]["distribution"][answer-1] = 1.0

with open("../data/human-data.json", "w") as json_file:
    json.dump(dict, json_file, indent=4)