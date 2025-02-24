import pandas as pd
import json
df = pd.read_csv("../gpt-3.5-turbo-16k_synth_evaluations_FIXED.tsv")
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
df = pd.read_csv("../human_judges_synth_all_FIXED_ANON.tsv")
dict = {}
for i in range(len(df)):
    raw_data = df.iloc[i][0]
    temp_list = raw_data.split("\t")
    text_id = temp_list[1]
    judge_id = temp_list[-1]
    question_num = temp_list[-2]
    if text_id not in dict.keys():
        dict[text_id] = {}
    dict[text_id][judge_id] = {}
    for i in range(5, 14):
        if i == 13:
            dict[text_id][judge_id]["Q0"] = int(float(temp_list[i]))
        dict[text_id][judge_id][f"Q{i-5+1}"] = int(float(temp_list[i]))

with open("../data/human-data.json", "w") as json_file:
    json.dump(dict, json_file, indent=4)