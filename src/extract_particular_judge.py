import json
with open("../data/joint_model_data/in_domain_train.json", "r") as file:
    data = json.load(file)
judge_list = list(range(2, 23))
for judge in judge_list:
    list = []
    for entry in data:
        if entry["judge"] == str(judge):
            list.append(entry)
    with open(f"../data/joint_model_data/in_domain_train{judge}.json", "w") as file:
        json.dump(list, file, indent=4)
with open("../data/joint_model_data/in_domain_dev.json", "r") as file:
    data = json.load(file)
for judge in judge_list:
    list = []
    for entry in data:
        if entry["judge"] == str(judge):
            list.append(entry)
    with open(f"../data/joint_model_data/in_domain_dev{judge}.json", "w") as file:
        json.dump(list, file, indent=4)