import json
with open("../data/out_domain.json", "r") as file:
    out_data = json.load(file)

with open("../data/in_domain.json", "r") as file:
    in_data = json.load(file)
combinations = []
for entry in in_data:
    if set(entry["known_questions"]) not in combinations:
        combinations.append(set(entry["known_questions"]))
combinations1 = []
for entry in out_data:
    if set(entry["known_questions"]) not in combinations1:
        combinations1.append(set(entry["known_questions"]))
    if set(entry["known_questions"]) in combinations:
        print("True")
print(combinations1)
print(combinations)