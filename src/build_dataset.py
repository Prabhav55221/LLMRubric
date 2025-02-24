import copy
import random
import json
def main(num_partition):
    out_domain_size = int(num_partition / 5)
    question_list = ["Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    with open("../data/gpt-3.5-turbo-data.json", "r") as file:
        llm_data = json.load(file)
    with open("../data/human-data.json", "r") as file:
        human_data = json.load(file)
    data_list = []
    combinations = []
    count = 0
    while count < num_partition - out_domain_size:
        if random.random() < 0.8:
            num_known = random.randint(1, 4)
        else:
            num_known = random.randint(5, 8)
        temp_list = copy.deepcopy(question_list)
        random.shuffle(temp_list)
        known_questions = temp_list[:num_known]
        if set(known_questions) in combinations:
            continue
        combinations.append(set(known_questions))
        for text_id, info in human_data.items():
            try:
                entry = {"known_questions": [], "input": [], "answers": [], "annotators": []}
                annotators = info.keys()
                for question in question_list:
                    if question in known_questions:
                        entry["known_questions"].append(1)
                        entry["input"].append([0.0] + llm_data[text_id][question])
                        entry["annotators"].append(-1)
                        entry["answers"].append(llm_data[text_id][question])
                    else:
                        entry["known_questions"].append(0)
                        entry["input"].append([1.0, 0.0, 0.0, 0.0, 0.0])
                        entry["annotators"].append(-1)
                        entry["answers"].append(llm_data[text_id][question])
                for question in question_list:
                    temp = random.random()
                    annotator = random.choice(list(annotators))
                    if (question in known_questions and temp > 0.5) or (question not in known_questions and temp < 0.5):
                        entry["known_questions"].append(1)
                        input = [0.0, 0.0, 0.0, 0.0, 0.0]
                        if human_data[text_id][annotator][question] == 0:
                            index = 1
                        else:
                            index = human_data[text_id][annotator][question]
                        input[index] = 1.0
                        entry["input"].append(input)
                        entry["annotators"].append(int(annotator))
                        entry["answers"].append(input[1:])
                    else:
                        entry["known_questions"].append(0)
                        input = [1.0, 0.0, 0.0, 0.0, 0.0]
                        answer = [0.0, 0.0, 0.0, 0.0, 0.0]
                        if human_data[text_id][annotator][question] == 0:
                            index = 1
                        else:
                            index = human_data[text_id][annotator][question]
                        answer[index] = 1.0
                        entry["input"].append(input)
                        entry["annotators"].append(int(annotator))
                        entry["answers"].append(answer[1:])
                data_list.append(entry)
            except Exception as e:
                continue
        count += 1
    random.shuffle(data_list)
    in_domain_size = num_partition - out_domain_size
    in_domain_train_size = int(in_domain_size * 0.8) * 225
    in_domain_dev_size = int(in_domain_size * 0.1) * 225
    with open("../data/in_domain_train.json", "w") as file:
        json.dump(data_list[:in_domain_train_size], file)
    with open("../data/in_domain_dev.json", "w") as file:
        json.dump(data_list[in_domain_train_size: in_domain_train_size + in_domain_dev_size], file)
    with open("../data/in_domain_test.json", "w") as file:
        json.dump(data_list[in_domain_train_size + in_domain_dev_size:], file)
    data_list = []
    count = 0
    out_domain_test_size = int(out_domain_size / 4)
    while count < out_domain_test_size:
        if random.random() < 0.8:
            num_known = random.randint(1, 4)
        else:
            num_known = random.randint(5, 8)
        temp_list = copy.deepcopy(question_list)
        random.shuffle(temp_list)
        known_questions = temp_list[:num_known]
        if set(known_questions) in combinations:
            continue
        combinations.append(set(known_questions))
        for text_id, info in human_data.items():
            try:
                entry = {"known_questions": [], "input": [], "answers": [], "annotators": []}
                annotators = info.keys()
                for question in question_list:
                    if question in known_questions:
                        entry["known_questions"].append(1)
                        entry["input"].append([0.0] + llm_data[text_id][question])
                        entry["annotators"].append(-1)
                        entry["answers"].append(llm_data[text_id][question])
                    else:
                        entry["known_questions"].append(0)
                        entry["input"].append([1.0, 0.0, 0.0, 0.0, 0.0])
                        entry["annotators"].append(-1)
                        entry["answers"].append(llm_data[text_id][question])
                for question in question_list:
                    temp = random.random()
                    annotator = random.choice(list(annotators))
                    if (question in known_questions and temp > 0.5) or (question not in known_questions and temp < 0.5):
                        entry["known_questions"].append(1)
                        input = [0.0, 0.0, 0.0, 0.0, 0.0]
                        if human_data[text_id][annotator][question] == 0:
                            index = 1
                        else:
                            index = human_data[text_id][annotator][question]
                        input[index] = 1.0
                        entry["input"].append(input)
                        entry["annotators"].append(int(annotator))
                        entry["answers"].append(input[1:])
                    else:
                        entry["known_questions"].append(0)
                        input = [1.0, 0.0, 0.0, 0.0, 0.0]
                        answer = [0.0, 0.0, 0.0, 0.0, 0.0]
                        if human_data[text_id][annotator][question] == 0:
                            index = 1
                        else:
                            index = human_data[text_id][annotator][question]
                        answer[index] = 1.0
                        entry["input"].append(input)
                        entry["annotators"].append(int(annotator))
                        entry["answers"].append(answer[1:])
                data_list.append(entry)
            except Exception as e:
                continue
        count += 1
    with open("../data/out_domain_test.json", "w") as file:
        json.dump(data_list, file)
    data_list = []
    count = 0
    while count < int(out_domain_size * 0.75):
        if random.random() < 0.8:
            num_known = random.randint(1, 4)
        else:
            num_known = random.randint(5, 8)
        temp_list = copy.deepcopy(question_list)
        random.shuffle(temp_list)
        known_questions = temp_list[:num_known]
        if set(known_questions) in combinations:
            continue
        combinations.append(set(known_questions))
        for text_id, info in human_data.items():
            try:
                entry = {"known_questions": [], "input": [], "answers": [], "annotators": []}
                annotators = info.keys()
                for question in question_list:
                    if question in known_questions:
                        entry["known_questions"].append(1)
                        entry["input"].append([0.0] + llm_data[text_id][question])
                        entry["annotators"].append(-1)
                        entry["answers"].append(llm_data[text_id][question])
                    else:
                        entry["known_questions"].append(0)
                        entry["input"].append([1.0, 0.0, 0.0, 0.0, 0.0])
                        entry["annotators"].append(-1)
                        entry["answers"].append(llm_data[text_id][question])
                for question in question_list:
                    temp = random.random()
                    annotator = random.choice(list(annotators))
                    if (question in known_questions and temp > 0.5) or (question not in known_questions and temp < 0.5):
                        entry["known_questions"].append(1)
                        input = [0.0, 0.0, 0.0, 0.0, 0.0]
                        if human_data[text_id][annotator][question] == 0:
                            index = 1
                        else:
                            index = human_data[text_id][annotator][question]
                        input[index] = 1.0
                        entry["input"].append(input)
                        entry["annotators"].append(int(annotator))
                        entry["answers"].append(input[1:])
                    else:
                        entry["known_questions"].append(0)
                        input = [1.0, 0.0, 0.0, 0.0, 0.0]
                        answer = [0.0, 0.0, 0.0, 0.0, 0.0]
                        if human_data[text_id][annotator][question] == 0:
                            index = 1
                        else:
                            index = human_data[text_id][annotator][question]
                        answer[index] = 1.0
                        entry["input"].append(input)
                        entry["annotators"].append(int(annotator))
                        entry["answers"].append(answer[1:])
                data_list.append(entry)
            except Exception as e:
                continue
        count += 1
    with open("../data/out_domain_dev.json", "w") as file:
        json.dump(data_list, file)

if __name__ == "__main__":
    main(200)