import copy
import random
import json
def main(num_partition):
    out_domain_size = int(num_partition / 5)
    question_list = ["Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    with open("../data/original_data_llm_human_answers/gpt-3.5-turbo-data.json", "r") as file:
        llm_data = json.load(file)
    with open("../data/original_data_llm_human_answers/human-data.json", "r") as file:
        human_data = json.load(file)
    data_list = []
    combinations = []
    count = 0
    while count < num_partition - out_domain_size:
        if random.random() < 0.8:
            num_known = random.randint(1, 4)
        else:
            num_known = random.randint(5, 7)
        temp_list = copy.deepcopy(question_list)
        random.shuffle(temp_list)
        temp_list.remove("Q0")
        temp_list.append("Q0") # Make sure q0 is the last one
        known_questions = temp_list[:num_known]
        unknown_questions = temp_list[num_known:]
        if random.random() < 0.25: # 1/4 probability of knowing LLM's answer to Q0
            unknown_questions.remove("Q0")
            known_questions.append("Q0")
        if set(known_questions) in combinations:
            continue
        combinations.append(set(known_questions))
        for text_id, questions in human_data.items():
            try:
                dict = {"known_questions": known_questions, "llm_answers": [], "final_answer": "", "judge": "", "text_id": text_id}
                dict["final_answer"] = questions["Q0"]["distribution"].index(1.0) + 1
                dict["judge"] = questions["Q0"]["judge"]
                info = llm_data[text_id]
                for question in known_questions:
                    dict["llm_answers"].append(info[question])
                data_list.append(dict)
            except Exception as e:
                continue
        count += 1
    random.shuffle(data_list)
    in_domain_size = num_partition - out_domain_size
    in_domain_train_size = int(in_domain_size * 0.8) * 223
    in_domain_dev_size = int(in_domain_size * 0.1) * 223
    with open("../data/in_domain_train.json", "w") as file:
        json.dump(data_list[:in_domain_train_size], file, indent=4)
    with open("../data/in_domain_dev.json", "w") as file:
        json.dump(data_list[in_domain_train_size: in_domain_train_size + in_domain_dev_size], file, indent=4)
    with open("../data/in_domain_test.json", "w") as file:
        json.dump(data_list[in_domain_train_size + in_domain_dev_size:], file, indent=4)
    data_list = []
    count = 0
    out_domain_test_size = int(out_domain_size / 4)
    while count < out_domain_test_size:
        if random.random() < 0.8:
            num_known = random.randint(1, 4)
        else:
            num_known = random.randint(5, 7)
        temp_list = copy.deepcopy(question_list)
        random.shuffle(temp_list)
        temp_list.remove("Q0")
        temp_list.append("Q0") # Make sure q0 is the last one
        known_questions = temp_list[:num_known]
        unknown_questions = temp_list[num_known:]
        if random.random() < 0.25: # 1/4 probability of knowing LLM's answer to Q0
            unknown_questions.remove("Q0")
            known_questions.append("Q0")
        if set(known_questions) in combinations:
            continue
        combinations.append(set(known_questions))
        for text_id, questions in human_data.items():
            try:
                dict = {"known_questions": known_questions, "llm_answers": [], "final_answer": "", "judge": "", "text_id": text_id}
                dict["final_answer"] = questions["Q0"]["distribution"].index(1.0) + 1
                dict["judge"] = questions["Q0"]["judge"]
                info = llm_data[text_id]
                for question in known_questions:
                    dict["llm_answers"].append(info[question])
                data_list.append(dict)
            except Exception as e:
                continue
        count += 1
    with open("../data/out_domain_test.json", "w") as file:
        json.dump(data_list, file, indent=4)
    data_list = []
    count = 0
    while count < out_domain_size - out_domain_test_size:
        if random.random() < 0.8:
            num_known = random.randint(1, 4)
        else:
            num_known = random.randint(5, 7)
        temp_list = copy.deepcopy(question_list)
        random.shuffle(temp_list)
        temp_list.remove("Q0")
        temp_list.append("Q0") # Make sure q0 is the last one
        known_questions = temp_list[:num_known]
        unknown_questions = temp_list[num_known:]
        if random.random() < 0.25: # 1/4 probability of knowing LLM's answer to Q0
            unknown_questions.remove("Q0")
            known_questions.append("Q0")
        if set(known_questions) in combinations:
            continue
        combinations.append(set(known_questions))
        for text_id, questions in human_data.items():
            try:
                dict = {"known_questions": known_questions, "llm_answers": [], "final_answer": "", "judge": "", "text_id": text_id}
                dict["final_answer"] = questions["Q0"]["distribution"].index(1.0) + 1
                dict["judge"] = questions["Q0"]["judge"]
                info = llm_data[text_id]
                for question in known_questions:
                    dict["llm_answers"].append(info[question])
                data_list.append(dict)
            except Exception as e:
                continue
        count += 1
    with open("../data/out_domain_dev.json", "w") as file:
        json.dump(data_list, file, indent=4)




if __name__ == "__main__":
    main(200)
