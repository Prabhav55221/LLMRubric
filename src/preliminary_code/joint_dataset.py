import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class JointDataset(Dataset):
    def __init__(self, json_path):
        self.known_questions = []
        self.llm_results = []
        self.human_results = []
        self.judge_ids = []
        with open(json_path, "r") as file:
            data = json.load(file)
            for entry in tqdm(data, desc="Processing Data"):
                llm_entry = []
                self.known_questions.append(entry["known_questions"])
                for i in range(9):
                    if f"Q{i}" not in entry["known_questions"]: #for unknown questions, data is in the form 1, 0, 0, ...
                        llm_entry.append([1.0] + [0.0 for _ in range(len(entry["llm_answers"][0]))])
                    else:
                        llm_entry.append([0.0] + entry["llm_answers"][entry["known_questions"].index(f"Q{i}")])
                human_answer = [0.0, 0.0, 0.0, 0.0]
                human_answer[entry["final_answer"] - 1] = 1.0
                self.human_results.append(human_answer)
                self.judge_ids.append(entry["judge"])
                self.llm_results.append(llm_entry)
        self.llm_results = torch.tensor(self.llm_results)
        self.human_results = torch.tensor(self.human_results)


    def __len__(self):
        return len(self.human_results)

    def __getitem__(self, idx):
        return (self.llm_results[idx], self.human_results[idx])
if __name__ == "__main__":
    dataset = JointDataset("../data/joint_model_data/out_domain_test.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for input, label in dataloader:
        print(input)
        print(label)