import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class JointDataset(Dataset):
    def __init__(self, json_path):
        self.known_questions = []
        self.inputs = []
        self.answers = []
        self.judge_ids = []
        with open(json_path, "r") as file:
            data = json.load(file)
            for entry in tqdm(data, desc="Processing Data"):
                self.known_questions.append(entry["known_questions"])
                self.inputs.append(entry["input"])
                self.answers.append(entry["answers"])
                self.judge_ids.append(entry["annotators"])
        for input in self.inputs:
            for i in input:
                if not len(i) == 5:
                    print(input.index(i))
        self.known_questions = torch.tensor(self.known_questions)
        self.inputs = torch.tensor(self.inputs)
        self.answers = torch.tensor(self.answers)
        self.judge_ids = torch.tensor(self.judge_ids)


    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return (self.known_questions[idx], self.inputs[idx], self.answers[idx], self.judge_ids[idx])
if __name__ == "__main__":
    dataset = JointDataset("../data/out_domain_test.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for index, batch in enumerate(dataloader):
        print(batch)