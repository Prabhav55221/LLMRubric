# LLMRubric

> [!NOTE]  
> Documentation is still under construction. There might be things I have forgotten to add - Please create an issue if there are problems in replicating or running the package!

## Introduction

The base implementation is based on a[^1]. The original LLMRubric paper presents a framework for the automated evaluation of natural language texts. A manually constructed rubric describes how to assess multiple dimensions of interest. To evaluate a text, a large language model (LLM) is prompted with each rubric question and produces a distribution over potential responses. Responses are calibrated by training a small feed-forward neural network that includes both judge-specific and judge-independent parameters.

This repository attempts an extension to the work by introducing the suggestions for future works in the paper:
- Generalizing the LLMRubric method in a package that can be extended to any form of calibrated LLM-based evaluation.
- Extends only Option based evaluation to ranking, rating and text based evaluation.
- Option to construct a meta-rubric using varying parameters for each question. Expands dimensionality of rubric to $D >>> d$.
- Adaptivity: Attempts to use active learning to dynamically choose which question provides the most benefit to the LLM. This reduces complexity and computational cost of evaluation without sacrificing performance.

## Basic Setup

To use the package, first clone the GitHub Repo and cd into the folder:

```cmd
git clone https://github.com/argolab/LLMRubric.git
cd LLMRubric
```

We provide an env.yaml for setting up the conda environment required to run the codebase.

```cmd
conda create -f env.yaml
conda activate llm_rubric_env
```

> [!WARNING]  
> The requirements are not optimized for Apple Silicon based architetures: Testing was done on a Linux Distro with GPU and without GPU. You might need to modify torch installation for Silicon architecture!

After installation, confirm you have the below folders created for sanity and logging:

```cmd
mkdir outputs
mkdir models
```

We use OpenAI as of now to allow for LLM Calling. **You will need to setup a Paid API Key for the module to work!**. We recommend setting up the same as a .env file in the LLMRubric folder:

```cmd
echo OPENAI_API_KEY:<API_KEY> > .env
```

## Folder Structure

Before running the module, be sure you understand the below structure of the codebase - This will allow you to change configs and datasets easily!

```cmd
├── README.md
├── data
│   ├── hanna.yaml [Your dataset YAML]
│   ├── hanna_stories_annotations_updated.csv [Your dataset CSV]
├── env.yaml
├── main.py
├── models [Calibration Models will be stored here]
├── outputs [Logs and Metrics]
├── prompts
│   ├── rubric.yaml [Your rubric]
│   └── system_prompt.yaml [Your system prompt]
└── src
    ├── CODEBASE
```

## Inputs

We will define some of the inputs you need to setup here.

### Dataset

The module can support any text based input for evalutaion. Ideally, we require a CSV with the *at least* the following columns (You can have as many evaluation dimensions as you want - **as long as you keep one overall dimension**):

> [!WARNING]  
> If a Q0 or overall evaluation column does not exist, the codebase defaults to mean of all possible dimensions.

```csv
Eval_Dim1 | Eval_Dim2 | Eval_Dim3 | Eval_Dim4 | Overall Satisfaction | Judge_IDS| TEXT
```

- Eval_Dims: Can be named anything as long as they are FLOAT quantities.
- Overall Satisfaction: Will default to mean if not present.
- Judge_IDS: This is necessary to calibrate to human evaluation. If you have just 1 annotator, fill in a single ID throughout.
- TEXT: What needs to be evaluated. Can be a conversation, long text, short text etc!

> [!Note]  
> Text evalutation is TODO!

##### Example Dataset

We modify the HANNA-BENCHMARK[^2] to create a sample dataset. A sample row is shown below:

| Relevance | Coherence | Empathy | Surprise | Engagement | Complexity | Overall Satisfaction | Judge ID           | Text |
|-----------|----------|---------|----------|------------|------------|----------------------|--------------------|------|
| 4         | 4        | 3       | 2        | 4          | 4          | 3.5                  | A2VE5IV9OD2SK1     | When you ... I’ve tried. |

##### Setting up dataset.yaml

```yaml
NAME: Unique_Identifer
CSV_PATH: Path to your CSV
SYSTEM_PATH: Path to your system prompt (See next Subsection)
DIMENSIONS:
  Q1: Relevance
  Q2: Coherence
  # Define Dimensions with Convention of Q1 - QN for each. Q0 will be overall evaluation.
JUDGE_IDS:
  1: A2VE5IV9OD2SK1
  2: A1IZ4NX41GKU4X
  # Optional: In case you want to calibrate for specific judges! 
```

### Rubric

Next, you will have to setup the Rubric. This is located in the `/prompts` folder. You will have to setup 2 files!

`system_prompt.yaml`
```yaml
system_prompt:
  name: Unique_Identifier
  rubric_path: Path to rubric.yaml # Described next
  prompt: "You are given a story created written by an author based on a prompt. Your primary job is to evaluate the story based on a criteria. To do so, read the prompt and the story, and then answer the followed question, by selecting one of the choices.\n\n\nPrompt and Story: {TEXT}\n\nQuestion: **{QUESTION}**\n\nOptions:\n{OPTIONS}\n\nOnly print the number corresponding to your answer choice."

  # Note the usage of {TEXT} - Comes from your CSV. {QUESTION}, {OPTIONS} - These come from the rubric.yaml!
```

`rubric.yaml`
```yaml
rubric:
  Q1:
    # Match question ID to the Dimensions!
    question_id: "Q1"
    category: "Relevance"
    # {QUESTION}
    prompt: "Only in terms of relevance, how relevant is the story to the given prompt? Disregard whether they are grounded in the search results or any other parameters to judge the story."
    # {OPTIONS}
    options:
      1: "Not Relevant"
      2: "Slightly Relevant"
      3: "Moderately Relevant"
      4: "Mostly Relevant"
      5: "Highly Relevant"
    # Not yet implemented
    scale_type: "ordinal"
    response_type: "multiple_choice"
```

## Configs

Finally, set up (if needed), configs in `src/config.py`. The file should be self-explanatory!

## Running It!

Simple command line argument:

```cmd
python main.py --dataset <path_to_dataset_yaml> --output <path_to_system_prompt>
```

## Outputs

Under construction, but expect the following:

`outputs/cache` - Stores your cache for LLM API Calls.

> [!WARNING]  
> While the cache is designed to never be used if configs are changed, be sure to delete the cache if you internally change the codebase in any way. The default TTS is 7 days!

`outputs/calibration_results` - RMSE and Correlation Scores

`outputs/llm_results` - Probabilites over options

`outputs/evaluation_TIMESTAMP.log` - Logs!

## Additional Details

TBD

## References

[^1]: ```
    Hashemi et al.,
    title = "{LLM}-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts",
    author = "Hashemi, Helia  and
      Eisner, Jason et al.,
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.745/",
    doi = "10.18653/v1/2024.acl-long.745"

[^2]: ```
    Cyril et al.,
    doi = {10.1162/tacl_a_00689},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00689/2470807/tacl\_a\_00689.pdf},
    issn = {2307-387X},
    journal = {Transactions of the Association for Computational Linguistics},
    pages = {1122--1142},
    publisher = {MIT Press},
    title = {Do Language Models Enjoy Their Own Stories? {P}rompting Large Language Models for Automatic Story Evaluation},
    url = {https://doi.org/10.1162/tacl\_a\_00689},
    volume = {12},
    year = {2024}