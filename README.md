# Extracting Prompts from Customized Large Language Models

This paper consists of the source code of paper: `Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models`.

You can also find and discuss this paper on HuggingFace: https://huggingface.co/papers/2408.02416

The PEAD dataset used in our paper can be also be found/downloaded/loaded via huggingface's `datasets` at: [liangzid/PEAD](https://huggingface.co/datasets/liangzid/PEAD)


## Source code explanations

+ PEAD dataset: [extractingPrompt/instructions/benchmark_collections/OVERALL_DATA_BENCHMARK.json]
+ Source code of all experiments: [extractingPrompt/]
  + Generalized evaluation
	+ Vanilla: [extractingPrompt/1.run_prompt_extraction.py]
	+ Function callings comparison: [extractingPrompt/5.funcall_comparison.py]
  + Scaling laws of prompt extraction
	+ Model size: [extractingPrompt/2.model_size_prompt_extraction_experiments.py]
	+ Sequence length: [extractingPrompt/4.varying_sequence_length.py]
  + Empirical analysis
	+ Convincing Premise: [extractingPrompt/6.ppl_comparison.py]
	+ Parallel-translation: [extractingPrompt/7.attention_visualize.py]
	+ Parallel-translation: [extractingPrompt/attention_visualize.py]
  + Defense strategies
    + Defending methods: [extractingPrompt/defending/ppl_high2_confusingBeginnings.py]
	+ Performance drops experiments of the defending: [extractingPrompt/defending/2.drops_of_defending.py]
	+ visualization: [extractingPrompt/defending/defense_visualization.py]
  + Close-AI experiments
	+ vanilla prompt extraction: [extractingPrompt/api_related_experiments/1.run_prompt_extraction.py]
	+ soft extraction: [extractingPrompt/api_related_experiments/2.soft_extraction_experiments.py]
	+ performance drops of defending: [extractingPrompt/api_related_experiments/3.1.drops_of_defense.py]

## Experimental environments

Run 
```sh
pip install -r re.txt
```

or install the following key packages manually:

```sh
datasets
numpy
pandas
peft
safetensors
scipy
tensorboard
tensorboardX
tiktoken
tokenizers
torch
tqdm
transformers
matplotlib
scikit-learn
thefuzz
einops
sentencepiece
```

