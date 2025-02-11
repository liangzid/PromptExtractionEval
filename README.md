# Extracting Prompts from Customized Large Language Models

This paper consists of the source code of paper: `Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models`([arxiv](https://arxiv.org/abs/2408.02416)).




## Source code explanations

+ PEAD dataset: [extractingPrompt/instructions/benchmark_collections/OVERALL_DATA_BENCHMARK.json](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/instructions/benchmark_collections/OVERALL_DATA_BENCHMARK.json)
+ Source code of all experiments: [extractingPrompt/](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/)
  + Generalized evaluation
	+ Vanilla: [extractingPrompt/1.run_prompt_extraction.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/1.run_prompt_extraction.py)
	+ Function callings comparison: [extractingPrompt/5.funcall_comparison.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/5.funcall_comparison.py)
  + Scaling laws of prompt extraction
	+ Model size: [extractingPrompt/2.model_size_prompt_extraction_experiments.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/2.model_size_prompt_extraction_experiments.py)
	+ Sequence length: [extractingPrompt/4.varying_sequence_length.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/4.varying_sequence_length.py)
  + Empirical analysis
	+ Convincing Premise: [extractingPrompt/6.ppl_comparison.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/6.ppl_comparison.py)
	+ Parallel-translation: [extractingPrompt/7.attention_visualize.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/7.attention_visualize.py)
	+ Parallel-translation: [extractingPrompt/attention_visualize.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/attention_visualize.py)
  + Defense strategies
    + Defending methods: [extractingPrompt/defending/ppl_high2_confusingBeginnings.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/defending/ppl_high2_confusingBeginnings.py)
	+ Performance drops experiments of the defending: [extractingPrompt/defending/2.drops_of_defending.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/defending/2.drops_of_defending.py)
	+ visualization: [extractingPrompt/defending/defense_visualization.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/defending/defense_visualization.py)
  + Close-AI experiments
	+ vanilla prompt extraction: [extractingPrompt/api_related_experiments/1.run_prompt_extraction.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/api_related_experiments/1.run_prompt_extraction.py)
	+ soft extraction: [extractingPrompt/api_related_experiments/2.soft_extraction_experiments.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/api_related_experiments/2.soft_extraction_experiments.py)
	+ performance drops of defending: [extractingPrompt/api_related_experiments/3.1.drops_of_defense.py](https://github.com/liangzid/attackFineTunedModels/blob/main/extractingPrompt/api_related_experiments/3.1.drops_of_defense.py)

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

## Contact the authors

Feel free to open an issue, or send the email to `zi1415926.liang@connect.polyu.hk` if there exists any problem.

Citation:

```bibtex
@misc{liang2024promptsleakedunravelingprompt,
      title={Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models}, 
      author={Zi Liang and Haibo Hu and Qingqing Ye and Yaxin Xiao and Haoyang Li},
      year={2024},
      eprint={2408.02416},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.02416}, 
}
```
