# MuscleLoRA

<div align="center">
  <h2 align="center">Acquiring Clean Language Models from Backdoor Poisoned Datasets</h2>
  <a href="https://arxiv.org/abs/2402.12026" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.12026-b31b1b.svg?style=flat">
  </a>
</div>

**This repository is the code implementation of our [paper](https://arxiv.org/abs/2402.12026):**
```
Acquiring Clean Language Models from Backdoor Poisoned Datasets
```

## Dependencies

* Install requirements.
  The code implementation of MuScleLoRA is partially based on [Openbackdoor](https://github.com/thunlp/OpenBackdoor). After cloning this repository, you can install the requirements by:

```bash
    pip3 install -r requirements.txt
```

Notably, if the installation of ``opendelta`` fails with pip, install ``opendelta`` from [github](https://github.com/thunlp/OpenDelta). Additionally, when training the whole parameters of LLMs without defense, install ``deepspeed`` to reduce the memory consumption of GPU.

* Training Data. We provide the backdoored training data in [./poison_data](./poison_data/).
* Weights of LM. To conduct [StyleBkd](https://doi.org/10.18653/v1/2021.emnlp-main.374), the ``lievan/[style]`` version of GPT-2 is required. You can download the weights from [huggingface](https://huggingface.co/lievan).

## Reproduce the results

### Reproduce the results of LLM

To reproduce the results of LLM, configure  `--config_path` and run `python llmDefense.py`.

Or simply run

```bash
bash llm.sh \
    [dataset:sst-2/hsol/lingspam/agnews/miniagnews] \
    [modelname:llama/gpt] \
    [way:vanilla/mslr/lora/ga+lora/ga+lora+mslr/prefix] \
    [start:0-3] \
    [end:1-4] \
    [poison_rate:0-1] \
    [notation]
```

to reproduce the defense results of Llama2-7B and GPT2-XL, where vanilla denotes no defense deployment, ga denotes gradient alignment, mslr denotes multiple radial scalings, lora denotes low-rank adaptation ([LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)), prefix denotes [Prefix-Tuning](https://doi.org/10.18653/v1/2021.acl-long.353). Additionally, the parameter ``start`` and ``end`` control the number of attack methods, where 0 denotes [Badnets](https://doi.org/10.18653/v1/2020.acl-main.249), 1 denotes [Addsent](https://doi.org/10.1109/ACCESS.2019.2941376), 2 denotes [StyleBkd](https://doi.org/10.18653/v1/2021.emnlp-main.374), and 3 denotes [HiddenKiller](https://doi.org/10.18653/v1/2021.acl-long.37).

### Reproduce the results of PLM

To reproduce the results of PLM, configure  `--config_path` and run `python plmDefense.py`.

Or simply run

```bash
bash plm.sh \
    [dataset:sst-2/hsol/lingspam/agnews/miniagnews] \
    [modelname:bert-large/roberta-large/bert/roberta] \
    [way:vanilla/ga/mslr/lora/ga+lora/ga+lora+mslr/adapter/prefix] \
    [start:0-3] \
    [end:1-4] \
    [poison_rate:0-1] \
    [notation]
```

to reproduce the defense results of BERT and RoBERTa, where vanilla denotes no defense deployment, ga denotes gradient alignment, mslr denotes multiple radial scalings, lora denotes low-rank adaptation ([LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)), prefix denotes [Prefix-Tuning](https://doi.org/10.18653/v1/2021.acl-long.353), adapter denotes [Adapter](https://proceedings.mlr.press/v97/houlsby19a.html). Additionally, the parameter ``start`` and ``end`` control the number of attack methods, where 0 denotes [Badnets](https://doi.org/10.18653/v1/2020.acl-main.249), 1 denotes [Addsent](https://doi.org/10.1109/ACCESS.2019.2941376), 2 denotes [StyleBkd](https://doi.org/10.18653/v1/2021.emnlp-main.374), and 3 denotes [HiddenKiller](https://doi.org/10.18653/v1/2021.acl-long.37).



### Reproduce the defense results of end-to-end baselines

To reproduce the results of end-to-end baselines, configure  `--config_path` and run `python e2ebaselineDefense.py`.

Or simply run

```bash
bash e2ebaseline.sh \
    [dataset:sst-2/hsol/lingspam/agnews/miniagnews] \
    [modelname:bert/roberta/bert-large/roberta-large/llama] \
    [defender:onion/bki/cube/strip/rap/onionllm/stripllm] \
    [start:0-3] \
    [end:1-4]
```

to reproduce the defense results of end-to-end baselines, Additionally, the parameter ``start`` and ``end`` control the number of attack methods, where 0 denotes [Badnets](https://doi.org/10.18653/v1/2020.acl-main.249), 1 denotes [Addsent](https://doi.org/10.1109/ACCESS.2019.2941376), 2 denotes [StyleBkd](https://doi.org/10.18653/v1/2021.emnlp-main.374), and 3 denotes [HiddenKiller](https://doi.org/10.18653/v1/2021.acl-long.37). 

Notably, for post-training baselines, i.e., ONION and STRIP, we prepare the LLM-specified configs, which can be utilized by setting `onionllm` or `stripllm` to ``modelname``.

### Reproduce the results of Fourier analyses

To reproduce the results of Fourier analyses, configure  `--config_path` and run `python fourierAnalysis.py`.

Or simply run

```bash
bash fourierAnalysis.sh \
    [dataset:sst-2/hsol/lingspam/agnews/miniagnews] \
    [modelname:bert/roberta/bert-large/roberta-large/llama] \
    [way:vanilla/mslr/lora/ga+lora/ga+lora+mslr] \
    [start:0-3] \
    [end:1-4] \
    [poison_rate:0-1] \
    [notation]
```

to reproduce the results of Fourier analyses, where vanilla denotes no defense deployment, ga denotes gradient alignment, mslr denotes multiple radial scalings, lora denotes low-rank adaptation ([LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)). Additionally, the parameter ``start`` and ``end`` control the number of attack methods, where 0 denotes [Badnets](https://doi.org/10.18653/v1/2020.acl-main.249), 1 denotes [Addsent](https://doi.org/10.1109/ACCESS.2019.2941376), 2 denotes [StyleBkd](https://doi.org/10.18653/v1/2021.emnlp-main.374), and 3 denotes [HiddenKiller](https://doi.org/10.18653/v1/2021.acl-long.37).


# Acknowledgement
This work can not be done without the help of the following repos:

- OpenBackdoor: [https://github.com/thunlp/OpenBackdoor](https://github.com/thunlp/OpenBackdoor)
- OpenDelta: [https://github.com/thunlp/OpenDelta](https://github.com/thunlp/OpenDelta)
- PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

# Citation

```ruby
@article{wu2024acquiring,
  title   = {Acquiring Clean Language Models from Backdoor Poisoned Datasets by Downscaling Frequency Space},
  author  = {Wu, Zongru and Zhang, Zhuosheng and Cheng, Pengzhou and Liu, Gongshen},
  journal = {arXiv preprint arXiv:2402.12026},
  year    = {2024}
}
```