<h3 align="center"><a href="" style="color:#9C276A">
Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models</a></h3>
<h5 align="center"> 


<h5 align="center">
[![arXiv](https://img.shields.io/badge/Arxiv-2506.13977-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2506.13977) 
[![CriticTool-Dataset](https://img.shields.io/badge/🤗HuggingFace-%20CriticTool-blue.svg)](https://huggingface.co/datasets/chocckaka/CriticTool-Dataset)
[![Project Page](https://img.shields.io/badge/ProjectPage-Website-green.svg?logo=github)]([YuZeng260/AGILE · Datasets at Hugging Face](https://huggingface.co/datasets/YuZeng260/AGILE))

## 🔍 Introduction

Official Implement of "CRITICTOOL: Evaluating Self-Critique Capabilities of Large Language Models in Tool-Calling Error Scenarios"

> The ability of large language models (LLMs) to utilize external tools has enabled them to tackle an increasingly diverse range of tasks. However, as the tasks become more complex and long-horizon, the intricate tool utilization process may trigger various unexpected errors. Therefore, how to effectively handle such errors, including identifying, diagnosing, and recovering from them, has emerged as a key research direction for advancing tool learning. In this work, we first extensively analyze the types of errors encountered during the function-calling process on several competitive tool evaluation benchmarks. Based on it, we introduce CriticTool, a comprehensive critique evaluation benchmark specialized for tool learning. Building upon a novel evolutionary strategy for dataset construction, CriticTool holds diverse tool-use errors with varying complexities, which better reflects real-world scenarios. We conduct extensive experiments on CriticTool, and validate the generalization and effectiveness of our constructed benchmark strategy. We also provide an in-depth analysis of the tool reflection ability on various LLMs, offering a new perspective on the field of tool learning in LLMs.

<div>
<center>
<img src="docs/figure/teaser.jpg">
</div>

## 🚀 News
**[2025/6/11]** Paper available on [Arxiv](https://arxiv.org/pdf/2506.13977).🔥🔥🔥

**[2025/6/24]** Release [CriticTool-Dataset](https://huggingface.co/datasets/chocckaka/CriticTool-Dataset).🤗🤗🤗

**[2025/8/20]** CriticTool is accepted by `EMNLP 2025`.🎉🎉🎉

**[2025/8/31]** Release test scripts and update [CriticTool-Dataset](https://huggingface.co/datasets/chocckaka/CriticTool-Dataset).✨✨✨

## 🧾 Todo
- [x] Release CriticTool dataset.

- [x] Release CriticTool evaluation code.


## 🔧 Install Dependencies
```
# Clone the CriticTool repository
git clone https://github.com/Shellorley0513/CriticTool.git

# Change directory to CriticTool
cd CriticTool

# Create a new Conda environment with Python 3.10
conda create -n critictool python=3.10

# Activate the new environment
conda activate critictool

# Install the package
pip install -r requirements.txt
```

## 🛫 Get Start
### Test Data
You can download the dataset through huggingface via this [CriticTool-Dataset](https://huggingface.co/datasets/chocckaka/CriticTool-Dataset).
Please place the test data files in the project directory with the following structure:
```
CriticTool/
├── critictool/
├── CriticTool-Dataset/
│   ├── external_error/
│   │   ├── ...
│   └── internal_error/
│       ├── ...
└── ...     
```

### Model Evaluation
The `run_test.sh` script provides a way to evaluate all types of data. To evaluate a local model, specify the parameters: `model_name`, `out_dir`, `template` and `batchsize`. To evaluate a API model, specify the parameters: `model_name`, `out_dir`, `api_key`, `base_url`, and `batchsize`. \
Then run the script with the following command:
```
sh run_test.sh
```
Note:\
When inferring with a local model, the messages will be converted into the raw string format using the appropriate `template`. `template` examples can be found in [chat_template.py](critictool/utils/chat_template.py).
We provide `llama2`, `llama3`, `qwen`, `ministral` and `glm4`templates. If you need to evaluate models with other templates, please organize them following the provided example format.

## 📊 Final Results
You can find the inference results and evaluation results for each query data in the `out_dir` directory. The average evaluation metrics are available in the `results.jsonl` file.

## 🖊️ Citation
If you find CriticTool useful for your research and applications, please cite using this BibTeX:
```
@article{huang2025critictool,
  title={CRITICTOOL: Evaluating Self-Critique Capabilities of Large Language Models in Tool-Calling Error Scenarios},
  author={Huang, Shiting and Fang, Zhen and Chen, Zehui and Yuan, Siyu and Ye, Junjie and Zeng, Yu and Chen, Lin and Mao, Qi and Zhao, Feng},
  journal={arXiv preprint arXiv:2506.13977},
  year={2025}
}
```