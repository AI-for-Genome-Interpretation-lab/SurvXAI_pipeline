
# SurvXAI: Narrative-based explainable AI for precision oncology survival prediction

  

## Overview
This repository accompanies the paper "Narrative-based Explainable AI for Precision Oncology Survival Prediction". 
SurvXAI is a pipeline that: 
1. Trains Random Survival Forest (RSF) for survival prediction 
2. Extracts patient-specific feature contributions using SurvSHAP(t) 
3. Generates natural language explanations via locally deployed LLMs

## Installation 
```bash 
pip install -r requirements.txt 
```
## LLM Setup 
Start a vLLM server in a separate terminal:
```bash
pip install vllm
vllm serve "openai/gpt-oss-120b"
```
Tested on HPC cluster with the following LLMs:
 - openai/gpt-oss-120b 
 - deepseek-ai/DeepSeek-R1-Distill-Llama-70B 
 - google/gemma-3-27b-it 
 - 01-ai/Yi-34B-Chat

##  Implementation

```bash
# Step 1: Generate survival predictions and SHAP values 
python src/1_rsf_shap_extraction.py 
# Step 2: Generate LLM narratives (requires vLLM server running) 
python src/2_prompt_withllm.py 
```
## Project Structure 
``` 
├── data/ # Dataset files 
├── src/ 
│   ├── 1_rsf_shap_extraction.py 
│   └── 2_prompt_withllm.py 
├── outputs/ # Survival results 
└── requirements.txt 
```
## Acknowledgments
Prompt structure adapted from [XAIstories](https://github.com/ADMAntwerp/XAIstories).
 
