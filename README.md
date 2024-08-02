# Overview

In this paper, we propose HLMEA, a novel hybrid language model-based unsupervised EA method that synergistically integrates small language models (SLMs) and LLMs. HLMEA formulates the EA task into a filtering and single-choice problem. SLMs filter top candidate entities based on textual representations generated from KG triples. Then, LLMs refine this selection to identify the most semantically aligned entities. An iterative self-training mechanism allows SLMs to distill knowledge from LLM outputs, enhancing the ability of hybrid language models in subsequent rounds cooperatively. The overall architecture is shown as follows:
<div align=center>
  <img src="https://github.com/user-attachments/assets/1f7e51ff-9103-4eb3-ba69-7e75bf313553" alt="architecture" width="700" height="350">
</div>

# Environment
To run the script, you need to install the conda first and then the following environment:
```
conda env create -f environment.yml
```

The expected structure of files is:
```
HLMEA
 |-- scripts
 |-- dataset
 |    |-- DBP15K_FR_EN
 |    |-- DBP15K_JA_EN
 |    |-- DBP15K_ZH_EN
 |    |-- DBP15K_DE_EN_V1
 |    |-- DBP15K_FR_EN_V1
 |    |-- DBP100K_DE_EN_V1
 |    |-- DBP100K_FR_EN_V1
 |    |-- DW15K_V1
 |    |-- DY15K_V1
```

# How to use
Firstly, modify llm_call.py with your API key (line 13), service URL (line 32), and access token (line 45).
Then, run the script:
```
cd scripts
xxx
```
