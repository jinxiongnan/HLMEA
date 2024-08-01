# HLMEA

In this paper, we propose HLMEA, a novel hybrid language model-based unsupervised EA method that synergistically integrates small language models (SLMs) and LLMs. HLMEA formulates the EA task into a filtering and single-choice problem. SLMs filter top candidate entities based on textual representations generated from KG triples. Then, LLMs refine this selection to identify the most semantically aligned entities. An iterative self-training mechanism allows SLMs to distill knowledge from LLM outputs, enhancing the ability of hybrid language models in subsequent rounds cooperatively. The overall architecture is shown as follows:
![architecture](https://github.com/user-attachments/assets/1f7e51ff-9103-4eb3-ba69-7e75bf313553){:width="400px" height="300px"}
