# MERLIN

Official github repository of MERLIN : Multimodal Embedding Refinement via LLM-based Iterative Navigation for Text-Video Retrieval-Rerank Pipeline

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2407.12508'><img src='https://img.shields.io/badge/Paper-PDF-blue'></a>
</div>
<br>

🔴 I updated the initial version of the code and tested with several cases of MSRVTT-1k. You will be able to followup how MERLIN works with current code. (you can check some examples in the output folder just in case)<br>
🔴 Soon I will clean up the codes and make test run them on other datasets too.<br>
🔴 I will update the README.md with more details about the code.<br>
🚧 Rest of the code will be updated ASAP🙏. 🚧


## Subsampled Testset
- [Link](https://drive.google.com/drive/folders/19IPbbbV-ugZdqYCHHY5R-m4bCECy4-hd?usp=sharing)

## Precomputed Embeddings

We provide precomputed vertex embeddings for the following datasets:
- [ActivityNet](https://drive.google.com/drive/folders/1xr49ADvTqIGbnbg6Xvd0PAh3gklwKvaI?usp=drive_link) 
- [MSRVTT-1k](https://drive.google.com/drive/folders/1xr49ADvTqIGbnbg6Xvd0PAh3gklwKvaI?usp=drive_link) 
- [MSVD](https://drive.google.com/drive/folders/1xr49ADvTqIGbnbg6Xvd0PAh3gklwKvaI?usp=drive_link) 

## MERLIN's log on ActivityNet, MSRVTT-1k, MSVD
- [Link](https://drive.google.com/drive/folders/1E2zjiMVTtuEQA4Hs6RwsCbxVwUQ88W_F?usp=sharing)

## Brief of MERLIN
There are 3 main components in MERLIN to help understand the pipeline.
1. Questioner 
2. Reranker
3. Answerer (it is human simulating agent and can be replaced with actual human). It is not part of MERLIN pipeline but it is used to replace human in the loop.

## Environment Setup 

* Since most of embedding computation is done by API, there is no strict version dependency. Code will run on most of versions of packages(torch, sklearn, numpy, etc.) you have. 

* I will update the environment setup with both pixi & requirements.txt soon though.

### Before running MERLIN

* You need to have a Google Cloud account with Vertex AI API enabled.
* You need to have an OpenAI API key.
* Vertex AI API cost should not cost much but be careful with using OpenAI API.
* You need to have a dataset downloaded and placed in the `data` folder.


## Running MERLIN

To run MERLIN on a specific dataset:

```bash
python run_merlin.py --dataset msrvtt --data_path /path/to/data --num_rounds 5
```

Available datasets:
- `msrvtt`: MSRVTT-1k dataset
- `msvd`: MSVD dataset
- `anet`: ActivityNet dataset

### Command Line Arguments

- `--dataset`: Dataset to process (required, choices: "msvd", "msrvtt", "anet")
- `--data_path`: Path to dataset directory (default: "data")
- `--output_dir`: Directory to save outputs (default: "outputs")
- `--num_rounds`: Number of rounds for question-answering iteration (default: 5)
- `--model_name`: OpenAI model to use (default from .env)
- `--max_tokens`: Maximum tokens for model response (default from .env)
- `--debug`: Enable debug mode
- `--env_file`: Path to .env file (default: ".env.local")
