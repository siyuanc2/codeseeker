# Reproduced Results
Codeseeker (aka coding like humans, CLH) produces code predictions per document.
This does not align with the industry standard medical coding practice of assigning codes to an encounter containing all documents for a outpatient visit or inpatient hospitalization.

The following results are reproduced by running `experiments/benchmark.py` with the following configurations, and then running `reports/score_codeseeker_by_encounter.py` to score the predictions by encounter.

Dataset: [MDACE dataset](https://github.com/solventum-oss/MDACE) (ICD-10 Inpatient v1.0 subset)
> Note: There are 302 encounters in this subset, but CLH only returns predictions for 299 encounters. The missing encounters are (hadm_id): 
> 100197, 102763, 106851. 
> The following results are reported excluding the missing encounters i.e., not considering ground truth codes for these encounters as false negatives.

## gpt-5 Full Vocab (>74,000 codes)

<details>
<summary>Click to expand config</summary>

```json
{
  "experiment_id": "agentic-system",
  "experiment_name": "v2_full",
  "dataset": "mdace-icd10cm",
  "seed": 1,
  "n_samples": 1,
  "base_model": {
    "provider": "azure",
    "deployment": "my-gpt-5-deployment",
    "api_base": "https://my-project-name.cognitiveservices.azure.com/",
    "endpoint": "chat/completions",
    "use_cache": false
  },
  "temperature": 1.0,
  "max_tokens": null,
  "analyse_agent": {
    "agent_type": "base",
    "prompt_name": "analyse_agent/strict_v3"
  },
  "locate_agent": {
    "agent_type": "split",
    "prompt_name": "locate_agent/locate_few_terms_v3"
  },
  "verify_agent": {
    "agent_type": "reasoning",
    "prompt_name": "verify_agent/one_per_term_v4"
  },
  "assign_agent": {
    "agent_type": "reasoning",
    "prompt_name": "assign_agent/reasoning_v5"
  },
  "batch_size": 1,
  "num_workers": 1,
  "all_codes": true,
  "topk_assignable_terms": 10,
  "embed_config": [
    {
      "type": "st",
      "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
      "query_key": "output"
    }
  ],
  "qdrant_config": {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334
  },
  "distance": "Cosine",
  "hnsw": {
    "m": 32,
    "ef_construct": 256
  },
  "debug": false,
  "use_cache": false
}
```
</details>

| Step | TP | FP | FN | Micro P | Micro R | Micro F1 |
|------|----|----|----|---------|---------|----------|
| Analyze | 2015 | 29881 | 1168 | 0.0632 | 0.6331 | 0.1149 |
| Locate | 1690 | 2793 | 1493 | 0.3770 | 0.5309 | 0.4409 |
| Verify | 1558 | 1742 | 1625 | 0.4721 | 0.4895 | 0.4806 |
| Assign | 1521 | 1635 | 1662 | 0.4819 | 0.4779 | 0.4799 |

[Log file](logs/gpt-5-simple.log)

## gpt-5-mini Full Vocab (>74,000 codes)

<details>
<summary>Click to expand config</summary>

```json
{
  "experiment_id": "agentic-system",
  "experiment_name": "v2_full",
  "dataset": "mdace-icd10cm",
  "seed": 1,
  "n_samples": 1,
  "base_model": {
    "provider": "azure",
    "deployment": "my-gpt-5-mini-deployment",
    "api_base": "https://my-project-name.cognitiveservices.azure.com/",
    "endpoint": "chat/completions",
    "use_cache": false
  },
  "temperature": 1.0,
  "max_tokens": null,
  "analyse_agent": {
    "agent_type": "base",
    "prompt_name": "analyse_agent/strict_v3"
  },
  "locate_agent": {
    "agent_type": "split",
    "prompt_name": "locate_agent/locate_few_terms_v3"
  },
  "verify_agent": {
    "agent_type": "reasoning",
    "prompt_name": "verify_agent/one_per_term_v4"
  },
  "assign_agent": {
    "agent_type": "reasoning",
    "prompt_name": "assign_agent/reasoning_v5"
  },
  "batch_size": 1,
  "num_workers": 2,
  "all_codes": true,
  "topk_assignable_terms": 10,
  "embed_config": [
    {
      "type": "st",
      "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
      "query_key": "output"
    }
  ],
  "qdrant_config": {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334
  },
  "distance": "Cosine",
  "hnsw": {
    "m": 32,
    "ef_construct": 256
  },
  "debug": false,
  "use_cache": false
}
```
</details>

| Step | TP | FP | FN | Micro P | Micro R | Micro F1 |
|------|----|----|----|---------|---------|----------|
| Analyze | 1951 | 36853 | 1232 | 0.0503 | 0.6129 | 0.0929 |
| Locate | 1737 | 6428 | 1446 | 0.2127 | 0.5457 | 0.3061 |
| Verify | 1494 | 2894 | 1689 | 0.3405 | 0.4694 | 0.3947 |
| Assign | 1448 | 2361 | 1735 | 0.3802 | 0.4549 | 0.4142 |

## gpt-oss-120b Full Vocab (>74,000 codes)

<details>
<summary>Click to expand config</summary>

```json
{
  "experiment_id": "agentic-system",
  "experiment_name": "v2_full",
  "dataset": "mdace-icd10cm",
  "seed": 1,
  "n_samples": 1,
  "base_model": {
    "provider": "vllm",
    "deployment": "openai/gpt-oss-120b",
    "api_base": "http://localhost:8000/v1",
    "endpoint": "chat/completions",
    "use_cache": true
  },
  "temperature": 1.0,
  "max_tokens": 10000,
  "analyse_agent": {
    "agent_type": "base",
    "prompt_name": "analyse_agent/strict_v3"
  },
  "locate_agent": {
    "agent_type": "split",
    "prompt_name": "locate_agent/locate_few_terms_v3"
  },
  "verify_agent": {
    "agent_type": "reasoning",
    "prompt_name": "verify_agent/one_per_term_v4"
  },
  "assign_agent": {
    "agent_type": "reasoning",
    "prompt_name": "assign_agent/reasoning_v5"
  },
  "batch_size": 1,
  "num_workers": 16,
  "all_codes": true,
  "topk_assignable_terms": 10,
  "embed_config": [
    {
      "type": "st",
      "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
      "query_key": "output"
    }
  ],
  "qdrant_config": {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334
  },
  "distance": "Cosine",
  "hnsw": {
    "m": 32,
    "ef_construct": 256
  },
  "debug": false,
  "use_cache": true
}
```
</details>

| Step | TP | FP | FN | Micro P | Micro R | Micro F1 |
|------|----|----|----|---------|---------|----------|
| Analyze | 2006 | 29432 | 1177 | 0.0638 | 0.6302 | 0.1159 |
| Locate | 1794 | 4279 | 1389 | 0.2954 | 0.5636 | 0.3876 |
| Verify | 1529 | 2015 | 1654 | 0.4314 | 0.4804 | 0.4546 |
| Assign | 1313 | 1616 | 1870 | 0.4483 | 0.4125 | 0.4296 |

## gpt-oss-120b Limited Vocab (898 codes)

<details>
<summary>Click to expand config</summary>

```json
{
  "experiment_id": "agentic-system",
  "experiment_name": "v2_full",
  "dataset": "mdace-icd10cm",
  "seed": 1,
  "n_samples": 1,
  "base_model": {
    "provider": "vllm",
    "deployment": "openai/gpt-oss-120b",
    "api_base": "http://localhost:8000/v1",
    "endpoint": "chat/completions",
    "use_cache": true
  },
  "temperature": 1.0,
  "max_tokens": 10000,
  "analyse_agent": {
    "agent_type": "base",
    "prompt_name": "analyse_agent/strict_v3"
  },
  "locate_agent": {
    "agent_type": "split",
    "prompt_name": "locate_agent/locate_few_terms_v3"
  },
  "verify_agent": {
    "agent_type": "reasoning",
    "prompt_name": "verify_agent/one_per_term_v4"
  },
  "assign_agent": {
    "agent_type": "reasoning",
    "prompt_name": "assign_agent/reasoning_v5"
  },
  "batch_size": 1,
  "num_workers": 16,
  "all_codes": false,
  "topk_assignable_terms": 10,
  "embed_config": [
    {
      "type": "st",
      "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
      "query_key": "output"
    }
  ],
  "qdrant_config": {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334
  },
  "distance": "Cosine",
  "hnsw": {
    "m": 32,
    "ef_construct": 256
  },
  "debug": false,
  "use_cache": true
}
```
</details>

| Step | TP | FP | FN | Micro P | Micro R | Micro F1 |
|------|----|----|----|---------|---------|----------|
| Analyze | 2186 | 21017 | 997 | 0.0942 | 0.6868 | 0.1657 |
| Locate | 1990 | 3291 | 1193 | 0.3768 | 0.6252 | 0.4702 |
| Verify | 1659 | 1526 | 1524 | 0.5209 | 0.5212 | 0.5210 |
| Assign | 1384 | 1135 | 1799 | 0.5494 | 0.4348 | 0.4854 |