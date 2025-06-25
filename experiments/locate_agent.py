from collections import OrderedDict
import typing as typ

import pydantic
import rich
from loguru import logger

from agents.base import HfBaseAgent
from agents.locate_agent import create_locate_agent
import dataloader
from dataloader.base import DatasetConfig
import config as cnf

import hashlib
from pathlib import Path

import datasets


from dataloader.interface import load_dataset
from trie.base import Trie
import utils as exp_utils
from retrieval.qdrant_search import client as qdrant_client
from retrieval.qdrant_search import models as qdrant_models
from retrieval.qdrant_search import factory as qdrant_factory


class Arguments(pydantic.BaseModel):

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6539/v1",
        "endpoint": "completions",
        "use_cache": True,
    }
    prompt_name: str = "locate_agent/locate_few_terms_v3"
    agent_type: str = "split"
    temperature: float = 0.0
    max_tokens: int = 5_000
    seed: int = 1  # e.g., "1:2:3:4:5"
    batch_size: int = 1
    num_workers: int = 4
    embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "query",
        },
    ]

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}
    rank: int = 15

    debug: bool = False

    experiment_id: str = "locate-agent"
    experiment_name: str = "v2"

    def get_hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(exclude={"experiment_id", "experiment_name"})
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    def get_experiment_folder(self) -> Path:
        """Get the experiment folder path."""
        experiment_folder = (
            cnf.DUMP_FOLDER
            / f"{self.experiment_id}/{self.experiment_name}/{self.get_hash()}"
        )
        experiment_folder.mkdir(parents=True, exist_ok=True)
        return experiment_folder


def decode_with_fallback(row: dict[str, typ.Any]) -> dict[str, typ.Any]:
    decoded_output = [
        [
            row["terms"][idx][term_idx - 1]["id"]
            for term_idx in group
            if len(row["terms"][idx]) >= term_idx > 0
        ]
        for idx, group in enumerate(row["output"])
    ]
    # Fallback: if all groups are empty, use all available terms for this sample
    if not any(decoded_output):
        # fallback: use all term ids from x["terms"]
        fallback = [[term["id"] for term in group] for group in row["terms"]]
        return {**row, "output": fallback}
    return {**row, "output": decoded_output}


def retrieve_terms(
    embed_config: list[dict[str, str]],
    xml_trie: Trie,
    dataset: datasets.Dataset,
    qdrant_service: qdrant_client.QdrantSearchService,
    rank: int,
) -> datasets.Dataset:
    """Run the pipeline for the analysis agent."""
    terms = [
        {
            **term.model_dump(),
            "group": exp_utils.make_group_id(term, cross_references=True),
        }
        for term in xml_trie.index.values()
    ]

    logger.info(f"Fetched `{len(terms)}` terms from the Trie.")

    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=terms,
        text_key="path",
        model_cfg=embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id", "group"],
        recreate=False,
    )

    logger.info("Searching for cross-references...")
    queries = dataset.to_list()
    cross_reference_results = qdrant_factory.search_by_group(
        data=dataset.to_list(),
        model_cfg=embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=rank,
        group_size=1,
        group_key="group",
    )

    for idx, res in enumerate(cross_reference_results):
        cross_references = []
        for i, point in enumerate(res.points, start=1):
            if not point.payload:
                continue
            term = xml_trie.index[point.payload["id"]]
            if len(cross_references) > 1:
                continue  # Limit to first 5 cross-references
            if term.see_also:
                cross_references.append(term.see_also)
        queries[idx]["query"].extend(cross_references)

    code_terms = [
        {
            **term.model_dump(),
            "group": exp_utils.make_group_id(term, cross_references=False),
        }
        for term in xml_trie.index.values()
    ]

    # Create and ensure index
    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=code_terms,
        text_key="path",
        model_cfg=embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id", "group"],
        recreate=False,
    )

    assignable_terms_results = qdrant_factory.search_by_group(
        data=queries,
        model_cfg=embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=rank,
        group_size=1,
        group_key="group",
    )

    def group_by_rank(x: list[str], size: int) -> list[list[str]]:
        """Group a list of strings by size."""
        groups = []
        for i in range(0, len(x), size):
            groups.append(x[i : i + size])
        return groups

    retrieved_codes = []
    retrieved_terms = []
    for res in assignable_terms_results:
        unique_codes = set()
        terms = []
        for point in res.points:
            if not point.payload:
                continue
            term = xml_trie.index[point.payload["id"]]
            if not term.code:
                continue
            terms.append(term.model_dump())
            unique_codes.update(xml_trie.get_term_codes(term.id, subterms=False))
        retrieved_codes.append(list(unique_codes))
        groups = group_by_rank(terms, size=rank)
        retrieved_terms.append(groups)

    average_length = sum(len(row) for row in retrieved_terms) / len(retrieved_terms)
    logger.info(
        f"[Analyse Agent] Average number of extracted snippets: {average_length:.2f}"
    )

    return datasets.Dataset.from_dict(
        {
            "output": retrieved_codes,
            "terms": retrieved_terms,
            "query": dataset["query"],
            "targets": dataset["targets"],
            "subset_targets": [
                [code for code in targets if code in retrieved]
                for targets, retrieved in zip(dataset["targets"], retrieved_codes)
            ],
            "note": dataset["note"],
            "note_type": dataset["note_type"],
            "aid": dataset["aid"],
        }
    )


def pipe(
    agent: HfBaseAgent,
    dataset: datasets.Dataset,
    trie: Trie,
    num_workers: int,
    batch_size: int,
    seed: int,
) -> datasets.Dataset:
    """Run the pipeline for the analysis agent."""

    locate_dataset = dataset.map(
        agent,
        num_proc=num_workers,
        batched=True,
        batch_size=batch_size,
        remove_columns=exp_utils._get_dataset(dataset).column_names,
        desc=f"[Locate Agent] Locating terms for seed {seed}.",
        load_from_cache_file=False,
    )

    locate_dataset = locate_dataset.map(
        decode_with_fallback,
        desc="[Locate Agent] Decoding output terms.",
        load_from_cache_file=False,
    )

    located_codes = []
    codes_to_verify = []
    terms_to_verify = []

    for idx, row in enumerate(locate_dataset["output"]):
        seen_groups = set()
        unique_codes = set()
        grouped_codes = []
        grouped_terms = []

        for group in row:
            if not group:
                continue

            codes = set()
            terms = []
            valid_term_ids = []

            for term_id in group:
                term_codes = set(trie.get_term_codes(term_id, subterms=False))

                if codes.intersection(term_codes):
                    # skip overlapping codes
                    continue

                codes.update(term_codes)
                terms.append(trie.index[term_id].model_dump())
                valid_term_ids.append(term_id)

            if not valid_term_ids:  # empty group -> skip
                continue

            group_signature = tuple(sorted(valid_term_ids))

            if group_signature not in seen_groups:
                grouped_codes.append(tuple(sorted(codes)))  # deterministic
                grouped_terms.append(tuple(terms))  # deterministic
                unique_codes.update(codes)
                seen_groups.add(group_signature)

        located_codes.append(tuple(sorted(unique_codes)))  # deterministic
        codes_to_verify.append(grouped_codes)
        terms_to_verify.append(grouped_terms)

    average_length = sum(len(row) for row in codes_to_verify) / len(codes_to_verify)
    logger.info(f"[Locate Agent] Average number of located codes: {average_length:.2f}")

    return datasets.Dataset.from_dict(
        {
            "codes": codes_to_verify,
            "terms": terms_to_verify,
            "output": located_codes,
            "targets": locate_dataset["targets"],
            "subset_targets": [
                [code for code in targets if code in retrieved]
                for targets, retrieved in zip(locate_dataset["targets"], located_codes)
            ],
            "note": locate_dataset["note"],
            "note_type": locate_dataset["note_type"],
            "aid": locate_dataset["aid"],
        }
    )


def decode_locate_output(row: dict[str, typ.Any]) -> dict[str, typ.Any]:
    """Decode the predictions from the locate agent output, with fallback if empty."""
    decoded_output = [
        [
            row["terms"][idx][term_idx - 1]["id"]
            for term_idx in group
            if len(row["terms"][idx]) >= term_idx > 0
        ]
        for idx, group in enumerate(row["output"])
    ]
    # Fallback: if all groups are empty, use all available terms for this sample
    if not any(decoded_output) or all(len(group) == 0 for group in decoded_output):
        fallback = [[term["id"] for term in group] for group in row["terms"]]
        return {**row, "output": fallback}
    return {**row, "output": decoded_output}


def run(args: Arguments) -> None:
    rich.print(args)
    with open(args.get_experiment_folder() / "config.json", "w") as f:  # type: ignore
        f.write(args.model_dump_json(indent=2))
    qdrant_service = qdrant_client.QdrantSearchService(
        **args.qdrant_config.model_dump()
    )
    xml_trie = exp_utils.build_icd_trie(year=2022)
    mdace = load_dataset(DatasetConfig(**dataloader.DATASET_CONFIGS["mdace-icd10cm"]))
    mdace = exp_utils.format_dataset(mdace, xml_trie, args.debug)
    mdace = mdace.select(range(100))  # For debugging purposes, limit to 100 samples
    mdace = mdace.map(
        lambda x: {
            **x,
            "query": [
                " ".join(
                    x["note"][loc[0] : loc[-1] + 1] for loc in annotation["locations"]
                )
                for annotation in x["evidence_spans"]
            ],
        }
    )
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    eval_trie: dict[str, int] = OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )

    mdace_with_retrieved_terms = retrieve_terms(
        embed_config=args.embed_config,
        xml_trie=xml_trie,
        dataset=mdace,
        qdrant_service=qdrant_service,
        rank=args.rank,
    )

    exp_utils.evaluate_and_dump_metrics(
        mdace_with_retrieved_terms,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix=f"{args.experiment_name}_upper_bound",
    )

    agent = create_locate_agent(
        agent_type=args.agent_type,
        prompt_name=args.prompt_name,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )

    task_maker = agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
        seed=args.seed,
    )

    locate_eval_data = pipe(
        agent=task_maker,
        dataset=mdace_with_retrieved_terms,
        trie=xml_trie,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    exp_utils.evaluate_and_dump_metrics(
        locate_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix=args.experiment_name,
    )


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
