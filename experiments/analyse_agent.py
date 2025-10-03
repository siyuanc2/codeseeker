from collections import OrderedDict
import typing as typ

import pydantic
import rich
from loguru import logger

from agents.analyse_agent import create_analyse_agent
from agents.base import HfBaseAgent
import dataloader
from dataloader.base import DatasetConfig
import config as cnf

import hashlib
from pathlib import Path

import datasets


from dataloader import load_dataset

from trie.base import Trie
import utils as exp_utils
from retrieval.qdrant_search import client as qdrant_client
from retrieval.qdrant_search import models as qdrant_models
from retrieval.qdrant_search import factory as qdrant_factory


class Arguments(pydantic.BaseModel):

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "openai/gpt-oss-20b",
        "api_base": "http://localhost:8000/v1",
        "endpoint": "chat/completions",
        "use_cache": True,
    }
    prompt_name: str = "analyse_agent/strict_v3"
    agent_type: str = "base"
    temperature: float = 1.0
    max_tokens: int = 10_000
    seed: int = 1
    batch_size: int = 2
    num_workers: int = 4
    embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "output",
        },
    ]

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}
    rank: int = 10

    debug: bool = False

    experiment_id: str = "analyse-agent"
    experiment_name: str = "v3"

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


def pipe(
    agent: HfBaseAgent,
    embed_config: list[dict[str, str]],
    trie: Trie,
    dataset: datasets.Dataset,
    qdrant_service: qdrant_client.QdrantSearchService,
    rank: int,
    num_workers: int,
    batch_size: int,
    seed: int,
    hnsw: dict[str, int],
    distance: str,
    all_codes: bool = True,
) -> datasets.Dataset:
    """Run the pipeline for the analysis agent."""
    target_codes = set(code for row in dataset["targets"] for code in row)
    terms = [
        term.model_dump()
        | {"group": exp_utils.make_group_id(term, cross_references=True)}
        for term in trie.index.values()
        if term.assignable and (all_codes or term.code in target_codes)
    ]

    logger.info(f"Fetched `{len(terms)}` terms from the Trie.")

    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=terms,
        text_key="path",
        model_cfg=embed_config,
        hnsw_cfg=hnsw,
        distance=distance,
        service=qdrant_service,
        payload_keys=["id", "group"],
        recreate=False,
    )

    analyse_dataset = dataset.map(
        agent,
        num_proc=num_workers,
        batched=True,
        batch_size=batch_size,
        remove_columns=exp_utils._get_dataset(dataset).column_names,
        desc=f"[Analyse Agent] Generating search queries for seed {seed}.",
        load_from_cache_file=False,
    )

    logger.info("Searching for cross-references...")
    queries = analyse_dataset.to_list()
    cross_reference_results = qdrant_factory.search(
        data=analyse_dataset.to_list(),
        model_cfg=embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=rank,
        merge_search=False,
    )

    for idx, res in enumerate(cross_reference_results):
        cross_references = []
        for point in res.points:
            if not point.payload:
                continue
            term = trie.index[point.payload["id"]]
            if len(cross_references) > 0:
                continue
            if term.see_also:
                cross_references.append(term.see_also)
        queries[idx]["output"].extend(cross_references)

    # Create and ensure index
    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=terms,
        text_key="path",
        model_cfg=embed_config,
        hnsw_cfg=hnsw,
        distance=distance,
        service=qdrant_service,
        payload_keys=["id", "group"],
        recreate=False,
    )

    assignable_terms_results = qdrant_factory.search(
        data=queries,
        model_cfg=embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=rank,
        merge_search=False,
    )

    def group_by_rank(x: list[typ.Any], size: int) -> list[list[typ.Any]]:
        """Group a list of strings by size."""
        groups = []
        for i in range(0, len(x), size):
            groups.append(x[i : i + size])
        return groups

    retrieved_codes = []
    retrieved_terms = []

    for res in assignable_terms_results:
        seen_groups = set()
        unique_codes = set()
        grouped_terms = []
        grouped_points = group_by_rank(res.points, size=rank)

        for group in grouped_points:
            tmp_group = []
            group_term_ids = []  # Collect IDs for signature

            for point in group:
                if not point.payload:
                    continue
                term = trie.index[point.payload["id"]]
                if not term.code:
                    continue

                tmp_group.append(term.model_dump())
                group_term_ids.append(term.id)
                unique_codes.update(trie.get_term_codes(term.id, subterms=False))

            # Skip empty groups
            if not tmp_group:
                continue

            # Use term IDs for signature (much faster than JSON serialization)
            group_signature = tuple(sorted(group_term_ids))

            if group_signature not in seen_groups:
                sorted_group = sorted(tmp_group, key=lambda x: x["path"])
                grouped_terms.append(sorted_group)
                seen_groups.add(group_signature)

        retrieved_codes.append(list(unique_codes))
        retrieved_terms.append(grouped_terms)

    average_length = sum(len(row) for row in retrieved_terms) / len(retrieved_terms)
    average_codes_length = (
        sum(len(row) for row in retrieved_codes) / len(retrieved_codes) / average_length
    )
    logger.info(
        "[Analyse Agent] Results: "
        f"Average extracted snippets: {average_length:.2f}, "
        f"Average extracted codes: {average_codes_length:.2f}"
    )

    return datasets.Dataset.from_dict(
        {
            "output": retrieved_codes,
            "terms": retrieved_terms,
            "query": analyse_dataset["output"],
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
    # mdace = mdace.select(range(100))
    mdace = mdace.map(
        lambda x: {
            **x,
            "evidence": [
                " ".join(
                    x["note"][loc[0] : loc[-1] + 1] for loc in annotation["locations"]
                )
                for annotation in x["evidence_spans"]
            ],
        }
    )
    # mdace_codes = set(code for row in mdace["targets"] for code in row)
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    eval_trie: dict[str, int] = OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )

    agent = create_analyse_agent(
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

    analyse_eval_data = pipe(
        agent=task_maker,
        embed_config=args.embed_config,
        trie=xml_trie,
        dataset=mdace,
        qdrant_service=qdrant_service,
        rank=args.rank,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
        hnsw=args.hnsw,
        distance=args.distance,
        all_codes=True,  # Set to True to include all codes in the Trie
    )

    exp_utils.evaluate_and_dump_metrics(
        analyse_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix=args.experiment_name,
    )


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
