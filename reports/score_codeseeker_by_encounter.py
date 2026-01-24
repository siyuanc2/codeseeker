"""
Codeseeker (aka CLH coding like humans) does code predictions per document.
This does not align with the standard medical coding practice of assigning codes to a finalized encounter.
This script scores the codeseeker by encounter by aggregating the predictions by unique hadm_id.
"""

import json
from collections import defaultdict
from datasets import load_from_disk
from typing import Any, List, Tuple

def score_example(targets: List[str], preds: List[str], score_prefix_only: bool = False) -> Tuple[int, int, int]:
    """
    Score a single example by calculating true positives, false positives, and false negatives.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    if score_prefix_only:
        preds = [p[:3] for p in preds]
        targets = [t[:3] for t in targets]
    ref_set = set(targets)
    pred_set = set(preds)
    true_positives = len(ref_set & pred_set)
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)
    return true_positives, false_positives, false_negatives

def add_hadm_id(row: dict[str, Any]) -> dict[str, Any]:
    row['hadm_id'] = int(row['aid'].split('_')[0])
    return row

def main():
    data_path = "/home/ubuntu/Desktop/resources/files/codeseeker/experiments/agentic-system/v2_full/gpt-5-full-vocab/"
    agent_stage = "assign"
    ds = load_from_disk(data_path + f"dataset_{agent_stage}")
    ds = ds.map(add_hadm_id)

    # Pre-process dataset to group by hadm_id for efficient lookup
    print("Pre-processing dataset for efficient lookup...")
    hadm_id_to_preds = defaultdict(list)
    hadm_id_to_refs = defaultdict(list)

    for item in ds:
        hadm_id_to_preds[item["hadm_id"]].extend(item["output"])
        hadm_id_to_refs[item["hadm_id"]].extend(item["targets"])

    print(f"Pre-processed {len(hadm_id_to_preds)} unique hadm_ids")
    # print(hadm_id_to_preds.keys())

    predictions_payload = [
        {
            "hadm_id": hadm_id,
            "predicted_codes": hadm_id_to_preds[hadm_id],
            "inside_vocab_target_codes": hadm_id_to_refs[hadm_id],
        }
        for hadm_id in sorted(hadm_id_to_preds)
    ]
    # Optional: Save the predictions to a file
    # output_path = data_path + f"hadm_id_to_preds_{agent_stage}.json"
    # with open(output_path, "w", encoding="utf-8") as handle:
    #     json.dump(predictions_payload, handle, indent=2)
    # print(f"Saved hadm_id predictions to {output_path}")

    # Score the predictions by encounter
    all_true_positives, all_false_positives, all_false_negatives = 0, 0, 0
    for this_hadm_id, this_refs in hadm_id_to_refs.items():
        true_positives, false_positives, false_negatives = score_example(this_refs, hadm_id_to_preds[this_hadm_id], score_prefix_only=False)
        all_true_positives += true_positives
        all_false_positives += false_positives
        all_false_negatives += false_negatives
    print(f"TP: {all_true_positives}, FP: {all_false_positives}, FN: {all_false_negatives}")
    print(f"P: {all_true_positives / (all_true_positives + all_false_positives):.4f}, R: {all_true_positives / (all_true_positives + all_false_negatives):.4f}, F1: {2 * all_true_positives / (2 * all_true_positives + all_false_positives + all_false_negatives):.4f}")

if __name__ == "__main__":
    main()