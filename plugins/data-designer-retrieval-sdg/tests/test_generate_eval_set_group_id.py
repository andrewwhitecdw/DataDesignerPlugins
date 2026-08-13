# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd

from data_designer_retrieval_sdg.chunking import normalize_source_id
from data_designer_retrieval_sdg.convert import generate_eval_set


def test_generate_eval_set_group_id_qrels_reference_corpus_ids(tmp_path):
    """Regression: group-id eval qrels must reference corpus.jsonl _id values."""
    raw_source = "my-source"
    file_identifier = normalize_source_id(raw_source)
    corpus = {
        "first chunk text": "d_1111111111111111",
        "second chunk text": "d_2222222222222222",
    }
    chunk_mapping = {
        (file_identifier, 0): "first chunk text",
        (file_identifier, 1): "second chunk text",
    }
    eval_df = pd.DataFrame(
        [
            {"source_id": raw_source, "question": "q1", "segment_ids": [0]},
            {"source_id": raw_source, "question": "q2", "segment_ids": [1]},
        ]
    )

    count = generate_eval_set(
        corpus=corpus,
        chunk_mapping=chunk_mapping,
        eval_df=eval_df,
        output_dir=str(tmp_path),
        max_pos_docs=5,
        eval_only=True,
        use_group_id_in_eval=True,
    )
    assert count == 2

    corpus_ids = set()
    with open(tmp_path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            corpus_ids.add(entry["_id"])

    missing: list[str] = []
    with open(tmp_path / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            _, corpus_id, _ = line.strip().split("\t")
            if corpus_id not in corpus_ids:
                missing.append(corpus_id)

    assert not missing, f"qrels references corpus ids not present in corpus.jsonl: {missing}"
