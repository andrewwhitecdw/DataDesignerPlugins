# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BEIR eval corpus/qrel id consistency."""

import json
import os
import tempfile

import pandas as pd
import pytest

from data_designer_retrieval_sdg.convert import generate_eval_set, get_corpus_id


def test_eval_corpus_id_matches_qrels_with_group_id():
    text = "The quick brown fox jumps over the lazy dog."
    hash_id = get_corpus_id(text)
    corpus = {text: hash_id}
    chunk_mapping = {("src", 0): text}

    eval_df = pd.DataFrame(
        [
            {
                "file_name": ["src"],
                "segment_ids": [0],
                "question": "What animal jumps?",
            }
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_eval_set(
            corpus,
            chunk_mapping,
            eval_df,
            tmpdir,
            max_pos_docs=5,
            eval_only=True,
            use_group_id_in_eval=True,
        )

        corpus_ids = set()
        group_ids = set()
        with open(os.path.join(tmpdir, "corpus.jsonl")) as f:
            for line in f:
                entry = json.loads(line)
                corpus_ids.add(entry["_id"])
                group_ids.add(entry.get("group_id"))

        qrels_corpus_ids = set()
        with open(os.path.join(tmpdir, "qrels", "test.tsv")) as f:
            header = next(f)
            assert header.strip() == "query-id\tcorpus-id\tscore"
            for line in f:
                parts = line.strip().split("\t")
                qrels_corpus_ids.add(parts[1])

        assert qrels_corpus_ids.issubset(corpus_ids)
        assert hash_id in corpus_ids
