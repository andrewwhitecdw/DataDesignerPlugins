"""Unit tests for postprocess helpers."""

import json
from pathlib import Path

import pandas as pd
import pytest

from data_designer_retrieval_sdg.postprocess import load_positive_docs_with_modality


@pytest.mark.parametrize("corpus_id_header", ["corpus-id", "corpus-id "])
def test_load_positive_docs_with_modality_accepts_standard_and_legacy_headers(
    tmp_path: Path, corpus_id_header: str
) -> None:
    """Standard and legacy trailing-space qrels headers both load."""
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    qrels_df = pd.DataFrame(
        [{"query-id": "q1", corpus_id_header: "doc1", "score": 1}]
    )
    qrels_tsv = qrels_dir / "test.tsv"
    qrels_df.to_csv(qrels_tsv, sep="\t", index=False)

    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"_id": "doc1", "text": "hello world", "group_id": "doc1"})
        + "\n"
    )

    split_json = tmp_path / "split.json"
    split_json.write_text(json.dumps({"text": ["q1"]}))

    positive_docs_df, doc_to_modality = load_positive_docs_with_modality(
        qrels_tsv, corpus_jsonl, split_json
    )

    assert doc_to_modality == {"doc1": "text"}
    assert positive_docs_df["doc_id"].tolist() == ["doc1"]
    assert positive_docs_df["modality"].tolist() == ["text"]
    assert positive_docs_df["text"].tolist() == ["hello world"]
