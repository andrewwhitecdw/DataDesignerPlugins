# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for relative multi_doc_manifest path resolution."""

import json

import fsspec

from data_designer.engine.resources.seed_reader import SeedReaderFileSystemContext
from data_designer_retrieval_sdg.seed_reader import DocumentChunkerSeedReader
from data_designer_retrieval_sdg.seed_source import DocumentChunkerSeedSource


def test_relative_multi_doc_manifest_resolved_to_root(tmp_path, monkeypatch):
    """A relative manifest path is resolved against context.root_path, not cwd."""
    root = tmp_path / "root"
    root.mkdir()
    file_a = root / "a.txt"
    file_b = root / "b.txt"
    file_c = root / "c.txt"
    (root / "manifest.json").write_text(json.dumps([[str(file_a), str(file_b)]]))
    file_a.write_text("A")
    file_b.write_text("B")
    file_c.write_text("C")

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    source = DocumentChunkerSeedSource(
        file_pattern="*.txt",
        recursive=False,
        multi_doc=True,
        multi_doc_manifest="manifest.json",
        bundle_size=2,
        max_docs_per_bundle=2,
    )
    context = SeedReaderFileSystemContext(
        root_path=root,
        fs=fsspec.filesystem("file"),
    )
    reader = DocumentChunkerSeedReader(source=source)
    manifest = reader.build_manifest(context=context)

    # With the bug, the manifest would not be found from cwd and auto-bundling
    # would produce two bundles ([a, b] and [c]) instead of one.
