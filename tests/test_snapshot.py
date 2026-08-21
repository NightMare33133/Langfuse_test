"""Unit tests for storage/snapshot.py consistency snapshots."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage.snapshot import (
    create_consistency_snapshot,
    get_snapshot_by_id,
    list_consistency_snapshots,
)


class TestConsistencySnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_manifest = Path(self.tmp_dir.name) / "test_manifest.jsonl"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_create_and_list_snapshots(self):
        with patch("storage.snapshot.SNAPSHOTS_FILE", self.tmp_manifest), \
             patch("storage.snapshot.SNAPSHOTS_DIR", Path(self.tmp_dir.name)):
            
            # 1. 初始列表为空
            self.assertEqual(list_consistency_snapshots(), [])

            # 2. 创建快照
            meta = {
                "document_title": "IT采购通用条款",
                "document_type": "通用条款",
                "topics": ["保密", "知识产权"],
            }
            snap = create_consistency_snapshot(
                file_name="Appendix A.docx",
                content_hash="abc123hash",
                minio_version_id="uuid-ver-1",
                dify_dataset_id="ds-999",
                dify_document_id="doc-888",
                metadata=meta,
                contract_package="baseline_2_4",
            )

            self.assertTrue(snap["snapshot_id"].startswith("snap_"))
            self.assertEqual(snap["storage"]["version_id"], "uuid-ver-1")
            self.assertEqual(snap["knowledge_base"]["document_id"], "doc-888")
            self.assertEqual(snap["metadata"]["document_title"], "IT采购通用条款")

            # 3. 列出快照
            snapshots = list_consistency_snapshots()
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0]["snapshot_id"], snap["snapshot_id"])

            # 4. 根据 ID 精准查询
            found = get_snapshot_by_id(snap["snapshot_id"])
            self.assertIsNotNone(found)
            self.assertEqual(found["file_name"], "Appendix A.docx")


if __name__ == "__main__":
    unittest.main()
