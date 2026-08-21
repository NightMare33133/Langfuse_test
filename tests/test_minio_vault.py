"""Unit tests for minio_vault.py module."""

import io
import unittest
from unittest.mock import MagicMock, patch

from minio_vault import (
    DEFAULT_CONTRACTS_BUCKET,
    ensure_bucket,
    get_minio_client,
    get_presigned_download_url,
    get_vault_file_bytes,
    list_vault_documents,
    upload_file_to_vault,
)


class TestMinIOVault(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()

    def test_ensure_bucket_creates_if_not_exists(self):
        self.mock_client.bucket_exists.return_value = False
        res = ensure_bucket(self.mock_client, "test-bucket", enable_versioning=True)
        self.assertTrue(res)
        self.mock_client.make_bucket.assert_called_once_with("test-bucket")
        self.mock_client.set_bucket_versioning.assert_called_once()

    def test_upload_file_to_vault(self):
        self.mock_client.bucket_exists.return_value = True
        mock_put_res = MagicMock()
        mock_put_res.version_id = "v-123"
        mock_put_res.etag = "etag-456"
        self.mock_client.put_object.return_value = mock_put_res
        self.mock_client.presigned_get_object.return_value = "http://localhost:9005/test.docx?token=xyz"

        result = upload_file_to_vault(
            "test.docx",
            b"test content bytes",
            bucket_name="contracts-vault",
            client=self.mock_client,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["object_name"], "test.docx")
        self.assertEqual(result["version_id"], "v-123")
        self.assertEqual(result["size"], 18)
        self.assertEqual(result["presigned_url"], "http://localhost:9005/test.docx?token=xyz")

    def test_get_presigned_download_url(self):
        self.mock_client.presigned_get_object.return_value = "http://localhost:9005/download"
        url = get_presigned_download_url("doc.docx", client=self.mock_client)
        self.assertEqual(url, "http://localhost:9005/download")

    def test_get_vault_file_bytes(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"contract binary data"
        self.mock_client.get_object.return_value = mock_resp

        data = get_vault_file_bytes("contract.docx", client=self.mock_client)
        self.assertEqual(data, b"contract binary data")
        mock_resp.close.assert_called_once()

    def test_save_and_get_cleaned_text(self):
        from minio_vault import save_cleaned_text, get_cleaned_text
        self.mock_client.bucket_exists.return_value = True
        mock_put_res = MagicMock()
        mock_put_res.version_id = "clean-v1"
        self.mock_client.put_object.return_value = mock_put_res

        res = save_cleaned_text("Appendix A.docx", "这是清洗后的纯净文本", client=self.mock_client)
        self.assertTrue(res["success"])
        self.assertEqual(res["object_name"], "Appendix A.docx.cleaned.txt")

        mock_resp = MagicMock()
        mock_resp.read.return_value = "这是清洗后的纯净文本".encode("utf-8")
        self.mock_client.get_object.return_value = mock_resp

        cleaned = get_cleaned_text("Appendix A.docx", client=self.mock_client)
        self.assertEqual(cleaned, "这是清洗后的纯净文本")


if __name__ == "__main__":
    unittest.main()
