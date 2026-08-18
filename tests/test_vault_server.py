"""Unit tests for vault_server.py."""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from vault_server import app


class TestVaultServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "minio-contract-vault"})

    @patch("vault_server.upload_file_to_vault")
    def test_upload_endpoint(self, mock_upload):
        mock_upload.return_value = {
            "bucket": "contracts-vault",
            "object_name": "contract.docx",
            "version_id": "v-1",
            "presigned_url": "http://localhost:9005/link",
            "success": True,
        }

        files = {"file": ("contract.docx", b"fake docx binary data", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        data = {"contract_package": "baseline_2_4"}
        response = self.client.post("/api/vault/upload", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["version_id"], "v-1")
        mock_upload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
