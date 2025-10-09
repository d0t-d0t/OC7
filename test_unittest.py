import unittest
import api as api
import requests
from fastapi.testclient import TestClient
from api import app  # Adjust import based on your app structure

client = TestClient(app)

class Test_Global(unittest.TestCase):
    # unmute only if model is on github repo
    # def test_pipeline_import(self):
    #     self.assertIsNotNone(api.latest_model)


    def test_read_root():
        """Test the root endpoint returns expected response"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Hello": "World"}

if __name__ == '__main__':
    unittest.main()