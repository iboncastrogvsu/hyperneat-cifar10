import json
import time
from datetime import datetime, timezone
from google.cloud import storage

def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

class GCSLogger:
    """
    Uploads per-generation results and overall history to a GCS bucket.
    Uses Application Default Credentials (ADC) for authentication.
    No JSON key required.
    """

    def __init__(self, bucket_name, prefix="hyperneat"):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.rstrip("/")

    def _object_name(self, path):
        return f"{self.prefix}/{path}"

    def upload_generation(self, gen_entry):
        """Upload a generation result to GCS as JSON."""
        gen_idx = gen_entry.get("gen", int(time.time()))
        gen_entry = dict(gen_entry)
        gen_entry["timestamp"] = _utcnow_iso()

        filename = f"history/gen_{int(gen_idx):04d}.json"
        blob = self.bucket.blob(self._object_name(filename))
        blob.upload_from_string(json.dumps(gen_entry), content_type="application/json")

    def upload_json(self, filename, data):
        """Upload arbitrary JSON data to the bucket."""
        blob = self.bucket.blob(self._object_name(filename))
        payload = {
            "timestamp": _utcnow_iso(),
            "data": data
        }
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
