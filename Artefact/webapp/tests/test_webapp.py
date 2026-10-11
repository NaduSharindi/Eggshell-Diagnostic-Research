"""
test_webapp.py
Tests for the Flask web layer: routing, form handling, the JSON API, downloads and error paths.

Run locally:   pytest Artefact/webapp/tests -v
Run in CI:     the integration job in .github/workflows/main.yaml
"""
from __future__ import annotations

from conftest import upload


# ------------------------------------------------------------------ pages
class TestPages:
    def test_root_redirects_to_inspect(self, client):
        r = client.get("/")
        assert r.status_code == 302 and "/inspect" in r.headers["Location"]

    def test_inspect_page(self, client):
        r = client.get("/inspect")
        assert r.status_code == 200
        assert b"Measured performance" in r.data

    def test_batch_page(self, client):
        assert client.get("/batch").status_code == 200

    def test_model_card_states_limitations(self, client):
        r = client.get("/model")
        assert r.status_code == 200
        assert b"Known limitations" in r.data
        assert b"not validated against hatch outcomes" in r.data.lower() or \
               b"never validated against hatch outcomes" in r.data.lower()

    def test_no_emoji_in_pages(self, client):
        for path in ("/inspect", "/batch", "/model"):
            text = client.get(path).get_data(as_text=True)
            assert not any(0x1F300 <= ord(ch) <= 0x1FAFF for ch in text), path

    def test_unknown_page_is_404(self, client):
        assert client.get("/does-not-exist").status_code == 404


# ------------------------------------------------------------------ inspect
class TestInspect:
    def test_cracked_egg_full_diagnosis(self, client, egg_jpg):
        r = client.post("/inspect", data={"image": upload(egg_jpg(), "E283.jpg"),
                                          "age_class": "Peak production"},
                        content_type="multipart/form-data")
        body = r.get_data(as_text=True)
        assert r.status_code == 200
        assert "Micro-crack detected" in body
        assert "Probable root cause" in body
        assert "data:image/jpeg;base64," in body          # overlay rendered in memory

    def test_missing_file_is_rejected(self, client):
        r = client.post("/inspect", data={}, content_type="multipart/form-data")
        assert r.status_code == 400

    def test_wrong_extension_is_rejected(self, client):
        r = client.post("/inspect", data={"image": upload(b"x", "notes.txt")},
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_corrupt_image_does_not_crash(self, client):
        r = client.post("/inspect", data={"image": upload(b"not an image", "bad.jpg")},
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_age_class_is_validated(self, client, egg_jpg):
        """A forged age class must not reach the rule lookup."""
        r = client.post("/inspect", data={"image": upload(egg_jpg(), "a.jpg"),
                                          "age_class": "<script>x</script>"},
                        content_type="multipart/form-data")
        assert r.status_code == 200
        assert b"<script>x</script>" not in r.data


# ------------------------------------------------------------------ batch
class TestBatch:
    def _run(self, client, egg_jpg, n=5):
        files = [upload(egg_jpg(i), f"E{i:03d}.jpg") for i in range(n)]
        return client.post("/batch", data={"images": files, "age_class": "Late lay",
                                           "flock_id": "Siridara N10", "operator": "Tester",
                                           "notes": "belt speed raised"},
                           content_type="multipart/form-data")

    def test_batch_redirects_to_job(self, client, egg_jpg):
        r = self._run(client, egg_jpg)
        assert r.status_code == 302 and "job=" in r.headers["Location"]

    def test_batch_summary_and_actions(self, client, egg_jpg):
        r = self._run(client, egg_jpg)
        page = client.get(r.headers["Location"]).get_data(as_text=True)
        assert "Eggs inspected" in page
        assert "Recommended actions for this flock" in page
        assert "Per-egg results" in page

    def test_action_text_is_never_fragmented(self, client, egg_jpg):
        """Regression: an earlier build split advice at decimal points, e.g. '2%), and check'."""
        r = self._run(client, egg_jpg)
        page = client.get(r.headers["Location"]).get_data(as_text=True)
        assert "Actions for Star in Late lay (about 3.2% Ca)." in page

    def test_csv_and_report_downloads(self, client, egg_jpg):
        loc = self._run(client, egg_jpg).headers["Location"]
        job = loc.split("job=")[1]
        csv = client.get(f"/batch/{job}/download.csv")
        assert csv.status_code == 200 and csv.mimetype == "text/csv"
        assert "file,status,pattern" in csv.get_data(as_text=True)
        rep = client.get(f"/batch/{job}/report.md")
        text = rep.get_data(as_text=True)
        assert rep.status_code == 200
        assert "Siridara N10" in text and "belt speed raised" in text
        assert "has not been validated against hatch outcomes" in text

    def test_expired_job_is_handled(self, client):
        assert client.get("/batch/doesnotexist/download.csv").status_code == 404
        r = client.get("/batch?job=doesnotexist")
        assert r.status_code == 200 and b"expired" in r.data

    def test_batch_limit_enforced(self, client, egg_jpg, app):
        app.config["MAX_BATCH_FILES"] = 2
        r = self._run(client, egg_jpg, n=3)
        assert r.status_code == 400


# ------------------------------------------------------------------ API
class TestAPI:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200 and r.get_json()["status"] == "ok"

    def test_ready(self, client):
        j = client.get("/api/v1/ready").get_json()
        assert j["ready"] is True and j["models_loaded"] == 3

    def test_info_carries_threshold_rationale(self, client):
        j = client.get("/api/v1/info").get_json()
        assert j["decision"]["detection_threshold"] == 0.30
        assert "no cracked egg was missed" in j["decision"]["threshold_rationale"]
        assert len(j["limitations"]) >= 5

    def test_rules_complete(self, client):
        j = client.get("/api/v1/rules").get_json()
        assert len(j["rules"]) == 9

    def test_api_inspect(self, client, egg_jpg):
        r = client.post("/api/v1/inspect", data={"image": upload(egg_jpg(), "a.jpg"),
                                                 "age_class": "Early lay"},
                        content_type="multipart/form-data")
        j = r.get_json()
        assert r.status_code == 200
        assert j["result"]["status"] in ("Intact", "Micro_cracked")
        assert abs(sum(j["result"]["probabilities"].values()) - 1.0) < 1e-3

    def test_api_inspect_requires_image(self, client):
        r = client.post("/api/v1/inspect", data={}, content_type="multipart/form-data")
        assert r.status_code == 400 and r.get_json()["error"] == "bad_request"

    def test_api_batch(self, client, egg_jpg):
        files = [upload(egg_jpg(i), f"b{i}.jpg") for i in range(3)]
        r = client.post("/api/v1/batch", data={"images": files, "age_class": "Peak production"},
                        content_type="multipart/form-data")
        j = r.get_json()
        assert r.status_code == 200
        assert j["summary"]["n_readable"] == 3
        assert "flock_assessment" in j

    def test_api_errors_are_json(self, client):
        r = client.get("/api/v1/nope")
        assert r.status_code == 404 and r.is_json


# ------------------------------------------------------------------ settings
class TestSettings:
    def test_threshold_is_clamped(self, app):
        from app.utils import read_threshold
        with app.test_request_context():
            t, lc = read_threshold({"threshold": "5", "low_confidence": "-3"})
            assert t == 0.95 and lc == 0.0

    def test_default_threshold_is_the_evidenced_one(self, app):
        assert app.config["DETECTION_THRESHOLD"] == 0.30

    def test_job_store_is_bounded(self):
        from app.jobs import JobStore
        js = JobStore(ttl_seconds=3600, max_jobs=3)
        ids = [js.put({"n": i}) for i in range(6)]
        assert js.stats()["jobs"] == 3
        assert js.get(ids[0]) is None and js.get(ids[-1]) == {"n": 5}
