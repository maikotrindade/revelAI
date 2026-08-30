"""``revelai serve``: the local run server.

Three things are being defended here, and they are not the same thing.

*The folder check has to be real.* "Only images" is a promise the server makes
before it runs, and a promise that trusts the file extension is not one. Every
rejection path has a test, and so does the case where the name says JPEG and
the bytes say otherwise.

*The guards have to hold.* A server listening on a loopback port is reachable
by every page the person browsing has open. Origin and Host are checked before
the route is looked at, and the tests drive them directly rather than trusting
that they are wired in.

*The two decisions that are the operator's stay the operator's.* No request
body can select a hosted backend or turn on the operations that invent detail.

Nothing here touches the network: the server is bound to 127.0.0.1 on a port
the operating system picks, which is a socket on this machine and not a
connection to anywhere.
"""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from http.client import HTTPConnection

import cv2
import pytest

import synth
from revelai.server import ServeConfig, build_server
from revelai.server.config import ServeConfigError, is_loopback_host
from revelai.server.folder import check_listing, safe_name, verify_image_bytes
from revelai.server.jobs import JobError, JobStore, build_result_zip, stage_sequence

LIMITS = {"max_files": 10, "max_file_bytes": 4 << 20, "max_total_bytes": 16 << 20}


def png_bytes(page) -> bytes:
    ok, buffer = cv2.imencode(".png", page.image)
    assert ok
    return buffer.tobytes()


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class Client:
    """A minimal HTTP client, so the tests exercise the wire and not a shim."""

    def __init__(self, port: int) -> None:
        self.port = port

    def call(self, method, path, body=None, headers=None, raw=False):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=60)
        head = dict(headers or {})
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        if data is not None and "Content-Type" not in head:
            head["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=data, headers=head)
            response = conn.getresponse()
            payload = response.read()
            if raw:
                return response.status, payload, dict(response.getheaders())
            return response.status, json.loads(payload or b"{}")
        finally:
            conn.close()

    # Convenience wrappers, because every test says these words.

    def create(self, files, stage="split", options=None):
        return self.call(
            "POST", "/api/jobs", {"stage": stage, "files": files, "options": options or {}}
        )

    def upload(self, job_id, index, data):
        return self.call(
            "PUT",
            f"/api/jobs/{job_id}/files/{index}",
            data,
            {"Content-Type": "application/octet-stream"},
        )

    def run_to_completion(self, job_id, timeout=120.0):
        status, _ = self.call("POST", f"/api/jobs/{job_id}/start", b"")
        assert status == 202
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, snapshot = self.call("GET", f"/api/jobs/{job_id}")
            if snapshot["state"] in ("done", "failed", "cancelled"):
                return snapshot
            time.sleep(0.05)
        raise AssertionError("the job never finished")


def start_server(**overrides):
    config = ServeConfig(port=0, **overrides)
    server, store = build_server(config)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return server, store, Client(server.server_address[1])


@pytest.fixture
def running():
    server, store, client = start_server(**{"max_files": 10})
    yield client
    server.shutdown()
    server.server_close()
    store.close()


@pytest.fixture(scope="module")
def two_pages() -> dict[str, bytes]:
    return {
        "page_1.png": png_bytes(synth.tiny_page(seed=70, photos=2)),
        "page_2.png": png_bytes(synth.tiny_page(seed=71, photos=1)),
    }


def listing_for(pages: dict[str, bytes]) -> list[dict]:
    return [{"name": name, "size": len(data)} for name, data in pages.items()]


def send_all(client: Client, job_id: str, pages: dict[str, bytes]) -> None:
    for index, data in enumerate(pages.values()):
        status, _ = client.upload(job_id, index, data)
        assert status == 200


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class TestConfig:
    def test_loopback_origins_are_always_allowed(self):
        config = ServeConfig(allow_origins=())
        for origin in (
            "http://localhost:3000",
            "http://127.0.0.1:8765",
            "https://localhost",
        ):
            assert config.origin_allowed(origin), origin

    def test_a_stranger_is_not(self):
        assert not ServeConfig().origin_allowed("https://evil.example")

    def test_the_published_site_is_allowed_out_of_the_box(self):
        # Otherwise the hosted run page needs a flag nobody would guess at.
        assert ServeConfig().origin_allowed("https://maikotrindade.com")

    def test_a_missing_origin_is_allowed(self):
        # Same-origin GETs from --ui-dir do not send one.
        assert ServeConfig().origin_allowed(None)
        assert ServeConfig().origin_allowed("null")

    def test_extra_origins_can_be_named(self):
        config = ServeConfig(allow_origins=("https://example.test",))
        assert config.origin_allowed("https://example.test")
        assert not config.origin_allowed("https://other.test")

    def test_an_origin_must_look_like_one(self):
        with pytest.raises(ServeConfigError, match="not an origin"):
            ServeConfig(allow_origins=("example.test/path",))

    @pytest.mark.parametrize(
        "host,expected",
        [
            ("localhost:8765", True),
            ("127.0.0.1:8765", True),
            ("127.0.0.1", True),
            ("[::1]:8765", True),
            ("attacker.example", False),
            ("revelai.attacker.example:8765", False),
            ("", False),
        ],
    )
    def test_host_header_recognition(self, host, expected):
        assert is_loopback_host(host) is expected

    def test_a_host_bound_elsewhere_trusts_its_own_name(self):
        # Somebody who bound to a LAN address owns the name it answers on.
        assert ServeConfig(host="0.0.0.0").host_allowed("photos.local")

    def test_port_range(self):
        with pytest.raises(ServeConfigError):
            ServeConfig(port=99999)
        assert ServeConfig(port=0).port == 0

    def test_describe_names_what_was_enabled(self):
        text = ServeConfig(allow_generative=True).describe()
        assert "local only" in text
        assert "allowed" in text


# --------------------------------------------------------------------------
# The folder check
# --------------------------------------------------------------------------


class TestFolderCheck:
    def test_a_folder_of_images_is_accepted(self):
        check = check_listing(
            [{"name": "a.jpg", "size": 100}, {"name": "b.PNG", "size": 200}], **LIMITS
        )
        assert check.ok
        assert [e["name"] for e in check.accepted] == ["a.jpg", "b.PNG"]
        assert check.total_bytes == 300

    def test_an_empty_folder_is_refused(self):
        assert not check_listing([], **LIMITS).ok

    def test_a_non_image_is_named_in_the_refusal(self):
        check = check_listing(
            [{"name": "a.jpg", "size": 10}, {"name": "notes.txt", "size": 10}], **LIMITS
        )
        assert not check.ok
        assert check.problems[0].name == "notes.txt"
        assert ".txt" in check.problems[0].reason

    def test_a_file_with_no_extension_is_refused(self):
        check = check_listing([{"name": "README", "size": 10}], **LIMITS)
        assert "no extension" in check.problems[0].reason

    def test_a_subfolder_is_refused_rather_than_flattened(self):
        check = check_listing(
            [{"name": "a.jpg", "relativePath": "pages/nested/a.jpg", "size": 10}], **LIMITS
        )
        assert not check.ok
        assert "subfolder" in check.problems[0].reason

    def test_the_chosen_folder_itself_is_not_a_subfolder(self):
        # The picker reports "pages/a.jpg" for a file directly in the folder.
        check = check_listing(
            [{"name": "a.jpg", "relativePath": "pages/a.jpg", "size": 10}], **LIMITS
        )
        assert check.ok

    def test_desktop_clutter_is_skipped_not_refused(self):
        check = check_listing(
            [{"name": "a.jpg", "size": 10}, {"name": ".DS_Store", "size": 6}], **LIMITS
        )
        assert check.ok
        assert check.skipped[0].name == ".DS_Store"

    def test_two_files_with_one_name(self):
        check = check_listing(
            [{"name": "a.jpg", "size": 10}, {"name": "A.JPG", "size": 10}], **LIMITS
        )
        assert "share a name" in check.problems[0].reason

    def test_an_empty_file(self):
        assert "empty" in check_listing([{"name": "a.jpg", "size": 0}], **LIMITS).problems[0].reason

    def test_too_many_files(self):
        entries = [{"name": f"p{i}.jpg", "size": 10} for i in range(LIMITS["max_files"] + 1)]
        assert "over the limit" in check_listing(entries, **LIMITS).problems[0].reason

    def test_too_many_bytes_in_total(self):
        entries = [{"name": f"p{i}.jpg", "size": 3 << 20} for i in range(6)]
        assert "in total" in check_listing(entries, **LIMITS).problems[0].reason

    def test_one_file_too_large(self):
        check = check_listing([{"name": "a.jpg", "size": 8 << 20}], **LIMITS)
        assert "limit for one file" in check.problems[0].reason

    @pytest.mark.parametrize(
        "name", ["../../etc/passwd", "a/b.jpg", "a\\b.jpg", "C:evil.jpg", "bad\x00.jpg", "", ".."]
    )
    def test_names_that_are_not_names(self, name):
        assert safe_name(name) is None

    def test_a_traversal_attempt_is_refused_by_the_listing_check(self):
        check = check_listing([{"name": "../../evil.jpg", "size": 10}], **LIMITS)
        assert not check.ok
        assert "will write" in check.problems[0].reason


class TestVerifyBytes:
    def test_a_real_image_passes(self):
        assert verify_image_bytes(png_bytes(synth.tiny_page(seed=5, photos=1))) is None

    def test_a_text_file_called_jpg_does_not(self):
        # The whole reason the check happens twice: the name was never evidence.
        assert verify_image_bytes(b"this is not an image at all") is not None

    def test_an_empty_file_does_not(self):
        assert verify_image_bytes(b"") == "the file is empty"

    def test_a_truncated_png_does_not(self):
        data = png_bytes(synth.tiny_page(seed=6, photos=1))
        assert verify_image_bytes(data[: len(data) // 3]) is not None


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


class TestStages:
    def test_run_means_both(self):
        assert stage_sequence("run") == ("split", "enhance")

    def test_either_stage_alone(self):
        assert stage_sequence("split") == ("split",)
        assert stage_sequence("enhance") == ("enhance",)

    def test_anything_else_is_an_error(self):
        with pytest.raises(JobError, match="unknown stage"):
            stage_sequence("sharpen")


class TestResultZip:
    def test_only_pngs_go_in(self, tmp_path):
        source = tmp_path / "out"
        source.mkdir()
        (source / "photo_00000001.png").write_bytes(b"\x89PNG\r\n\x1a\nreal enough")
        (source / "notes.txt").write_bytes(b"not a photograph")
        (source / "sneaky.png").write_bytes(b"not a png despite the name")

        info = build_result_zip(source, tmp_path / "out.zip")
        assert info["names"] == ["photo_00000001.png"]

    def test_a_restored_jpeg_is_named_for_what_it_is(self, tmp_path):
        # enhance keeps the input filename, and everything RevelAI writes is a
        # PNG. A zip entry called .jpg holding PNG bytes would break importers.
        source = tmp_path / "out"
        source.mkdir()
        (source / "holiday.jpg").write_bytes(b"\x89PNG\r\n\x1a\ncontent")
        assert build_result_zip(source, tmp_path / "z.zip")["names"] == ["holiday.png"]

    def test_nothing_to_zip_is_an_error_with_a_reason(self, tmp_path):
        empty = tmp_path / "out"
        empty.mkdir()
        with pytest.raises(JobError, match="nothing to download"):
            build_result_zip(empty, tmp_path / "z.zip")


class TestJobStore:
    def test_discard_takes_the_files_with_it(self):
        store = JobStore()
        job = store.create(stage="split", options={}, expected=[{"name": "a.png", "size": 1}])
        root = job.root
        assert root.exists()
        assert store.discard(job.id)
        assert not root.exists()
        assert store.get(job.id) is None
        store.close()

    def test_closing_removes_the_whole_temporary_tree(self):
        store = JobStore()
        store.create(stage="split", options={}, expected=[{"name": "a.png", "size": 1}])
        base = store.base
        store.close()
        assert not base.exists()

    def test_a_job_will_not_start_before_every_file_arrives(self):
        store = JobStore()
        job = store.create(
            stage="split",
            options={},
            expected=[{"name": "a.png", "size": 1}, {"name": "b.png", "size": 1}],
        )
        with pytest.raises(JobError, match="never arrived"):
            store.start(job, backend="local", jobs=1)
        store.close()


# --------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------


class TestHealth:
    def test_it_says_what_it_will_allow(self, running):
        status, body = running.call("GET", "/api/health")
        assert status == 200
        assert body["product"] == "RevelAI"
        assert body["backend"] == "local"
        assert body["uploadsPhotographs"] is False
        assert body["allowGenerative"] is False
        assert ".jpg" in body["acceptedSuffixes"]

    def test_an_unknown_route_is_not_a_traceback(self, running):
        status, body = running.call("GET", "/api/nope")
        assert status == 404
        assert "no route" in body["error"]


class TestGuards:
    def test_a_stranger_origin_is_refused(self, running):
        status, body = running.call("GET", "/api/health", headers={"Origin": "https://evil.test"})
        assert status == 403
        assert body["code"] == "origin_not_allowed"

    def test_the_published_site_is_not(self, running):
        status, _ = running.call(
            "GET", "/api/health", headers={"Origin": "https://maikotrindade.com"}
        )
        assert status == 200

    def test_a_rebound_hostname_is_refused(self, running):
        status, body = running.call("GET", "/api/health", headers={"Host": "attacker.example"})
        assert status == 421
        assert "DNS rebinding" in body["error"]

    def test_preflight_answers_the_private_network_question(self, running):
        status, _, headers = running.call(
            "OPTIONS",
            "/api/jobs",
            raw=True,
            headers={
                "Origin": "https://maikotrindade.com",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert status == 204
        assert headers["Access-Control-Allow-Private-Network"] == "true"

    def test_a_body_without_a_length_is_refused(self, running):
        conn = HTTPConnection("127.0.0.1", running.port, timeout=30)
        conn.putrequest("POST", "/api/jobs")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(b"0\r\n\r\n")
        assert conn.getresponse().status == 411
        conn.close()


class TestOperatorOnlyDecisions:
    def test_a_request_cannot_choose_the_backend(self, running):
        status, body = running.call(
            "POST",
            "/api/jobs",
            {
                "stage": "enhance",
                "files": [{"name": "a.png", "size": 9}],
                "options": {"backend": "hosted"},
            },
        )
        assert status == 400
        assert body["code"] == "backend_not_selectable"

    @pytest.mark.parametrize("operation", ["faces", "colorize"])
    def test_generative_operations_are_refused_by_default(self, running, operation):
        status, body = running.create(
            [{"name": "a.png", "size": 9}], stage="enhance", options={operation: True}
        )
        assert status == 403
        assert body["code"] == "generative_refused"

    def test_they_are_allowed_when_the_operator_said_so(self, two_pages):
        server, store, client = start_server(allow_generative=True)
        try:
            status, _ = client.create(
                listing_for(two_pages), stage="enhance", options={"faces": True}
            )
            assert status == 201
        finally:
            server.shutdown()
            server.server_close()
            store.close()


class TestFolderEndpoint:
    def test_it_answers_before_anything_is_uploaded(self, running):
        status, body = running.call(
            "POST",
            "/api/folder/check",
            {"files": [{"name": "a.jpg", "size": 10}, {"name": "notes.txt", "size": 10}]},
        )
        assert status == 200
        assert body["ok"] is False
        assert body["problems"][0]["name"] == "notes.txt"

    def test_a_good_folder_comes_back_clean(self, running):
        _, body = running.call(
            "POST", "/api/folder/check", {"files": [{"name": "a.jpg", "size": 10}]}
        )
        assert body["ok"] is True
        assert body["problems"] == []

    def test_a_malformed_body_is_a_sentence(self, running):
        status, body = running.call("POST", "/api/folder/check", b"{oh dear")
        assert status == 400
        assert "not valid JSON" in body["error"]


class TestCreatingAJob:
    def test_a_bad_folder_is_refused_with_every_reason(self, running):
        status, body = running.create(
            [{"name": "a.jpg", "size": 10}, {"name": "b.pdf", "size": 10}]
        )
        assert status == 400
        assert body["code"] == "folder_rejected"
        assert [p["name"] for p in body["problems"]] == ["b.pdf"]

    def test_an_unknown_stage_is_refused(self, running):
        status, body = running.create([{"name": "a.jpg", "size": 10}], stage="polish")
        assert status == 400
        assert "unknown stage" in body["error"]

    def test_a_file_whose_bytes_are_not_an_image_is_refused_on_upload(self, running):
        _, job = running.create([{"name": "a.jpg", "size": 30}])
        status, body = running.upload(job["id"], 0, b"plain text pretending to be a photograph")
        assert status == 400
        assert body["code"] == "not_an_image"

    def test_an_index_outside_the_listing_is_refused(self, running, two_pages):
        _, job = running.create(listing_for(two_pages))
        status, _ = running.upload(job["id"], 9, next(iter(two_pages.values())))
        assert status == 400

    def test_starting_early_is_a_conflict_not_a_crash(self, running, two_pages):
        _, job = running.create(listing_for(two_pages))
        status, body = running.call("POST", f"/api/jobs/{job['id']}/start", b"")
        assert status == 409
        assert "never arrived" in body["error"]

    def test_downloading_before_it_is_done(self, running, two_pages):
        _, job = running.create(listing_for(two_pages))
        status, _ = running.call("GET", f"/api/jobs/{job['id']}/result.zip")
        assert status == 409

    def test_an_unknown_job(self, running):
        status, _ = running.call("GET", f"/api/jobs/{'0' * 32}")
        assert status == 404


class TestAFullRun:
    """The path the website actually walks, end to end."""

    @pytest.fixture(scope="class")
    @classmethod
    def finished(cls, two_pages):
        server, store, client = start_server()
        try:
            status, job = client.create(listing_for(two_pages), stage="split")
            assert status == 201
            send_all(client, job["id"], two_pages)
            snapshot = client.run_to_completion(job["id"])
            _, data, headers = client.call("GET", f"/api/jobs/{job['id']}/result.zip", raw=True)
            yield snapshot, data, headers, client, job["id"]
        finally:
            server.shutdown()
            server.server_close()
            store.close()

    def test_it_finishes(self, finished):
        snapshot, *_ = finished
        assert snapshot["state"] == "done"

    def test_the_summary_counts_what_happened(self, finished):
        snapshot, *_ = finished
        assert snapshot["summary"]["split"]["pagesRead"] == 2
        assert snapshot["summary"]["split"]["photographs"] == 3

    def test_progress_arrives_page_by_page(self, finished):
        _, _, _, client, job_id = finished
        _, body = client.call("GET", f"/api/jobs/{job_id}?after=0")
        steps = [e for e in body["events"] if e["type"] == "progress" and e["done"]]
        # One event per page, not one jump from nothing to everything.
        assert [e["done"] for e in steps] == [1, 2]
        assert [e["percent"] for e in steps] == [50.0, 100.0]

    def test_the_zip_holds_photographs_and_nothing_else(self, finished):
        _, data, headers, _, _ = finished
        assert headers["Content-Type"] == "application/zip"
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        assert names == ["photo_00000001.png", "photo_00000002.png", "photo_00000003.png"]

    def test_every_entry_really_is_a_png(self, finished):
        _, data, _, _, _ = finished
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            assert archive.read(name).startswith(b"\x89PNG\r\n\x1a\n")

    def test_discarding_removes_the_photographs_from_disk(self, finished):
        _, _, _, client, job_id = finished
        status, body = client.call("DELETE", f"/api/jobs/{job_id}")
        assert status == 200 and body["discarded"] is True
        status, _ = client.call("GET", f"/api/jobs/{job_id}")
        assert status == 404


class TestBothStages:
    def test_run_splits_then_restores_and_reports_two_stages(self, two_pages):
        server, store, client = start_server()
        try:
            pages = {"page_1.png": two_pages["page_1.png"]}
            _, job = client.create(listing_for(pages), stage="run", options={"color": True})
            send_all(client, job["id"], pages)
            snapshot = client.run_to_completion(job["id"])
            assert snapshot["state"] == "done"
            summary = snapshot["summary"]
            assert summary["stages"] == ["split", "enhance"]
            assert summary["enhance"]["restored"] == summary["split"]["photographs"]
            assert summary["enhance"]["backend"] == "local"

            _, body = client.call("GET", f"/api/jobs/{job['id']}?after=0")
            stages = [e["stage"] for e in body["events"] if e["type"] == "progress"]
            assert stages.count("split") and stages.count("enhance")
            # The bar never goes backwards across the stage boundary.
            percents = [e["percent"] for e in body["events"] if e["type"] == "progress"]
            assert percents == sorted(percents)
        finally:
            server.shutdown()
            server.server_close()
            store.close()


class TestTheCommand:
    """`revelai serve` itself. No user mistake may surface a traceback."""

    def test_a_port_already_in_use_is_a_sentence(self, capsys):
        from revelai.cli import EXIT_ERROR, main

        server, store, client = start_server()
        try:
            code = main(["serve", "--port", str(client.port)])
            assert code == EXIT_ERROR
            error = capsys.readouterr().err
            assert "already listening" in error
            assert "Traceback" not in error
        finally:
            server.shutdown()
            server.server_close()
            store.close()

    def test_a_refused_origin_is_a_sentence_not_a_traceback(self, capsys):
        from revelai.cli import EXIT_ERROR, main

        assert main(["serve", "--allow-origin", "not-an-origin"]) == EXIT_ERROR
        assert "not an origin" in capsys.readouterr().err

    def test_serve_is_in_the_help(self, capsys):
        from revelai.cli import main

        with pytest.raises(SystemExit):
            main(["--help"])
        assert "serve" in capsys.readouterr().out


class TestStaticUI:
    def test_without_ui_dir_it_says_so_rather_than_404ing_blankly(self, running):
        status, body = running.call("GET", "/")
        assert status == 404
        assert "run page" in body["error"]

    def test_with_ui_dir_it_serves_the_page(self, tmp_path):
        (tmp_path / "index.html").write_text("<h1>RevelAI</h1>", encoding="utf-8")
        server, store, client = start_server(ui_dir=str(tmp_path))
        try:
            status, data, headers = client.call("GET", "/", raw=True)
            assert status == 200
            assert b"RevelAI" in data
            assert headers["Content-Type"] == "text/html"
        finally:
            server.shutdown()
            server.server_close()
            store.close()

    def test_it_will_not_serve_a_file_outside_the_ui_folder(self, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("not yours", encoding="utf-8")
        ui = tmp_path / "ui"
        ui.mkdir()
        (ui / "index.html").write_text("ok", encoding="utf-8")
        server, store, client = start_server(ui_dir=str(ui))
        try:
            status, data, _ = client.call("GET", "/../secret.txt", raw=True)
            assert b"not yours" not in data
            assert status in (403, 404)
        finally:
            server.shutdown()
            server.server_close()
            store.close()
