"""Phase 8: trusted-proxy client IP, serving the built frontend, image preparation, and checks
on the Docker files."""

from fastapi.testclient import TestClient

from medquery.api.answer_cache import AnswerCache
from medquery.api.app import create_app
from medquery.api.limits import client_ip, trusted_networks
from medquery.api.prepare import EVALUATED_CORPUS_V2_SHA256, sha_note
from tests.test_api import GOLDEN, PASSAGES, FakeLLM, FakeRetriever

PROXIES = trusted_networks("10.0.0.0/8, 172.16.0.1")


def test_header_is_ignored_unless_the_peer_is_a_trusted_proxy():
    assert client_ip("203.0.113.9", "1.2.3.4", PROXIES) == "203.0.113.9"  # spoof attempt
    assert client_ip("10.1.2.3", None, PROXIES) == "10.1.2.3"
    assert client_ip("10.1.2.3", "1.2.3.4", []) == "10.1.2.3"  # no proxies configured


def test_trusted_proxy_gives_the_rightmost_untrusted_hop():
    assert client_ip("10.1.2.3", "198.51.100.7", PROXIES) == "198.51.100.7"
    # a client-supplied fake first hop is skipped: the proxies appended the real address
    assert client_ip("10.1.2.3", "6.6.6.6, 198.51.100.7, 172.16.0.1", PROXIES) == "198.51.100.7"
    assert client_ip("10.1.2.3", "10.9.9.9, 172.16.0.1", PROXIES) == "10.1.2.3"  # all trusted
    assert client_ip("10.1.2.3", "not-an-ip, 198.51.100.7", PROXIES) == "198.51.100.7"
    assert client_ip("10.1.2.3", "198.51.100.7, garbage", PROXIES) == "10.1.2.3"  # malformed
    assert trusted_networks("") == []


def app_for(tmp_path, **kw):
    return create_app(
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        chunks=PASSAGES,
        golden=GOLDEN,
        usage_path=tmp_path / "usage.json",
        answer_cache=AnswerCache(None, enabled=False),
        daily_cap=100,
        **kw,
    )


def test_rate_limit_is_per_real_client_behind_a_trusted_proxy(tmp_path):
    app = app_for(tmp_path, rate_per_min=1, trusted_proxies="10.0.0.0/8")
    c = TestClient(app, client=("10.0.0.5", 50000))

    def ask(ip):
        return c.post("/ask", json={"question": "q?"}, headers={"X-Forwarded-For": ip}).status_code

    assert [ask("198.51.100.1"), ask("198.51.100.2"), ask("198.51.100.1")] == [200, 200, 429]
    direct = TestClient(app_for(tmp_path, rate_per_min=1), client=("203.0.113.9", 50000))

    def spoof(ip):
        return direct.post(
            "/ask", json={"question": "q?"}, headers={"X-Forwarded-For": ip}
        ).status_code

    assert [spoof("1.1.1.1"), spoof("2.2.2.2")] == [200, 429]  # header ignored: one client


def test_built_frontend_is_served_at_root_after_the_api(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>MedQuery app</html>", encoding="utf-8")
    c = TestClient(app_for(tmp_path, static_dir=dist))
    assert "MedQuery app" in c.get("/").text
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/samples").status_code == 200
    no_frontend = TestClient(app_for(tmp_path, static_dir=tmp_path / "missing"))
    assert no_frontend.get("/").status_code == 404


def test_prepare_reports_whether_the_corpus_is_the_evaluated_one():
    assert "identical" in sha_note(EVALUATED_CORPUS_V2_SHA256)
    assert sha_note("0" * 64).startswith("WARNING")


def test_docker_files_pin_their_bases_and_hold_no_secrets():
    import re

    from medquery import paths

    dockerfile = (paths.ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (paths.ROOT / "compose.yaml").read_text(encoding="utf-8")
    images = re.findall(r"^FROM\s+(\S+)", dockerfile, re.M) + re.findall(
        r"--from=(\S+/\S+:\S+)", dockerfile
    )
    external = [i for i in images if ":" in i]  # stage names (app, corpus) have no tag
    assert external and all(re.search(r":\d+\.\d+\.\d+", i) for i in external), external
    for text in (dockerfile, compose):
        assert not re.search(r"(gsk_|sk-|nvapi-)[A-Za-z0-9]{8,}", text)
        assert "GROQ_API_KEY=" not in text
    ignored = (paths.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert {"data/", ".env", ".env.*", "**/node_modules"} <= set(ignored)
