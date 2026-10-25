import pytest

from surgidoc import tracing


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    """Tests never send traces, even when .env holds real Langfuse keys.

    tracing.enabled() reads .env, so without this a test that reaches a Claude call
    path would trace to the developer's Langfuse project. The tracing tests enable it
    themselves, with a fake client.
    """
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
                "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
