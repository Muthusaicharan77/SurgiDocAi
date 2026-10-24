from surgidoc.agent.graph import MAX_DRAFTS, node_drop, route_after_review, route_after_verify


def results(*statuses):
    return [{"claim_id": f"c{i}", "status": s, "basis": "none", "reasons": ["r"]}
            for i, s in enumerate(statuses)]


def test_verify_routes_to_retry_until_out_of_drafts():
    assert route_after_verify({"results": results("supported"), "drafts": 1}) == "drop_unsupported"
    assert route_after_verify({"results": results("invalid"), "drafts": 1}) == "draft_report"
    assert route_after_verify({"results": results("invalid"),
                               "drafts": MAX_DRAFTS}) == "drop_unsupported"


def test_drop_keeps_only_supported_claims():
    state = {"claims": [{"id": "c0", "text": "a"}, {"id": "c1", "text": "b"}],
             "results": results("supported", "unsupported")}
    out = node_drop(state)
    assert [c["id"] for c in out["claims"]] == ["c0"]
    assert out["dropped"][0]["claim"]["id"] == "c1"


def test_review_routes():
    assert route_after_review({"review": {"approve": True}}) == "finalize"
    assert route_after_review({"review": {"approve": False}}) == "__end__"
