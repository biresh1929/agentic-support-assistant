def test_health_reports_its_data_and_clock(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["orders_loaded"] is True
    assert body["policy_loaded"] is True
    # Pinned, not the wall clock -- the fixture set depends on it.
    assert body["reference_date"] == "2026-07-29"
