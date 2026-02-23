"""Tests for BlackRoad Ad Manager."""

import os
import sys
import tempfile
import pytest
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ad_manager import AdManager  # noqa: E402


@pytest.fixture
def mgr():
    """Return an in-memory AdManager."""
    return AdManager(db_path=":memory:")


@pytest.fixture
def campaign(mgr):
    return mgr.create_campaign(
        name="Test Campaign",
        objective="conversions",
        budget=5000.0,
        start_date=(date.today() - timedelta(days=10)).isoformat(),
        end_date=(date.today() + timedelta(days=20)).isoformat(),
    )


@pytest.fixture
def ad_set(mgr, campaign):
    return mgr.create_ad_set(
        campaign_id=campaign.id,
        name="Test Ad Set",
        targeting='{"age": "25-45"}',
        daily_budget=200.0,
        bid_strategy="CPC",
        bid_amount=2.50,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_campaign(mgr):
    """Campaign created with correct budget."""
    c = mgr.create_campaign(
        name="My Campaign",
        objective="awareness",
        budget=1000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    assert c.name == "My Campaign"
    assert c.budget_total == 1000.0
    assert c.budget_spent == 0.0
    assert c.status == "active"
    assert c.objective == "awareness"

    # Verify it's in the DB
    row = mgr.conn.execute(
        "SELECT * FROM campaigns WHERE id = ?", (c.id,)
    ).fetchone()
    assert row is not None
    assert row["budget_total"] == 1000.0


def test_record_performance(mgr, campaign):
    """Record metrics; verify CTR and ROAS calculations."""
    mgr.record_performance(
        entity_type="campaign",
        entity_id=campaign.id,
        date="2025-06-01",
        impressions=10000,
        clicks=200,
        conversions=20,
        spend=400.0,
        revenue=1600.0,
    )
    metrics = mgr.get_campaign_metrics(campaign.id)

    assert metrics["impressions"] == 10000
    assert metrics["clicks"] == 200
    assert metrics["conversions"] == 20
    assert abs(metrics["CTR"] - 0.02) < 1e-6       # 200/10000
    assert abs(metrics["ROAS"] - 4.0) < 1e-4       # 1600/400
    assert abs(metrics["CPC"] - 2.0) < 1e-4        # 400/200
    assert abs(metrics["CPA"] - 20.0) < 1e-4       # 400/20


def test_budget_pacing(mgr, campaign):
    """Record spend over multiple days; check pacing ratio."""
    for i in range(5):
        d = (date.today() - timedelta(days=5 - i)).isoformat()
        mgr.record_performance(
            entity_type="campaign",
            entity_id=campaign.id,
            date=d,
            impressions=1000,
            clicks=50,
            conversions=5,
            spend=100.0,
            revenue=300.0,
        )

    pacing = mgr.get_budget_pacing(campaign.id)

    assert pacing["budget_total"] == 5000.0
    assert pacing["budget_spent"] == pytest.approx(500.0, abs=1.0)
    assert "pacing_ratio" in pacing
    assert pacing["pacing_ratio"] >= 0.0
    assert "status" in pacing
    assert pacing["status"] in ("on_track", "under_pacing", "over_pacing")


def test_optimize_bids(mgr, campaign, ad_set):
    """High CPA scenario → bid reduction suggested."""
    # target_cpa = 5.0; record spend=200, conversions=2 → CPA=100 (very high)
    mgr.record_performance(
        entity_type="ad_set",
        entity_id=ad_set.id,
        date="2025-06-01",
        impressions=5000,
        clicks=100,
        conversions=2,
        spend=200.0,
        revenue=50.0,
    )
    suggestions = mgr.optimize_bids(campaign.id, target_cpa=5.0)

    assert len(suggestions) == 1
    s = suggestions[0]
    assert s["action"] == "reduce_bid"
    assert s["adjustment_pct"] < 0
    assert s["suggested_bid"] < s["current_bid"]


def test_rank_creatives(mgr, ad_set):
    """3 creatives with different CTRs; verify ranking order."""
    creatives = []
    for i, (impr, clicks, conv, spend, rev) in enumerate(
        [(10000, 500, 50, 250.0, 1500.0),   # high performer
         (10000, 100, 5, 50.0, 150.0),      # medium
         (10000, 20, 1, 10.0, 10.0)]        # low performer
    ):
        c = mgr.create_creative(
            ad_set_id=ad_set.id,
            name=f"Creative {i+1}",
            headline=f"Headline {i+1}",
            body=f"Body text {i+1}",
            cta="Buy Now",
        )
        mgr.record_performance(
            entity_type="creative",
            entity_id=c.id,
            date="2025-06-01",
            impressions=impr,
            clicks=clicks,
            conversions=conv,
            spend=spend,
            revenue=rev,
        )
        creatives.append(c)

    ranked = mgr.rank_creatives(ad_set.id)

    assert len(ranked) == 3
    assert ranked[0]["rank"] == 1
    assert ranked[0]["CTR"] >= ranked[1]["CTR"]
    assert ranked[1]["CTR"] >= ranked[2]["CTR"]
    # Best creative should be Creative 1
    assert ranked[0]["name"] == "Creative 1"


def test_generate_report(mgr, campaign, ad_set):
    """Full report contains all required keys."""
    mgr.record_performance(
        entity_type="campaign",
        entity_id=campaign.id,
        date="2025-06-01",
        impressions=8000,
        clicks=160,
        conversions=16,
        spend=320.0,
        revenue=960.0,
    )
    report = mgr.generate_report(campaign.id)

    required_keys = {
        "campaign", "metrics", "budget_pacing",
        "top_creatives", "bid_suggestions", "recommendations",
    }
    assert required_keys.issubset(report.keys()), (
        f"Missing keys: {required_keys - report.keys()}"
    )
    assert "CTR" in report["metrics"]
    assert "ROAS" in report["metrics"]
    assert "pacing_ratio" in report["budget_pacing"]
    assert isinstance(report["recommendations"], list)
    assert len(report["recommendations"]) >= 1
