"""
BlackRoad Ad Manager - Comprehensive advertising campaign management system.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BLUE = "\033[0;34m"
MAGENTA = "\033[0;35m"
BOLD = "\033[1m"
NC = "\033[0m"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    id: str
    name: str
    status: str
    objective: str
    budget_total: float
    budget_spent: float
    start_date: str
    end_date: str
    created_at: str


@dataclass
class AdSet:
    id: str
    campaign_id: str
    name: str
    targeting: str
    daily_budget: float
    status: str
    bid_strategy: str
    bid_amount: float


@dataclass
class Creative:
    id: str
    ad_set_id: str
    name: str
    headline: str
    body: str
    cta: str
    image_url: str
    status: str


@dataclass
class PerformanceMetric:
    id: str
    entity_type: str
    entity_id: str
    date: str
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float


@dataclass
class BudgetPacing:
    id: str
    campaign_id: str
    date: str
    planned_spend: float
    actual_spend: float
    pacing_ratio: float


# ---------------------------------------------------------------------------
# AdManager
# ---------------------------------------------------------------------------

class AdManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.init_db()

    def init_db(self) -> None:
        """Initialize the 5 database tables."""
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                objective TEXT NOT NULL,
                budget_total REAL NOT NULL,
                budget_spent REAL DEFAULT 0.0,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ad_sets (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                name TEXT NOT NULL,
                targeting TEXT NOT NULL,
                daily_budget REAL NOT NULL,
                status TEXT DEFAULT 'active',
                bid_strategy TEXT NOT NULL,
                bid_amount REAL NOT NULL,
                FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
            );

            CREATE TABLE IF NOT EXISTS creatives (
                id TEXT PRIMARY KEY,
                ad_set_id TEXT NOT NULL,
                name TEXT NOT NULL,
                headline TEXT NOT NULL,
                body TEXT NOT NULL,
                cta TEXT NOT NULL,
                image_url TEXT,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (ad_set_id) REFERENCES ad_sets(id)
            );

            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                date TEXT NOT NULL,
                impressions INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                conversions INTEGER DEFAULT 0,
                spend REAL DEFAULT 0.0,
                revenue REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS budget_pacing (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                date TEXT NOT NULL,
                planned_spend REAL NOT NULL,
                actual_spend REAL NOT NULL,
                pacing_ratio REAL NOT NULL,
                FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
            );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_campaign(
        self,
        name: str,
        objective: str,
        budget: float,
        start_date: str,
        end_date: str,
    ) -> Campaign:
        """Insert a campaign and return a Campaign dataclass."""
        cid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """INSERT INTO campaigns
               (id, name, status, objective, budget_total, budget_spent,
                start_date, end_date, created_at)
               VALUES (?, ?, 'active', ?, ?, 0.0, ?, ?, ?)""",
            (cid, name, objective, budget, start_date, end_date, now),
        )
        self.conn.commit()
        return Campaign(
            id=cid,
            name=name,
            status="active",
            objective=objective,
            budget_total=budget,
            budget_spent=0.0,
            start_date=start_date,
            end_date=end_date,
            created_at=now,
        )

    def create_ad_set(
        self,
        campaign_id: str,
        name: str,
        targeting: str,
        daily_budget: float,
        bid_strategy: str,
        bid_amount: float,
    ) -> AdSet:
        """Insert an ad set and return an AdSet dataclass."""
        aid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO ad_sets
               (id, campaign_id, name, targeting, daily_budget, status,
                bid_strategy, bid_amount)
               VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
            (aid, campaign_id, name, targeting, daily_budget, bid_strategy, bid_amount),
        )
        self.conn.commit()
        return AdSet(
            id=aid,
            campaign_id=campaign_id,
            name=name,
            targeting=targeting,
            daily_budget=daily_budget,
            status="active",
            bid_strategy=bid_strategy,
            bid_amount=bid_amount,
        )

    def create_creative(
        self,
        ad_set_id: str,
        name: str,
        headline: str,
        body: str,
        cta: str,
        image_url: str = "",
    ) -> Creative:
        """Insert a creative and return a Creative dataclass."""
        cid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO creatives
               (id, ad_set_id, name, headline, body, cta, image_url, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'active')""",
            (cid, ad_set_id, name, headline, body, cta, image_url),
        )
        self.conn.commit()
        return Creative(
            id=cid,
            ad_set_id=ad_set_id,
            name=name,
            headline=headline,
            body=body,
            cta=cta,
            image_url=image_url,
            status="active",
        )

    def record_performance(
        self,
        entity_type: str,
        entity_id: str,
        date: str,
        impressions: int,
        clicks: int,
        conversions: int,
        spend: float,
        revenue: float,
    ) -> PerformanceMetric:
        """Insert a performance metric and update campaign budget_spent."""
        mid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO performance_metrics
               (id, entity_type, entity_id, date, impressions, clicks,
                conversions, spend, revenue)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, entity_type, entity_id, date, impressions, clicks,
             conversions, spend, revenue),
        )
        # Update campaign budget_spent if entity is a campaign
        if entity_type == "campaign":
            self.conn.execute(
                "UPDATE campaigns SET budget_spent = budget_spent + ? WHERE id = ?",
                (spend, entity_id),
            )
        # Also propagate ad_set spend up to campaign
        elif entity_type == "ad_set":
            row = self.conn.execute(
                "SELECT campaign_id FROM ad_sets WHERE id = ?", (entity_id,)
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE campaigns SET budget_spent = budget_spent + ? WHERE id = ?",
                    (spend, row["campaign_id"]),
                )
        elif entity_type == "creative":
            row = self.conn.execute(
                "SELECT ad_sets.campaign_id FROM creatives "
                "JOIN ad_sets ON creatives.ad_set_id = ad_sets.id "
                "WHERE creatives.id = ?",
                (entity_id,),
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE campaigns SET budget_spent = budget_spent + ? WHERE id = ?",
                    (spend, row["campaign_id"]),
                )
        self.conn.commit()
        return PerformanceMetric(
            id=mid,
            entity_type=entity_type,
            entity_id=entity_id,
            date=date,
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            spend=spend,
            revenue=revenue,
        )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_campaign_metrics(
        self,
        campaign_id: str,
        start_date: str = "",
        end_date: str = "",
    ) -> Dict[str, Any]:
        """Aggregate performance metrics for a campaign."""
        params: List[Any] = [campaign_id]
        date_filter = ""
        if start_date:
            date_filter += " AND date >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND date <= ?"
            params.append(end_date)

        row = self.conn.execute(
            f"""SELECT
                SUM(impressions) AS total_impressions,
                SUM(clicks)      AS total_clicks,
                SUM(conversions) AS total_conversions,
                SUM(spend)       AS total_spend,
                SUM(revenue)     AS total_revenue
            FROM performance_metrics
            WHERE entity_type = 'campaign' AND entity_id = ?{date_filter}""",
            params,
        ).fetchone()

        impressions = row["total_impressions"] or 0
        clicks = row["total_clicks"] or 0
        conversions = row["total_conversions"] or 0
        spend = row["total_spend"] or 0.0
        revenue = row["total_revenue"] or 0.0

        ctr = clicks / impressions if impressions else 0.0
        cpc = spend / clicks if clicks else 0.0
        cpm = (spend / impressions * 1000) if impressions else 0.0
        roas = revenue / spend if spend else 0.0
        cpa = spend / conversions if conversions else 0.0

        return {
            "campaign_id": campaign_id,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "spend": round(spend, 4),
            "revenue": round(revenue, 4),
            "CTR": round(ctr, 6),
            "CPC": round(cpc, 4),
            "CPM": round(cpm, 4),
            "ROAS": round(roas, 4),
            "CPA": round(cpa, 4),
        }

    def get_budget_pacing(self, campaign_id: str) -> Dict[str, Any]:
        """Compute daily pacing, ideal vs actual spend, over/under budget %."""
        campaign = self.conn.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if not campaign:
            return {"error": "Campaign not found"}

        start = datetime.strptime(campaign["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(campaign["end_date"], "%Y-%m-%d").date()
        today = date.today()
        total_days = max((end - start).days + 1, 1)
        days_elapsed = max((min(today, end) - start).days + 1, 1)

        budget_total = campaign["budget_total"]
        budget_spent = campaign["budget_spent"]

        daily_budget = budget_total / total_days
        ideal_spend = daily_budget * days_elapsed
        pacing_ratio = budget_spent / ideal_spend if ideal_spend else 0.0
        over_under_pct = (pacing_ratio - 1.0) * 100

        # Record today's pacing snapshot
        pid = str(uuid.uuid4())
        today_str = today.isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO budget_pacing
               (id, campaign_id, date, planned_spend, actual_spend, pacing_ratio)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pid, campaign_id, today_str, round(ideal_spend, 4),
             round(budget_spent, 4), round(pacing_ratio, 4)),
        )
        self.conn.commit()

        return {
            "campaign_id": campaign_id,
            "budget_total": budget_total,
            "budget_spent": round(budget_spent, 4),
            "budget_remaining": round(budget_total - budget_spent, 4),
            "daily_budget": round(daily_budget, 4),
            "ideal_spend": round(ideal_spend, 4),
            "pacing_ratio": round(pacing_ratio, 4),
            "over_under_pct": round(over_under_pct, 2),
            "days_elapsed": days_elapsed,
            "total_days": total_days,
            "status": (
                "over_pacing" if pacing_ratio > 1.05
                else "under_pacing" if pacing_ratio < 0.95
                else "on_track"
            ),
        }

    def optimize_bids(
        self, campaign_id: str, target_cpa: float
    ) -> List[Dict[str, Any]]:
        """Suggest bid adjustments for each ad set based on CPA vs target."""
        ad_sets = self.conn.execute(
            "SELECT * FROM ad_sets WHERE campaign_id = ?", (campaign_id,)
        ).fetchall()

        suggestions = []
        for ad_set in ad_sets:
            metrics = self.conn.execute(
                """SELECT SUM(spend) AS spend, SUM(conversions) AS conversions
                   FROM performance_metrics
                   WHERE entity_type='ad_set' AND entity_id=?""",
                (ad_set["id"],),
            ).fetchone()

            spend = metrics["spend"] or 0.0
            conversions = metrics["conversions"] or 0
            current_cpa = spend / conversions if conversions else float("inf")

            if current_cpa == float("inf"):
                adjustment = 0.0
                action = "no_data"
            elif current_cpa > target_cpa:
                # CPA too high — reduce bid
                ratio = target_cpa / current_cpa
                adjustment = round((ratio - 1.0) * 100, 2)
                action = "reduce_bid"
            else:
                # CPA below target — can increase bid
                ratio = min(target_cpa / current_cpa, 1.5)
                adjustment = round((ratio - 1.0) * 100, 2)
                action = "increase_bid"

            new_bid = round(
                max(0.01, ad_set["bid_amount"] * (1 + adjustment / 100)), 4
            )

            suggestions.append({
                "ad_set_id": ad_set["id"],
                "ad_set_name": ad_set["name"],
                "current_bid": ad_set["bid_amount"],
                "suggested_bid": new_bid,
                "current_cpa": round(current_cpa, 4) if current_cpa != float("inf") else None,
                "target_cpa": target_cpa,
                "adjustment_pct": adjustment,
                "action": action,
            })

        return suggestions

    def get_performance_trend(
        self,
        entity_id: str,
        metric_name: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Time series for a metric with 7-day moving average."""
        rows = self.conn.execute(
            f"""SELECT date, {metric_name}
                FROM performance_metrics
                WHERE entity_id = ?
                ORDER BY date DESC
                LIMIT ?""",
            (entity_id, days),
        ).fetchall()

        series = [{"date": r["date"], "value": r[metric_name]} for r in rows]
        series.reverse()

        # 7-day moving average
        for i, point in enumerate(series):
            window = series[max(0, i - 6): i + 1]
            avg = sum(w["value"] for w in window) / len(window)
            point["moving_avg_7d"] = round(avg, 4)

        return series

    def rank_creatives(self, ad_set_id: str) -> List[Dict[str, Any]]:
        """Rank creatives by CTR, conversion rate, and composite quality score."""
        creatives = self.conn.execute(
            "SELECT * FROM creatives WHERE ad_set_id = ?", (ad_set_id,)
        ).fetchall()

        ranked = []
        for creative in creatives:
            metrics = self.conn.execute(
                """SELECT SUM(impressions) AS impressions,
                          SUM(clicks) AS clicks,
                          SUM(conversions) AS conversions,
                          SUM(spend) AS spend,
                          SUM(revenue) AS revenue
                   FROM performance_metrics
                   WHERE entity_type='creative' AND entity_id=?""",
                (creative["id"],),
            ).fetchone()

            impressions = metrics["impressions"] or 0
            clicks = metrics["clicks"] or 0
            conversions = metrics["conversions"] or 0
            spend = metrics["spend"] or 0.0
            revenue = metrics["revenue"] or 0.0

            ctr = clicks / impressions if impressions else 0.0
            conv_rate = conversions / clicks if clicks else 0.0
            roas = revenue / spend if spend else 0.0
            # Composite quality score (weighted)
            quality_score = round(ctr * 0.4 + conv_rate * 0.4 + min(roas / 10, 1.0) * 0.2, 6)

            ranked.append({
                "creative_id": creative["id"],
                "name": creative["name"],
                "headline": creative["headline"],
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "CTR": round(ctr, 6),
                "conversion_rate": round(conv_rate, 6),
                "ROAS": round(roas, 4),
                "quality_score": quality_score,
            })

        ranked.sort(key=lambda x: x["quality_score"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1

        return ranked

    def generate_report(self, campaign_id: str) -> Dict[str, Any]:
        """Full report dict with all metrics, budget status, top creatives, recommendations."""
        campaign = self.conn.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if not campaign:
            return {"error": "Campaign not found"}

        metrics = self.get_campaign_metrics(campaign_id)
        pacing = self.get_budget_pacing(campaign_id)

        # Top creatives across all ad sets
        ad_sets = self.conn.execute(
            "SELECT id FROM ad_sets WHERE campaign_id = ?", (campaign_id,)
        ).fetchall()
        all_creatives: List[Dict[str, Any]] = []
        for ad_set in ad_sets:
            all_creatives.extend(self.rank_creatives(ad_set["id"]))
        all_creatives.sort(key=lambda x: x["quality_score"], reverse=True)
        top_creatives = all_creatives[:5]

        # Bid optimization suggestions (target CPA = 20% of avg order)
        revenue = metrics["revenue"]
        conversions = metrics["conversions"]
        avg_order = revenue / conversions if conversions else 50.0
        target_cpa = avg_order * 0.2
        bid_suggestions = self.optimize_bids(campaign_id, target_cpa)

        # Recommendations
        recommendations = []
        if pacing["status"] == "over_pacing":
            recommendations.append("Reduce daily budgets — campaign is over-pacing.")
        elif pacing["status"] == "under_pacing":
            recommendations.append("Increase bids or expand targeting — under-pacing.")
        if metrics["CTR"] < 0.01:
            recommendations.append("CTR below 1% — refresh creatives.")
        if metrics["ROAS"] < 2.0:
            recommendations.append("ROAS below 2x — review targeting and bids.")
        if metrics["CPA"] > target_cpa * 1.2:
            recommendations.append("CPA exceeds target — consider pausing low-performers.")
        if not recommendations:
            recommendations.append("Campaign performing within targets.")

        return {
            "campaign": dict(campaign),
            "metrics": metrics,
            "budget_pacing": pacing,
            "top_creatives": top_creatives,
            "bid_suggestions": bid_suggestions,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_table(self, headers: List[str], rows: List[List[Any]]) -> None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(f"{BOLD}{CYAN}" + fmt.format(*headers) + NC)
        print("-" * (sum(col_widths) + 2 * len(headers)))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))

    def _print_progress_bar(
        self, label: str, value: float, total: float, width: int = 40
    ) -> None:
        pct = min(value / total, 1.0) if total else 0.0
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)
        color = GREEN if pct < 0.9 else YELLOW if pct < 1.0 else RED
        print(f"{label:30s} {color}[{bar}]{NC} {pct*100:5.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ad_manager",
        description=f"{BOLD}BlackRoad Ad Manager{NC}",
    )
    parser.add_argument("--db", default="ad_manager.db", help="SQLite database path")
    sub = parser.add_subparsers(dest="command")

    # campaign
    p_camp = sub.add_parser("campaign", help="Campaign operations")
    p_camp.add_argument("action", choices=["create", "list"])
    p_camp.add_argument("--name", default="")
    p_camp.add_argument("--objective", default="conversions")
    p_camp.add_argument("--budget", type=float, default=1000.0)
    p_camp.add_argument("--start", default=date.today().isoformat())
    p_camp.add_argument("--end", default=(date.today() + timedelta(days=30)).isoformat())

    # adset
    p_adset = sub.add_parser("adset", help="Ad set operations")
    p_adset.add_argument("action", choices=["create", "list"])
    p_adset.add_argument("--campaign-id", default="")
    p_adset.add_argument("--name", default="")
    p_adset.add_argument("--targeting", default="{}")
    p_adset.add_argument("--daily-budget", type=float, default=100.0)
    p_adset.add_argument("--bid-strategy", default="CPC")
    p_adset.add_argument("--bid-amount", type=float, default=1.0)

    # creative
    p_creative = sub.add_parser("creative", help="Creative operations")
    p_creative.add_argument("action", choices=["create", "list"])
    p_creative.add_argument("--ad-set-id", default="")
    p_creative.add_argument("--name", default="")
    p_creative.add_argument("--headline", default="")
    p_creative.add_argument("--body", default="")
    p_creative.add_argument("--cta", default="Learn More")
    p_creative.add_argument("--image-url", default="")

    # record
    p_record = sub.add_parser("record", help="Record performance metrics")
    p_record.add_argument("--entity-type", default="campaign")
    p_record.add_argument("--entity-id", required=True)
    p_record.add_argument("--date", default=date.today().isoformat())
    p_record.add_argument("--impressions", type=int, default=0)
    p_record.add_argument("--clicks", type=int, default=0)
    p_record.add_argument("--conversions", type=int, default=0)
    p_record.add_argument("--spend", type=float, default=0.0)
    p_record.add_argument("--revenue", type=float, default=0.0)

    # metrics
    p_metrics = sub.add_parser("metrics", help="Get campaign metrics")
    p_metrics.add_argument("--campaign-id", required=True)
    p_metrics.add_argument("--start-date", default="")
    p_metrics.add_argument("--end-date", default="")

    # report
    p_report = sub.add_parser("report", help="Generate full campaign report")
    p_report.add_argument("--campaign-id", required=True)

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimize bids")
    p_opt.add_argument("--campaign-id", required=True)
    p_opt.add_argument("--target-cpa", type=float, default=10.0)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    mgr = AdManager(db_path=args.db)

    if args.command == "campaign":
        if args.action == "create":
            c = mgr.create_campaign(
                args.name, args.objective, args.budget, args.start, args.end
            )
            print(f"{GREEN}✓ Campaign created:{NC} {c.id}")
            print(f"  Name: {c.name} | Budget: ${c.budget_total:,.2f}")
        elif args.action == "list":
            rows = mgr.conn.execute("SELECT * FROM campaigns").fetchall()
            mgr._print_table(
                ["ID", "Name", "Status", "Budget", "Spent", "Start", "End"],
                [[r["id"][:8], r["name"], r["status"],
                  f"${r['budget_total']:,.2f}", f"${r['budget_spent']:,.2f}",
                  r["start_date"], r["end_date"]] for r in rows],
            )

    elif args.command == "adset":
        if args.action == "create":
            a = mgr.create_ad_set(
                args.campaign_id, args.name, args.targeting,
                args.daily_budget, args.bid_strategy, args.bid_amount,
            )
            print(f"{GREEN}✓ Ad set created:{NC} {a.id}")
        elif args.action == "list":
            rows = mgr.conn.execute("SELECT * FROM ad_sets").fetchall()
            mgr._print_table(
                ["ID", "Campaign", "Name", "Daily Budget", "Bid", "Strategy"],
                [[r["id"][:8], r["campaign_id"][:8], r["name"],
                  f"${r['daily_budget']:,.2f}", f"${r['bid_amount']:.2f}",
                  r["bid_strategy"]] for r in rows],
            )

    elif args.command == "creative":
        if args.action == "create":
            c = mgr.create_creative(
                args.ad_set_id, args.name, args.headline,
                args.body, args.cta, args.image_url,
            )
            print(f"{GREEN}✓ Creative created:{NC} {c.id}")
        elif args.action == "list":
            rows = mgr.conn.execute("SELECT * FROM creatives").fetchall()
            mgr._print_table(
                ["ID", "AdSet", "Name", "Headline", "CTA", "Status"],
                [[r["id"][:8], r["ad_set_id"][:8], r["name"],
                  r["headline"][:30], r["cta"], r["status"]] for r in rows],
            )

    elif args.command == "record":
        m = mgr.record_performance(
            args.entity_type, args.entity_id, args.date,
            args.impressions, args.clicks, args.conversions,
            args.spend, args.revenue,
        )
        print(f"{GREEN}✓ Performance recorded:{NC} {m.id}")

    elif args.command == "metrics":
        data = mgr.get_campaign_metrics(
            args.campaign_id, args.start_date, args.end_date
        )
        print(f"\n{BOLD}{CYAN}Campaign Metrics{NC}")
        mgr._print_table(
            ["Metric", "Value"],
            [[k, str(v)] for k, v in data.items()],
        )

    elif args.command == "report":
        report = mgr.generate_report(args.campaign_id)
        if "error" in report:
            print(f"{RED}Error:{NC} {report['error']}")
            sys.exit(1)
        print(f"\n{BOLD}{MAGENTA}=== Campaign Report ==={NC}")
        print(json.dumps(report, indent=2, default=str))
        # Budget pacing bar
        pacing = report["budget_pacing"]
        print(f"\n{BOLD}Budget Pacing:{NC}")
        mgr._print_progress_bar(
            "Spend", pacing["budget_spent"], pacing["budget_total"]
        )
        print(f"\n{BOLD}Recommendations:{NC}")
        for rec in report["recommendations"]:
            print(f"  {YELLOW}•{NC} {rec}")

    elif args.command == "optimize":
        suggestions = mgr.optimize_bids(args.campaign_id, args.target_cpa)
        print(f"\n{BOLD}{CYAN}Bid Optimization Suggestions{NC}")
        mgr._print_table(
            ["AdSet", "Current Bid", "Suggested Bid", "Adj %", "Action"],
            [[s["ad_set_name"], f"${s['current_bid']:.4f}",
              f"${s['suggested_bid']:.4f}", f"{s['adjustment_pct']:+.1f}%",
              s["action"]] for s in suggestions],
        )


if __name__ == "__main__":
    main()
