"""
How the posted Reels performed, from the Instagram Graph API (same `.env` as `post_reel`).

    uv run python -m tools.reel_stats              # every scheduled Reel posted so far
    uv run python -m tools.reel_stats --last 20    # only the most recent 20
    uv run python -m tools.reel_stats --category Fractals

Each Reel is matched to its `SCHEDULE.md` row by title (the caption's first line), so the
category column comes from the schedule. Columns: reach (unique accounts), average watch time
(the videos are 15 s), likes, shares and saves; `*` marks a breakout (reach of 1,000 or more).
The footer gives the account's follows/unfollows and profile views over the last 28 days.

Reels reach is all-or-nothing. A Reel either breaks out (1,000+ accounts) or stays with the
first test audience (~100-450), and breakouts come with many more shares and saves. Per-Reel
follows and retention curves aren't in the API; they're in the app's Professional dashboard.
Reach keeps growing for ~2 days, so Reels younger than that are marked `new`.
"""

import argparse
import datetime as dt
import re
import statistics
import sys
import time

from loguru import logger

from tools.post_reel import SCHEDULE_PATH, GraphError, client, read_env

BREAKOUT = 1000
REEL_METRICS = "reach,views,saved,shares,ig_reels_avg_watch_time"


def schedule_by_title():
    rows = {}
    for line in SCHEDULE_PATH.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and re.fullmatch(r"\d{4}-\d\d-\d\d", cells[0]) and cells[-2]:
            rows[cells[-2].casefold()] = (cells[0], cells[2])
    return rows


def reels(ig):
    out, params = [], dict(
        fields="id,caption,media_product_type,timestamp,like_count,comments_count",
        limit=50,
    )
    while True:
        page = ig.get(f"{ig.user_id}/media", **params)
        out += [m for m in page["data"] if m.get("media_product_type") == "REELS"]
        after = page.get("paging", {}).get("cursors", {}).get("after")
        if not page.get("paging", {}).get("next") or not after:
            return out
        params["after"] = after


def account_footer(ig):
    now = int(time.time())
    span = dict(
        period="day", metric_type="total_value", since=now - 28 * 86400, until=now
    )
    try:
        follows = ig.get(
            f"{ig.user_id}/insights",
            metric="follows_and_unfollows",
            breakdown="follow_type",
            **span,
        )["data"][0]["total_value"]["breakdowns"][0]["results"]
        counts = {r["dimension_values"][0]: r["value"] for r in follows}
        views = ig.get(f"{ig.user_id}/insights", metric="profile_views", **span)
        views = views["data"][0]["total_value"]["value"]
        profile = ig.get(ig.user_id, fields="followers_count")["followers_count"]
        return (
            f"Account: {profile} followers | last 28 days: +{counts.get('FOLLOWER', 0)} "
            f"follows, -{counts.get('NON_FOLLOWER', 0)} unfollows, {views} profile views"
        )
    except (GraphError, KeyError, IndexError) as e:
        return f"Account insights unavailable: {e}"


def main():
    parser = argparse.ArgumentParser(description="Performance of the posted Reels.")
    parser.add_argument("--last", type=int, help="only the most recent N Reels")
    parser.add_argument("--category", help="only rows whose category contains this")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ig = client(read_env())
    schedule = schedule_by_title()
    rows = []
    for m in reels(ig):
        title = (m.get("caption") or "").strip().split("\n")[0].strip()
        date, category = schedule.get(title.casefold(), (None, None))
        if not date:
            continue  # posted before the schedule existed
        if args.category and args.category.casefold() not in category.casefold():
            continue
        try:
            data = ig.get(f"{m['id']}/insights", metric=REEL_METRICS)["data"]
            stats = {d["name"]: d["values"][0]["value"] for d in data}
        except GraphError as e:
            logger.warning(f"{date} {title}: {e}")
            continue
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(
            m["timestamp"].replace("+0000", "+00:00")
        )
        rows.append(
            dict(
                date=date, category=category, title=title, likes=m.get("like_count", 0),
                comments=m.get("comments_count", 0), new=age < dt.timedelta(days=2),
                **stats,
            )  # fmt: skip
        )
    rows.sort(key=lambda r: r["date"])
    if args.last:
        rows = rows[-args.last :]
    if not rows:
        raise SystemExit("No posted scheduled Reels matched")

    print(
        f"{'date':10}  {'category':22}  {'title':40}  {'reach':>6}  {'watch':>5}  "
        f"{'likes':>5}  {'shares':>6}  {'saves':>5}"
    )
    for r in rows:
        mark = "new" if r["new"] else ("*" if r["reach"] >= BREAKOUT else "")
        watch = (r.get("ig_reels_avg_watch_time") or 0) / 1000
        print(
            f"{r['date']:10}  {r['category'][:22]:22}  {r['title'][:40]:40}  "
            f"{r['reach']:>6}  {watch:>4.1f}s  {r['likes']:>5}  {r['shares']:>6}  "
            f"{r['saved']:>5}  {mark}"
        )
    settled = [r for r in rows if not r["new"]]
    if settled:
        hits = [r for r in settled if r["reach"] >= BREAKOUT]
        med = lambda key, g: statistics.median(r.get(key) or 0 for r in g)  # noqa: E731
        rest = [r for r in settled if r["reach"] < BREAKOUT]
        print(f"\nBreakouts: {len(hits)}/{len(settled)} settled Reels")
        for name, group in (("breakouts", hits), ("others", rest)):
            if group:
                print(
                    f"  {name:9}  median reach {med('reach', group):6.0f}  watch "
                    f"{med('ig_reels_avg_watch_time', group) / 1000:4.1f}s  shares "
                    f"{med('shares', group):3.0f}  saves {med('saved', group):3.0f}"
                )
    print(account_footer(ig))


if __name__ == "__main__":
    main()
