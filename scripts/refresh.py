#!/usr/bin/env python3
"""
LP Tracker Daily Refresh Script
Replicates lp-tracker-refresh SKILL logic for GitHub Actions.

Required env vars:
  ANTHROPIC_API_KEY  (optional — enables rich Chinese descriptions via Claude API)

How it works:
  1. Read public/index.html to learn which announcement IDs are already tracked.
  2. Query cninfo full-text search API for LP-relevant disclosures (past 7 days).
  3. Filter out duplicates, S基金, and public-fund (公募) items.
  4. For each new item, generate a DATA.rows entry.
     - With ANTHROPIC_API_KEY: calls claude-haiku for rich detail text (same quality as desktop skill).
     - Without key: uses a template-based description.
  5. Prepend new rows to DATA.rows, update lastUpdated and TODAY.
  6. Overwrite public/index.html (GitHub Actions then commits + pushes).
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Paths & config ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
HTML_PATH   = REPO_ROOT / "docs" / "index.html"
BEIJING_TZ  = ZoneInfo("Asia/Shanghai")
API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")

CNINFO_BASE = "http://www.cninfo.com.cn/new/fulltextSearch/full"

# Keywords to search; each query is de-duplicated by announcementId before merging.
SEARCH_KEYWORDS = [
    "认购份额",
    "参与设立",
    "对外投资",
    "产业基金",
    "母基金",
    "创业投资基金",
]

# Announcement title substrings that mean "not an LP investment" → skip.
EXCLUDE_PATTERNS = [
    "S基金", "份额转让", "接续基金", "continuation",
    "减持", "回购", "注销", "清算",
]

# pageColumn values that mean public funds (公募) → skip.
EXCLUDE_COLUMNS = {"SZJJ"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def today_beijing() -> datetime.date:
    return datetime.datetime.now(BEIJING_TZ).date()


def now_beijing_str() -> str:
    return datetime.datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")


def timestamp_to_date(ms: int) -> str:
    dt = datetime.datetime.fromtimestamp(ms / 1000, tz=BEIJING_TZ)
    return dt.strftime("%Y-%m-%d")


def escape_js(s: str) -> str:
    """Escape a value for embedding inside a JS double-quoted string."""
    return (s.replace("\\", "\\\\")
             .replace('"', '&quot;')
             .replace("\n", " ")
             .replace("\r", ""))


def cninfo_fetch(keyword: str, sdate: str, edate: str) -> list:
    params = {
        "searchkey": keyword,
        "sdate": sdate,
        "edate": edate,
        "isfulltext": "false",
        "sortName": "pubdate",
        "sortType": "desc",
        "pageNum": "1",
    }
    url = CNINFO_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        return data.get("announcements") or []
    except Exception as exc:
        print(f"  [cninfo] '{keyword}' error: {exc}")
        return []


def get_tracked_ids(html: str) -> set:
    """Extract announcement IDs already present in the HTML (from cninfo PDF URLs)."""
    return set(re.findall(r"finalpage/\d{4}-\d{2}-\d{2}/(\d{10,})\.PDF", html))


def is_relevant(item: dict) -> bool:
    title = item.get("announcementTitle", "")
    col   = item.get("pageColumn", "")
    # Exclude public funds
    if col in EXCLUDE_COLUMNS:
        return False
    # Exclude S基金 / non-LP patterns
    for pat in EXCLUDE_PATTERNS:
        if pat in title:
            return False
    # Must relate to fund/LP investment
    lp_terms = ["基金", "合伙企业", "股权投资", "创业投资", "私募", "对外投资", "认购"]
    return any(t in title for t in lp_terms)


# ── Detail generation ─────────────────────────────────────────────────────────

_HAIKU_MODEL = "claude-haiku-4-5"   # cost-efficient; swap to claude-sonnet-4-6 for higher quality


def _call_anthropic(prompt: str) -> str:
    body = json.dumps({
        "model": _HAIKU_MODEL,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"].strip()


def build_detail_llm(item: dict) -> tuple[str, str]:
    """Returns (detail_text, industry_text) via Anthropic API."""
    sec_name = item.get("secName", "")
    sec_code = item.get("secCode", "")
    col      = item.get("pageColumn", "")
    title    = re.sub(r"</?em>", "", item.get("announcementTitle", ""))
    date     = timestamp_to_date(item.get("announcementTime", 0))
    adj_url  = item.get("adjunctUrl", "")

    # Infer exchange from pageColumn prefix
    if col.startswith("SH"):
        exchange = "SH"
    elif col.startswith("SZ"):
        exchange = "SZ"
    else:
        exchange = col[:2] if len(col) >= 2 else ""

    prompt = f"""你是中国PE/VC一级市场LP遴选分析师。根据巨潮资讯公告信息，为LP追踪器写两个字段：

公告信息：
- 公司：{sec_name}（{sec_code}.{exchange}）
- 标题：{title}
- 日期：{date}
- adjunctUrl：{adj_url}

请用JSON格式输出（不要加markdown代码块），字段说明：
- detail（约120字）：①公司主营业务简介；②本次投资性质（CVC/上市公司做LP）；③可能的投资赛道方向；④结尾固定写"巨潮结构化层召回（adjunctUrl: {adj_url}）"
- industry（约20字）：2-4个行业标签，逗号分隔

示例输出：
{{"detail":"上市公司CVC出资；...；巨潮结构化层召回（adjunctUrl: {adj_url}）","industry":"半导体、模拟芯片、集成电路"}}"""

    try:
        raw = _call_anthropic(prompt)
        # Strip any accidental markdown fences
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        obj = json.loads(raw)
        return obj.get("detail", ""), obj.get("industry", "")
    except Exception as exc:
        print(f"  [anthropic] error for {sec_name}: {exc}")
        return "", ""


def build_detail_template(item: dict) -> tuple[str, str]:
    """Fallback: template-based detail without LLM."""
    sec_name = item.get("secName", "")
    sec_code = item.get("secCode", "")
    title    = re.sub(r"</?em>", "", item.get("announcementTitle", ""))
    adj_url  = item.get("adjunctUrl", "")
    detail = (
        f"上市公司CVC出资；{sec_name}（{sec_code}）；{title}；"
        f"巨潮结构化层召回（adjunctUrl: {adj_url}）"
    )
    return detail, "待补充"


# ── Row builder ───────────────────────────────────────────────────────────────

def item_to_row(item: dict) -> str:
    sec_name = item.get("secName", "")
    sec_code = item.get("secCode", "")
    col      = item.get("pageColumn", "SZZB")
    title    = re.sub(r"</?em>", "", item.get("announcementTitle", ""))
    date     = timestamp_to_date(item.get("announcementTime", 0))
    adj_url  = item.get("adjunctUrl", "")

    if col.startswith("SH"):
        exchange = "SH"
    elif col.startswith("SZ"):
        exchange = "SZ"
    else:
        exchange = col[:2] if len(col) >= 2 else ""

    full_code = f"{sec_code}.{exchange}" if exchange else sec_code
    name   = f"{sec_name}（{full_code}）· {title}"
    action = f"上市公司{sec_name}公告{title}，以LP身份参与产业基金投资"
    url    = f"http://static.cninfo.com.cn/{adj_url}"

    if API_KEY:
        detail, industry = build_detail_llm(item)
        time.sleep(0.3)  # gentle rate-limit
    else:
        detail, industry = build_detail_template(item)

    if not detail:
        detail, industry = build_detail_template(item)

    return (
        f'    {{ date: "{date}", precise: true, cat: "market", level: "provincial",'
        f' name: "{escape_js(name)}",'
        f' action: "{escape_js(action)}",'
        f' size: "详见公告", fbType: "none", fbText: "CVC/上市公司",'
        f' detail: "{escape_js(detail)}",'
        f' industry: "{escape_js(industry)}",'
        f' url: "{url}", urlLabel: "巨潮公告" }},\n'
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_str   = now_beijing_str()
    today_str = str(today_beijing())
    print(f"=== LP Tracker Refresh {now_str} (Beijing) ===")
    print(f"Anthropic API: {'enabled (' + _HAIKU_MODEL + ')' if API_KEY else 'disabled (template mode)'}")

    html = HTML_PATH.read_text(encoding="utf-8")
    tracked = get_tracked_ids(html)
    print(f"Already tracked IDs: {len(tracked)}")

    # Build date window: past 7 days
    end_d   = today_beijing()
    start_d = end_d - datetime.timedelta(days=7)
    sdate, edate = str(start_d), str(end_d)
    print(f"Search window: {sdate} → {edate}")

    # Fetch & deduplicate
    seen: dict[str, dict] = {}
    for kw in SEARCH_KEYWORDS:
        items = cninfo_fetch(kw, sdate, edate)
        print(f"  '{kw}': {len(items)} results")
        for it in items:
            aid = it.get("announcementId", "")
            if aid and aid not in seen:
                seen[aid] = it

    print(f"Unique announcements: {len(seen)}")

    # Filter
    new_items = [
        it for aid, it in seen.items()
        if aid not in tracked and is_relevant(it)
    ]
    # Sort newest first
    new_items.sort(key=lambda x: x.get("announcementTime", 0), reverse=True)
    print(f"New LP-relevant items: {len(new_items)}")

    # Generate rows
    new_rows = ""
    for it in new_items:
        sec  = it.get("secName", "")
        date = timestamp_to_date(it.get("announcementTime", 0))
        titl = re.sub(r"</?em>", "", it.get("announcementTitle", ""))[:50]
        print(f"  + {date} {sec}: {titl}")
        new_rows += item_to_row(it)

    # Patch HTML
    html = re.sub(
        r'lastUpdated:\s*"[^"]+"',
        f'lastUpdated: "{now_str}"',
        html, count=1
    )
    html = re.sub(
        r'const TODAY\s*=\s*new Date\("[^"]+"\)',
        f'const TODAY = new Date("{today_str}")',
        html, count=1
    )
    if new_rows:
        html = html.replace("  rows: [\n", "  rows: [\n" + new_rows, 1)

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Written: {HTML_PATH} ({len(html):,} bytes)")
    print(f"Done. +{len(new_items)} rows, lastUpdated={now_str}")


if __name__ == "__main__":
    main()
