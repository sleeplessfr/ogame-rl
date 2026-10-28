#!/usr/bin/env python3
"""Show what the agents of run_real.py are doing, straight from the server's API (no need for the game's scoreboard).

    python tools/agents_status.py --base-url http://localhost/api/v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real.api import ApiClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost/api/v1")
ap.add_argument("--tokens", default="runs/tokens.json")
a = ap.parse_args()
tokens = json.loads(Path(a.tokens).read_text())
for email, tok in sorted(tokens.items()):
    s = ApiClient(a.base_url, tok).get("/status")
    p = s["player"]
    print(f"{email}: player '{p['username']}' (id {p['id']}), points {p['points']}, {len(s['planets'])} planet(s)")
    for pl in s["planets"]:
        c = pl["coordinates"]
        bq = ", ".join(f"{q['machine_name']}->{q.get('target_level')}" for q in pl["building_queue"]) or "-"
        uq = ", ".join(f"{q['machine_name']}x{q.get('amount_remaining')}" for q in pl["unit_queue"]) or "-"
        print(f"   [{c['galaxy']}:{c['system']}:{c['position']}] metal {pl['resources']['metal']} crystal {pl['resources']['crystal']} | building: {bq} | units: {uq}")
    rq = ", ".join(f"{q['machine_name']}->{q.get('target_level')}" for q in s["research"]["queue"]) or "-"
    lv = {k: v for k, v in dict(s["research"]["levels"]).items() if v}
    print(f"   research: {rq} | levels: {lv}")
