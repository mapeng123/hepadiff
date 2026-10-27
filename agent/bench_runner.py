#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LiverBench v2: grounded agent, bare-LLM control and independent judging.

Usage: python bench_runner.py agent|bare|judge|cover|report [limit]
API credentials are read from GLM_KEY and are never written to the benchmark.
"""
import json, os, sys, time, random, threading
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
BANK = json.load(open(f"{BASE}/liverbench_v2.json", encoding="utf-8"))
OUT = os.environ.get("LIVERBENCH_OUT", f"{BASE}/runs")
os.makedirs(OUT, exist_ok=True)

API = os.environ.get("GLM_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
KEY = os.environ.get("GLM_KEY", "")
MODEL_AGENT = os.environ.get("GLM_MODEL", "glm-4.6")
MODEL_JUDGE = os.environ.get("GLM_JUDGE_MODEL", MODEL_AGENT)
SERVER = os.environ.get("HEPADIFF_SERVER", "http://127.0.0.1:6010")

_lock = threading.Lock()

def llm(messages, tools=None, temperature=None, max_retries=4):
    body = {"model": MODEL_AGENT, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    for i in range(max_retries):
        try:
            r = requests.post(API, json=body, timeout=180,
                              headers={"Authorization": f"Bearer {KEY}"})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]
            if r.status_code in (429, 500, 502, 503):
                time.sleep(10 * (i + 1)); continue
            return {"role": "assistant", "content": f"[API error {r.status_code}: {r.text[:200]}]"}
        except Exception as e:
            time.sleep(10 * (i + 1))
    return {"role": "assistant", "content": "[API unreachable]"}

TOOL_SPEC = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
    for n, d, p in [
        ("list_cell_types", "List the 22 liver cell types with a healthy reference state.", {"type": "object", "properties": {}}),
        ("list_conditions", "List disease/condition states the virtual liver can generate.", {"type": "object", "properties": {}}),
        ("check_gene", "Check whether a gene is in the panel and in the TRAINED perturbation vocabulary (knockout supported only if trained).", {"type": "object", "properties": {"gene": {"type": "string"}}, "required": ["gene"]}),
        ("knockout", "Knock out a gene in a liver cell type and get expression deltas and pathway shifts. ONLY works for trained-vocabulary genes; the tool abstains otherwise.", {"type": "object", "properties": {"gene": {"type": "string"}, "cell_type": {"type": "string", "default": "hepatocyte"}, "n_cells": {"type": "integer", "default": 300}}, "required": ["gene"]}),
        ("generate_condition", "Generate liver cells under a disease condition and report expression shifts vs healthy.", {"type": "object", "properties": {"condition": {"type": "string"}, "cell_type": {"type": "string", "default": "hepatic stellate cell"}, "n_cells": {"type": "integer", "default": 300}}, "required": ["condition"]}),
        ("compound_perturbation", "Predict on-target mechanism of a compound via annotated trained target-gene knockout. Abstains when no trained target exists.", {"type": "object", "properties": {"compound": {"type": "string"}, "cell_type": {"type": "string", "default": "hepatocyte"}}, "required": ["compound"]}),
        ("query_drug_target", "ChEMBL/DrugBank target, action type and mechanism annotation for a drug.", {"type": "object", "properties": {"drug_name": {"type": "string"}}, "required": ["drug_name"]}),
        ("query_drug_library", "Check whether a compound is in the FDA DILIrank dataset and its vDILI-concern severity category.", {"type": "object", "properties": {"compound": {"type": "string"}}, "required": ["compound"]}),
        ("query_ligand_receptor", "Check whether a ligand-receptor pair is curated in CellPhoneDB v5.", {"type": "object", "properties": {"ligand": {"type": "string"}, "receptor": {"type": "string"}}, "required": ["ligand", "receptor"]}),
        ("server_info", "Model version, corpus scale and capability boundary of this virtual liver.", {"type": "object", "properties": {}}),
    ]
]

SYS_AGENT = """You are a hepatology research agent with access to HepaDiff, a perturbable virtual liver (masked-diffusion model trained on 18.48 million liver cells from 482 datasets). Use the provided tools when the question requires model prediction or database lookup. Important honesty rules:
1. Knockout prediction is ONLY supported for genes in the trained vocabulary — always check_gene first when unsure; if the tool returns ABSTAIN, report the limitation explicitly instead of guessing.
2. Distinguish model-derived evidence from your own background knowledge, and say which is which.
3. Give a concise final answer after using tools (or state that tools are not needed)."""

SYS_BARE = "You are a hepatology research assistant. Answer the question concisely from your own knowledge. If you do not know, say so."

def call_tool(name, args):
    try:
        r = requests.post(f"{SERVER}/tools/{name}", json=args, timeout=600)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def run_agent(item, temperature=0.3):
    msgs = [{"role": "system", "content": SYS_AGENT},
            {"role": "user", "content": item["question"]}]
    hist = list(msgs)
    for _ in range(6):
        m = llm(msgs, tools=TOOL_SPEC, temperature=temperature)
        hist.append(m)
        if m.get("tool_calls"):
            msgs.append(m)
            for tc in m["tool_calls"]:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                res = call_tool(fn, args)
                tm = {"role": "tool", "tool_call_id": tc["id"], "name": fn,
                      "content": json.dumps(res, ensure_ascii=False)[:6000]}
                msgs.append(tm); hist.append(tm)
            continue
        return m.get("content", ""), hist
    return "[max tool rounds reached]", hist

def run_bare(item, temperature=0.3):
    msgs = [{"role": "system", "content": SYS_BARE},
            {"role": "user", "content": item["question"]}]
    m = llm(msgs, temperature=temperature)
    return m.get("content", ""), msgs + [m]

JUDGE_PROMPT = """You are grading an answer to a hepatology benchmark question against a gold answer from an independent source.

Question: {q}
Gold answer (authoritative): {gold}
Gold source: {src}
Candidate answer: {ans}

Grading rules:
- If the gold answer says the correct behavior is to REFUSE/abstain, grade correct=true ONLY if the candidate explicitly refuses or states the capability limitation.
- Otherwise grade correct=true if the candidate's key claim matches the gold answer in substance (paraphrase allowed; extra correct detail is fine).
- Grade correct=false for wrong, contradictory, or evasive answers.
Reply with exactly one JSON object: {{"correct": true/false, "rationale": "one sentence"}}"""

def judge(item, ans):
    m = llm([{"role": "user", "content": JUDGE_PROMPT.format(
        q=item["question"], gold=item["gold_answer"], src=item["gold_source"], ans=ans[:3000])}],
        temperature=0.0)
    txt = m.get("content", "")
    try:
        s = txt[txt.index("{"):txt.rindex("}") + 1]
        d = json.loads(s)
        return bool(d.get("correct")), str(d.get("rationale", ""))[:300]
    except Exception:
        return None, txt[:300]

def _save(mode, qid, rec):
    with _lock:
        with open(f"{OUT}/{mode}_{qid}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)

def _done(mode, qid):
    return os.path.exists(f"{OUT}/{mode}_{qid}.json")

def main_agent(limit=None):
    items = BANK[:limit] if limit else BANK
    def work(item):
        if _done("agent", item["id"]):
            return
        ans, hist = run_agent(item)
        _save("agent", item["id"], {"item": item, "answer": ans, "history": hist})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, items))
    print("agent done")

def main_bare(limit=None):
    items = BANK[:limit] if limit else BANK
    def work(item):
        if _done("bare", item["id"]):
            return
        ans, hist = run_bare(item)
        _save("bare", item["id"], {"item": item, "answer": ans, "history": hist})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, items))
    print("bare done")

def main_judge(limit=None):
    items = BANK[:limit] if limit else BANK
    def work(item):
        for mode in ("agent", "bare"):
            if _done(f"judge_{mode}", item["id"]):
                continue
            f = f"{OUT}/{mode}_{item['id']}.json"
            if not os.path.exists(f):
                continue
            rec = json.load(open(f, encoding="utf-8"))
            ok, rat = judge(item, rec["answer"])
            _save(f"judge_{mode}", item["id"], {"id": item["id"], "correct": ok, "rationale": rat})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, items))
    print("judge done")

def main_cover(limit=None):
    rng = random.Random(7)
    by_cat = {}
    for it in BANK:
        by_cat.setdefault(it["category"], []).append(it)
    subset = []
    for cat, its in sorted(by_cat.items()):
        k = max(4, round(120 * len(its) / len(BANK)))
        subset += rng.sample(its, min(k, len(its)))
    subset = subset[:limit] if limit else subset
    print(f"coverage subset: {len(subset)}")
    def work(item):
        for r in range(4):
            if _done(f"cover{r}", item["id"]):
                continue
            ans, hist = run_agent(item, temperature=0.7)
            ok, rat = judge(item, ans)
            _save(f"cover{r}", item["id"], {"item": item, "answer": ans,
                                            "correct": ok, "rationale": rat})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, subset))
    print("cover done")

def main_report():
    import csv
    from collections import Counter
    rows = []
    for it in BANK:
        row = {"id": it["id"], "category": it["category"]}
        for mode in ("agent", "bare"):
            f = f"{OUT}/judge_{mode}_{it['id']}.json"
            row[mode] = json.load(open(f, encoding="utf-8"))["correct"] if os.path.exists(f) else None
        rows.append(row)
    ag = [r for r in rows if r["agent"] is not None]
    ba = [r for r in rows if r["bare"] is not None]
    both = [r for r in rows if r["agent"] is not None and r["bare"] is not None]
    n_ag = sum(1 for r in ag if r["agent"]); n_ba = sum(1 for r in ba if r["bare"])
    # McNemar
    b = sum(1 for r in both if r["agent"] and not r["bare"])
    c = sum(1 for r in both if r["bare"] and not r["agent"])
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    from scipy.stats import chi2 as chi2dist
    p = float(chi2dist.sf(chi2, 1))
    # coverage@k
    cov = {}
    ids = set()
    for f in os.listdir(OUT):
        if f.startswith("cover0_"):
            ids.add(f[7:-5])
    for k in (1, 2, 3, 4):
        hit = 0; n = 0
        for qid in ids:
            oks = []
            for r in range(k):
                f = f"{OUT}/cover{r}_{qid}.json"
                oks.append(json.load(open(f, encoding="utf-8"))["correct"] if os.path.exists(f) else None)
            if any(o is not None for o in oks):
                n += 1; hit += int(any(oks))
        cov[k] = round(hit / n, 4) if n else None
    per_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in both if r["category"] == cat]
        per_cat[cat] = {"n": len(sub),
                        "agent": round(sum(r["agent"] for r in sub) / len(sub), 3) if sub else None,
                        "bare": round(sum(r["bare"] for r in sub) / len(sub), 3) if sub else None}
    rep = {"n_questions": len(BANK),
           "agent": {"n": len(ag), "correct": n_ag, "acc": round(n_ag / len(ag), 4) if ag else None},
           "bare": {"n": len(ba), "correct": n_ba, "acc": round(n_ba / len(ba), 4) if ba else None},
           "mcnemar": {"b": b, "c": c, "chi2_cc": round(chi2, 2), "p": p},
           "coverage_at_k": cov, "per_category": per_cat}
    json.dump(rep, open(f"{BASE}/liverbench_v2_report.json", "w"), indent=1)
    with open(f"{BASE}/liverbench_v2_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "agent", "bare"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    mode = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"agent": main_agent, "bare": main_bare, "judge": main_judge,
     "cover": main_cover, "report": main_report}[mode](limit)
