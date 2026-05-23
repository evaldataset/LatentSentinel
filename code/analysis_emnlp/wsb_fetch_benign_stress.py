#!/usr/bin/env python
"""WS-B prep: cache & normalize benign over-refusal stress sets (no GPU).

These are *benign* prompts a deployed filter must NOT flag. Over-refusal here =
false-positive rate. Robust to HF-ID drift: tries candidate IDs, skips gracefully,
writes a manifest of what was obtained. Output: data/emnlp2026/wsb/benign_stress.jsonl
with rows {dataset, split, prompt}.

Env: pinned in requirements.txt (see `bootstrap-latents` conda env)
"""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/emnlp2026/wsb"


def _try(loaders):
    for fn in loaders:
        try:
            rows = fn()
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001
            print(f"   .. candidate failed: {type(e).__name__}: {str(e)[:120]}")
    return None


def main() -> int:
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {
        # XSTest: 250 *safe* prompts that superficially resemble unsafe ones.
        # In xstest-v2, unsafe prompts have type starting "contrast_"; safe = the rest.
        "xstest_safe": [
            lambda: [r["prompt"] for r in load_dataset("natolambert/xstest-v2-copy",
                     split="prompts") if not str(r.get("type", "")).startswith("contrast")],
            lambda: [r["prompt"] for r in load_dataset("walledai/XSTest", split="test")
                     if str(r.get("label", "")).lower() in ("safe", "benign", "1")],
        ],
        # OR-Bench hard: seemingly-toxic but benign (the canonical over-refusal set).
        "orbench_hard": [
            lambda: [r["prompt"] for r in load_dataset("bench-llm/or-bench",
                     "or-bench-hard-1k", split="train")],
            lambda: [r["prompt"] for r in load_dataset("bench-llm/or-bench",
                     "or-bench-80k", split="train").select(range(2000))],
        ],
        # ToxicChat: real user queries; keep strictly benign (non-toxic, non-jailbreak).
        "toxicchat_benign": [
            lambda: [r["user_input"] for r in load_dataset("lmsys/toxic-chat",
                     "toxicchat0124", split="test")
                     if int(r.get("toxicity", 0)) == 0 and int(r.get("jailbreaking", 0)) == 0][:2000],
        ],
        # OKTest: "OverKill" benign questions with sensitive keywords.
        "oktest": [
            lambda: [r["question"] for r in load_dataset("liuchengyuan123/OKTest", split="train")],
            lambda: [r["prompt"] for r in load_dataset("yuchenlin/OKTest", split="train")],
        ],
        # WildGuardMix benign prompts (diverse, real).
        "wildguard_benign": [
            lambda: [r["prompt"] for r in load_dataset("allenai/wildguardmix",
                     "wildguardtrain", split="train")
                     if str(r.get("prompt_harm_label", "")).lower() == "unharmful"][:2000],
        ],
    }

    manifest, total = {}, 0
    with (OUT / "benign_stress.jsonl").open("w") as f:
        for name, loaders in specs.items():
            print(f"[{name}] resolving ...")
            rows = _try(loaders)
            if not rows:
                manifest[name] = {"status": "UNAVAILABLE", "n": 0}
                print(f"[{name}] UNAVAILABLE — skipped")
                continue
            rows = [str(p).strip() for p in rows if p and str(p).strip()]
            seen, uniq = set(), []
            for p in rows:
                if p not in seen:
                    seen.add(p); uniq.append(p)
            for p in uniq:
                f.write(json.dumps({"dataset": name, "split": "benign", "prompt": p}) + "\n")
            manifest[name] = {"status": "ok", "n": len(uniq)}
            total += len(uniq)
            print(f"[{name}] OK n={len(uniq)}")

    manifest["_total_benign_prompts"] = total
    (OUT / "benign_stress_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nTOTAL benign-stress prompts: {total}")
    print(f"written: {OUT}/benign_stress.jsonl  +  benign_stress_manifest.json")
    return 0 if total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
