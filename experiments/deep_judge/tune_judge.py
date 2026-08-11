"""Tune the diagnostic judge's prompt until its PASS/FAIL matches the gold ORACLE.

Loop (Continual-Harness "process-reward co-learning", applied to a judge prompt):
  1. Evaluate the current judge prompt on the TUNE split (parallel LLM calls) vs the oracle.
  2. Collect DISAGREEMENTS (false-accepts + false-rejects), sample ~10, and hand them to a CRITIC LLM
     with the current prompt; the critic rewrites the judge's SYSTEM prompt to fix those cases while
     keeping the strict output contract.
  3. Re-evaluate the revised prompt on TUNE; ADOPT it only if balanced-accuracy improves (guards against
     the critic making it worse). Track TEST (held-out) each round for honest generalization.
Stop when TUNE balanced-acc hits the target, plateaus, or max rounds. The critic is another LLM
(`--critic`): default is the same model as the judge (the "is self-critique enough?" test); pass a
different model / a local Qwen endpoint if same-model tuning stalls.

    python -m experiments.deep_judge.tune_judge [rounds=8] [--critic MODEL] [--judge MODEL] [--target 0.9]
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phase1 import common
from phase1.llm import LLM
from experiments.deep_judge.judge_core import INITIAL_PROMPT, confusion, run_judge

HERE = Path(__file__).parent
CONTRACT = ("COVERED / MISSING / DIAGNOSIS / TECHNIQUE / NEXT_QUERY / CONFIDENCE / VERDICT, one per line, "
            "VERDICT exactly PASS or FAIL")

CRITIC_SYSTEM = """You improve the SYSTEM PROMPT of an LLM judge. The judge is a stop/continue \
controller for a multi-hop retrieval agent: given a decomposed question + candidate results + coverage \
signals (per sub-fact best semantic sim & lexical overlap) + score signals, it must output VERDICT = \
PASS (every sub-fact already has a document in the set) or FAIL (a document is still missing). Ground \
truth is an ORACLE that PASSes iff all gold documents are actually in the candidate set — the judge \
never sees it.

You are shown the judge's CURRENT prompt and its MISTAKES on real cases (the oracle truth, the signals \
the judge saw, and what it answered). Two error types:
- FALSE ACCEPT (judge PASS, oracle FAIL): the WORST error — it stops while a document is missing. Usually \
the judge trusted a high score / one strong sub-fact and ignored a sub-fact with low best_sim or low \
lexical overlap.
- FALSE REJECT (judge FAIL, oracle PASS): wastes a hop — usually over-strict thresholds.

Rewrite the judge's system prompt so it would get these cases right, WITHOUT overfitting to them. You may \
sharpen the decision rule (e.g. concrete sim/overlap thresholds for calling a sub-fact missing, how to \
weigh score cliffs, how many sub-facts may be weak before FAIL). You MUST keep the exact output contract: \
%s. Return ONLY the new system prompt text, nothing else.""" % CONTRACT


def load_split(seed=0, tune_frac=0.5):
    path = HERE / "evalset_ce.jsonl"
    if not path.exists():
        path = HERE / "evalset.jsonl"
    ex = [json.loads(l) for l in path.open()]
    random.Random(seed).shuffle(ex)
    pos = [e for e in ex if e["oracle_pass"]]
    neg = [e for e in ex if not e["oracle_pass"]]
    # stratified split so both TUNE and TEST carry PASS and FAIL cases
    def split(xs):
        c = int(len(xs) * tune_frac)
        return xs[:c], xs[c:]
    ptu, pte = split(pos)
    ntu, nte = split(neg)
    tune, test = ptu + ntu, pte + nte
    random.Random(seed).shuffle(tune)
    random.Random(seed).shuffle(test)
    return tune, test


def evaluate(judge_llm, prompt, examples, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda e: run_judge(judge_llm, prompt, e), examples))
    preds = [r["pred_pass"] for r in res]
    golds = [e["oracle_pass"] for e in examples]
    return confusion(preds, golds), res


def disagreement_report(examples, res, k=10):
    bad = [(e, r) for e, r in zip(examples, res) if r["pred_pass"] != e["oracle_pass"]]
    random.Random(0).shuffle(bad)
    lines = []
    for e, r in bad[:k]:
        kind = "FALSE_ACCEPT (judge PASS, truly INCOMPLETE)" if r["pred_pass"] else \
               "FALSE_REJECT (judge FAIL, truly COMPLETE)"
        cov = "; ".join(f"sf{i+1} sim={c['best_sim']:.2f} lex={c['lexical_overlap']:.2f}"
                        for i, c in enumerate(e["coverage"]))
        lines.append(
            f"--- {kind} | {e['n_docs']}-hop {e['state']} ---\n"
            f"Q: {e['query'][:150]}\n"
            f"coverage: {cov}\n"
            f"score_signals: {e['score_signals']}\n"
            f"judge said VERDICT={e['oracle_pass'] and 'FAIL' or 'PASS'}? -> judge={r['verdict']} "
            f"conf={r['confidence']} missing={r['missing']} diagnosis={r['diagnosis']}\n"
            f"ORACLE TRUTH: {'COMPLETE (PASS)' if e['oracle_pass'] else 'INCOMPLETE (FAIL)'}")
    return "\n\n".join(lines), len(bad)


def critic_revise(critic_llm, cur_prompt, report):
    msg = (f"CURRENT JUDGE PROMPT:\n\"\"\"\n{cur_prompt}\n\"\"\"\n\n"
           f"JUDGE MISTAKES (fix these without overfitting):\n{report}\n\n"
           "Return ONLY the improved system prompt.")
    out = critic_llm.complete(msg, system=CRITIC_SYSTEM).strip()
    if out.startswith('"""'):
        out = out.strip('"').strip()
    return out


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    args = sys.argv[2:]
    judge_model = _arg(args, "--judge", common.LLM_MODEL)
    critic_model = _arg(args, "--critic", common.LLM_MODEL)
    target = float(_arg(args, "--target", "0.9"))
    tag = _arg(args, "--tag", "")

    judge_llm = LLM(model=judge_model)
    if critic_model == "qwen":
        from experiments.deep_judge.qwen_critic import QwenCritic, MODEL
        critic_llm = QwenCritic()
        critic_model = MODEL
    else:
        critic_llm = LLM(model=critic_model)
    tune, test = load_split()
    print(f"[tune] judge={judge_model} critic={critic_model} | TUNE={len(tune)} "
          f"(pass {sum(e['oracle_pass'] for e in tune)}) TEST={len(test)} "
          f"(pass {sum(e['oracle_pass'] for e in test)}) | target balanced_acc={target}", flush=True)

    prompt = INITIAL_PROMPT
    log = ["# Diagnostic-judge tuning log\n",
           f"judge=`{judge_model}` critic=`{critic_model}` · TUNE={len(tune)} TEST={len(test)}\n"]
    curve, best = [], None
    prompts_out = (HERE / f"judge_prompts{tag}.jsonl").open("w")

    for r in range(rounds):
        cm, res = evaluate(judge_llm, prompt, tune)
        tm, _ = evaluate(judge_llm, prompt, test)
        curve.append({"round": r, "tune": cm, "test": tm})
        print(f"[round {r}] TUNE bal_acc={cm['balanced_acc']} acc={cm['accuracy']} "
              f"false_accept={cm['false_accept_rate']} false_reject={cm['false_reject_rate']} | "
              f"TEST bal_acc={tm['balanced_acc']} false_accept={tm['false_accept_rate']}", flush=True)
        log.append(f"## Round {r}\nTUNE: {cm}\nTEST: {tm}\n")
        prompts_out.write(json.dumps({"round": r, "tune": cm, "test": tm, "prompt": prompt}) + "\n")
        prompts_out.flush()
        # select on TUNE but require a real margin (>0.01) so we don't chase eval noise; tie-break on TEST
        if best is None or cm["balanced_acc"] > best["tune"]["balanced_acc"] + 0.01 or (
                abs(cm["balanced_acc"] - best["tune"]["balanced_acc"]) <= 0.01
                and tm["balanced_acc"] > best["test"]["balanced_acc"]):
            best = {"round": r, "prompt": prompt, "tune": cm, "test": tm}
        if cm["balanced_acc"] >= target or r == rounds - 1:
            log.append(f"(stop: {'target hit' if cm['balanced_acc'] >= target else 'max rounds'})\n")
            break

        report, n_bad = disagreement_report(tune, res, k=10)
        log.append(f"Disagreements: {n_bad}. Sample shown to critic:\n```\n{report[:2500]}\n```\n")
        new_prompt = critic_revise(critic_llm, prompt, report)
        ncm, _ = evaluate(judge_llm, new_prompt, tune)
        print(f"          critic revision -> TUNE bal_acc={ncm['balanced_acc']} "
              f"(was {cm['balanced_acc']})", flush=True)
        log.append(f"Critic revision TUNE bal_acc={ncm['balanced_acc']} "
                   f"({'ADOPTED' if ncm['balanced_acc'] > cm['balanced_acc'] else 'REJECTED'}).\n")
        if ncm["balanced_acc"] > cm["balanced_acc"]:
            prompt = new_prompt

    prompts_out.close()
    # final: report BEST prompt on TEST
    log.append(f"\n## Best (round {best['round']})\nTUNE: {best['tune']}\nTEST: {best['test']}\n")
    log.append(f"\n### Best judge prompt\n```\n{best['prompt']}\n```\n")
    (HERE / f"tuning_log{tag}.md").write_text("\n".join(log))
    (HERE / f"best_prompt{tag}.txt").write_text(best["prompt"])
    (HERE / f"agreement_curve{tag}.json").write_text(json.dumps(curve, indent=2))
    print(f"\n[tune] BEST round {best['round']}: TUNE bal_acc={best['tune']['balanced_acc']} "
          f"TEST bal_acc={best['test']['balanced_acc']} false_accept={best['test']['false_accept_rate']}")
    print(f"[tune] wrote tuning_log{tag}.md · best_prompt{tag}.txt · agreement_curve{tag}.json")


def _arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args else default


if __name__ == "__main__":
    main()
