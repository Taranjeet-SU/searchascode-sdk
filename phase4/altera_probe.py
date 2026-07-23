"""Sanity-check gte-alt-v1 dense retrieval against the Altera ft_document index."""
from phase4 import altera

print("ping:", altera.ping().splitlines().__len__(), "indices reachable")
print("warming embedder (downloads gte-alt-v1 first time)...", flush=True)
for q in ["Stratix 10 power gating DSP and M20K memory blocks",
          "How do I configure a Nios V processor",
          "Agilex 7 transceiver channel count"]:
    print(f"\n### {q}", flush=True)
    for d in altera.dense(q, 3):
        title = str(d["title"])[:64]
        text = str(d["text"])[:110].replace("\n", " ")
        print(f"  {d['score']:.3f} | {title} | url:{d['url']}", flush=True)
        print(f"       {text}", flush=True)
