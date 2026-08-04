"""Does the Qwen reranker help where ms-marco hurt? A/B on the FLATTEST FiQA queries
(dense@10 vs dense-top50 -> Qwen-rerank -> @10)."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import search_as_code as sac
from phase1 import common

q = json.loads((common.DATA_DIR / "queries.json").read_text())
qr = json.loads((common.DATA_DIR / "qrels.json").read_text())
em = SentenceTransformer(common.EMB_MODEL, device="cuda")
embed = lambda ts: em.encode(list(ts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
store = sac.connect("opensearch", index="fiqa", dim=768, hosts=[common.OS_HOST])
def R(ids, g): return len(set(ids[:10]) & g) / len(g) if g else 0.0

cand = [x for x in qr if any(v > 0 for v in qr[x].values())][:150]
rows = []
for x in cand:
    h = store.query_vector(embed([q[x]])[0], top_k=100)
    sc = [hh.score for hh in h]
    rows.append((sc[0] - sc[9], x, h))
rows.sort()
flat = rows[:40]                      # 40 flattest = where dense is weakest / most rerank headroom

qwen = sac.QwenReranker()
msm = sac.CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
d10, qw10, ms10, r100 = [], [], [], []
for gap, x, h in flat:
    g = {d for d, v in qr[x].items() if v > 0}
    d10.append(R(h.top(10).ids(), g))
    r100.append(len(set(h.ids()) & g) / len(g) if g else 0.0)
    qw10.append(R(sac.rerank(q[x], h.top(50), reranker=qwen, top_k=10).ids(), g))
    ms10.append(R(sac.rerank(q[x], h.top(50), reranker=msm, top_k=10).ids(), g))
print(f"\n=== FLAT FiQA queries (n={len(flat)}) — recall@10 ===")
print(f"  dense@10             {np.mean(d10):.3f}")
print(f"  dense+ms-marco rerank {np.mean(ms10):.3f}")
print(f"  dense+QWEN rerank     {np.mean(qw10):.3f}")
print(f"  recall@100 (ceiling)  {np.mean(r100):.3f}")
