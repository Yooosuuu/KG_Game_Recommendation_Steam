# KG Embedding models comparison 

| Metric | RotatE + inv | TransE + inv | DistMult | ComplEx + inv |
|---|---|---|---|---|
| **Hits@1** | 0.107 | 0.024 | 0.019 | 0.003 |
| **Hits@3** | 0.344 | 0.206 | 0.032 | 0.010 |
| **Hits@5** | 0.428 | 0.286 | 0.041 | 0.017 |
| **Hits@10** | 0.543 | 0.404 | 0.094 | 0.031 |
| **MRR (iHMR)** | 0.263 | 0.155 | 0.041 | 0.016 |
| **Median Rank** | 8 | 18 | 267 | 467 |
| **Arithmetic Mean Rank** | 83 | 94 | 411 | 822 |
| **Geometric Mean Rank** | 11 | 20 | 181 | 296 |
| **AGMRI** | 0.991 | 0.989 | 0.894 | 0.828 |
| **AAMRI** | 0.965 | 0.960 | 0.823 | 0.648 |
| **Inverse triples** | ✅ | ✅ | ❌ | ✅ |

> **Best model : RotatE + inverse triples** — dominates on all metrics.  
