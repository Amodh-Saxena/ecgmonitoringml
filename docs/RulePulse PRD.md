# RulePulse: Product Requirements Document

Oct 9, 2026 · @Amodh

## 1. Overview

RulePulse is a retail analytics system that finds product association rules and shows how each rule strengthens or fades over time. It replaces one-off, static market-basket analysis with recency-weighted mining, prediction and segmentation in a single dashboard.

**Problem.** Classic association rule mining treats every transaction equally, so a rule that held two years ago looks as strong as one driving sales today. Analysts get long, unranked rule lists with no sense of trend, no link to future demand and no way to see which customers or products behave alike.

**Vision.** A category manager opens RulePulse and sees which product pairings are rising, which are decaying, which segments they belong to, and how likely a basket is to follow a rule next period.

**Core approach.**

- Temporal-decay FP-Growth weights recent transactions more heavily when mining frequent itemsets and rules.
- LightGBM models predict rule or basket outcomes from mined signals.
- UMAP with HDBSCAN groups similar products or customers without a preset cluster count.
- A Streamlit dashboard exposes all of it for non-technical users.

## 2. Goals, non-goals and users

The first release must make rule trends visible and trustworthy for retail analysts, not replace a full recommendation engine.

**Goals**

- Mine association rules where recent transactions count more than old ones.
- Show each rule's strength over time so users can tell rising rules from decaying ones.
- Predict outcomes from mined signals with LightGBM.
- Cluster products or customers with UMAP and HDBSCAN.
- Let a non-technical user explore everything in a Streamlit dashboard.

**Non-goals (v1)**

- Real-time, per-session recommendations on a live storefront.
- Automated price or promotion execution.
- Personal-data handling beyond anonymised transaction and customer IDs.

**Target users**

| User | Need |
| --- | --- |
| Category manager | Know which pairings to bundle, place or promote now |
| Retail data analyst | Tune decay and thresholds, export rules, validate models |
| Merchandising lead | See segments and trends at a glance |

## 3. Use cases and user stories

1. As a category manager, I want the top rules ranked by recent strength so I can choose which bundles to promote this month.
2. As a category manager, I want to see a rule's strength over time so I can tell a fading habit from a growing one.
3. As an analyst, I want to change the decay rate and support, confidence and lift thresholds and re-run, so I can test how sensitive the rules are.
4. As an analyst, I want predicted outcomes with evaluation scores so I can judge whether the model is trustworthy.
5. As a merchandising lead, I want clusters of similar products or customers shown on a 2D map so I can spot segments.
6. As any user, I want to filter by date range, category or store and export the result as CSV.

## 4. Functional requirements

P0 items are required for v1; P1 follows once P0 is stable.

| ID | Module | Requirement | Priority |
| --- | --- | --- | --- |
| F1 | Ingestion | Load transaction data (transaction ID, timestamp, customer ID, item) and validate schema, nulls and duplicates | P0 |
| F2 | Temporal FP-Growth | Mine frequent itemsets with a configurable decay so older transactions weigh less | P0 |
| F3 | Temporal FP-Growth | Generate rules with support, confidence and lift; filter by user thresholds | P0 |
| F4 | Rule trends | Compute each rule's metrics per time window and flag rising, stable or decaying | P0 |
| F5 | LightGBM | Train a model on mined features and report held-out scores | P0 |
| F6 | Clustering | Embed items or customers with UMAP and cluster with HDBSCAN, including a noise label | P0 |
| F7 | Dashboard | Streamlit pages for rules, trends, predictions and clusters with date, category and threshold filters | P0 |
| F8 | Export | Download rules and cluster assignments as CSV | P1 |
| F9 | Config | Store decay, thresholds and random seeds in a config file so runs are reproducible | P1 |
| F10 | Caching | Cache mined results so changing a view does not re-run mining | P1 |

## 5. System architecture

Transactions are validated once, then three independent modules read them and publish results to a single Streamlit dashboard.

&#91;embedded content: RulePulse pipeline · 3 modules, 1 dashboard\]

Temporal FP-Growth produces rules and per-window trends; LightGBM adds predictions from the mined features; UMAP and HDBSCAN produce segments. Each module runs as its own stage, so one can be re-run without the others.

## 6. Data requirements

RulePulse needs timestamped, itemised transactions; everything else is optional enrichment.

| Field | Required | Purpose |
| --- | --- | --- |
| transaction\_id | Yes | Groups items into baskets |
| timestamp | Yes | Drives temporal decay and trend windows |
| item / SKU | Yes | The units of each rule |
| customer\_id | No | Customer clustering and repeat behaviour |
| category, store | No | Dashboard filters |
| price, quantity | No | Revenue-weighted views (future) |

The dataset used is not named in this PRD; record its source, size and date range here once confirmed. Data must be anonymised before use.

## 7. Success metrics

V1 succeeds if recency weighting measurably improves rule relevance and the dashboard is usable without help. Targets are proposals to confirm against real data.

| Area | Metric | Target |
| --- | --- | --- |
| Rules | Hit rate of top-N recent rules on a held-out later period vs the same rules mined without decay | Higher with decay |
| Rules | Share of reported rules with lift > 1 | ≥ 90% |
| Model | LightGBM held-out score (choose AUC or F1 once the target is fixed) | Beats a majority-class baseline |
| Clustering | Silhouette score and share of points labelled noise | Silhouette > 0.25; noise < 30% |
| Performance | Full mining run on the reference dataset | Under 5 min on a laptop |
| Usability | A new user finds the top rising rule | Under 2 min, unaided |

**Evaluation plan.** Split by time, never at random: mine on the earlier window and test on the later one, so the evaluation matches how rules are used.

## 8. Non-functional requirements

- **Performance:** dashboard interactions respond in under 2 s once results are cached.
- **Reproducibility:** fixed random seeds for LightGBM and UMAP; config stored with each run.
- **Scalability:** runs on a single machine; mining must handle minimum support pruning so large catalogues stay tractable.
- **Privacy:** only anonymised IDs; no names, emails or payment data stored.
- **Maintainability:** one module per stage (ingest, mine, model, cluster, app) with unit tests for decay and rule metrics.
- **Portability:** installable from `requirements.txt` on Windows, macOS and Linux.

## 9. Risks, assumptions and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Decay rate chosen arbitrarily | Rules favour the wrong period | Expose decay in the UI; compare several values on held-out data |
| Sparse data in recent windows | Noisy trends | Minimum support and window-size guards; show counts beside every metric |
| UMAP and HDBSCAN unstable across seeds | Clusters shift between runs | Fixed seeds; report cluster stability |
| Spurious rules from many comparisons | Misleading insights | Lift and confidence thresholds; held-out validation |
| Retail data is sensitive | Privacy exposure | Anonymise at ingestion; no personal fields |

**Assumptions.** The pipeline runs in batch, not live; transaction data has timestamps; users are internal analysts.

**Open questions**

- [ ] Which dataset is the reference, and what is its date range?
- [ ] What does LightGBM predict: next-basket item, rule firing or customer value?
- [ ] Are UMAP and HDBSCAN applied to items, customers or both?
- [ ] What decay form is used (exponential half-life or another)?

## 10. Milestones and roadmap

Delivery runs in four phases, and each phase closes only when its metric from section 7 is met.

&#91;embedded content: roadmap · 4 phases, 4 gates\]

Phases run in order and are not drawn to scale; durations are set once the dataset and team are confirmed. After v1, candidates are revenue-weighted rules, a scheduled refresh, and an API for storefront use.
