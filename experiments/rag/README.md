# Mistake DB for the coder (RAG experiment)

The DB holds reading errors from runs on parts outside every test set. Each error is keyed by the coder's description of the feature it got wrong. Run every stage from the repository root.

| Stage | Command | Writes |
|---|---|---|
| Parts | `python -m experiments.rag.samples N_ECCV N_ITERCAD` | ids for `run_parallel.sh --samples` |
| Runs | `zeroshot/run_parallel.sh --datasets train_eccv,itercad_pool --samples IDS ...` | one run per part |
| Pairs | `python -m experiments.rag.pairs PAIRS SWEEP...` | first and final reading with its build |
| Geometry | `python -m experiments.rag.geometry PAIRS DIR` | IoU, missing (M) and extra (E) pieces, pictures |
| Judge | `python -m experiments.rag.judge PAIRS DIR/comparisons.jsonl JUDGED` | Sol's cause per piece, and the reading errors, each with the question it got wrong |
| QA | `python -m experiments.rag.qa PAIRS DIR/comparisons.jsonl JUDGED [REPEAT]` | the judge's pass marks, and errors to check by eye |
| DB | `python -m experiments.rag.build_db PAIRS DIR/comparisons.jsonl JUDGED DB` | `zeroshot/pipeline/retrieval/mistakes.py` rows |
| Retrieval | `python -m experiments.rag.relevance DB [K]` | how often each method's warnings ask the question a persisted error got wrong, leaving the part out |
| Re-read | `python -m experiments.rag.reread PAIRS DIR/comparisons.jsonl JUDGED OUT` | a fresh reader's answers to the questions each feature's kind raises, flags where they differ from the reading, and their cost |
| Trees | `python -m experiments.rag.trees JUDGED TREES.json` | Sol's decision trees per question, in two folds of parts, from the first readings' mistakes |
| Tree labels | `python -m experiments.rag.oracle PAIRS DIR/comparisons.jsonl JUDGED TREES.json ORACLE` | Sol, shown the real part: which questions apply to each feature, the real answers and the trees' observations; then how often the trees decide and are right |
| Routing | `python -m experiments.rag.routing PAIRS JUDGED ORACLE LLM_ROUTES` | keyword, embedding and LLM routing of features to trees, against the tree labels |
| Grounded trees | `python -m experiments.rag.trees grounded JUDGED OUT.json [MARK...]` | Sol's tree per kind of mark, two folds |
| Tree tests | `python -m experiments.rag.tree_tests OUT.json TESTED.json [ROUNDS]` | each tree checked on worded scenes, Sol repairing failures |
| Merge | `python -m experiments.rag.trees merge TREES.json TESTED.json...` | the trees file a questions folder reads |
| Questions | `python -m experiments.rag.grounded questions PAIRS KEYS\|all DIR [KIND,...]` | marks from each reading and the images each question shows |
| Observe | `python -m experiments.rag.grounded observe DIR sol\|luna [WORKERS]` | a reader's answers to each question's tree |
| Report | `python -m experiments.rag.grounded report DIR PAIRS JUDGED READER...` | doubts against the judged errors, and cost |
| Hole truth | `python experiments/rag/eval/hole_oracle.py DIR OUT.jsonl` | evaluation only: whether the real part has material along each hole's axis |

The coder sees the DB when `workflow.mistake_db=DB` is set. It runs without the part's own errors. The DB route was not adopted (STATUS.md 2.1); it stays for reproducing those results.

## How the grounded trees are made

A tree reads one kind of mark (`MARKS` in `trees.py`): a description of what is drawn, its conclusions, and the lines that mislead readers. Sol writes it; nothing about the drawing at hand is learnt.

1. Judge readings (`judge.py`): each reading error gets a decision, the question it got wrong (`QUESTIONS` in `zeroshot/pipeline/retrieval/mistakes.py`), and the lines that decided it.
2. Write: `trees grounded` asks Sol (`grounded_trees.md`) for one tree per kind and fold. Its material is the mark's description and confusers, and the decisive lines of the first readings' errors of that fold's parts on the kind's questions (`Mark.cases`). Parts fall in folds by a hash of their id; a part's questions use the other fold's tree.
3. Test: `tree_tests` has Sol (`tree_observer.md`) answer each tree's questions for worded scenes with known conclusions (`SCENARIOS` in `tree_tests.py`), and has Sol repair a tree that concludes wrongly (`tree_repair.md`), up to ROUNDS times.
4. Merge the tested trees into the questions folder's `trees_grounded_tested.json`.

Sol's answers vary, so rerunning gives other trees of the same design. The trees each result used are kept, read-only, in `outputs/20261024_rag_pilot/analysis_100/frozen_20261026/` (see its README.txt).

Every error must explain a piece of the geometric difference, so Sol attributes and words errors but does not find them.
