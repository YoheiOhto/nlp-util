"""nlputil — shared NLP utilities for biomedical research experiments.

Subpackages
-----------
nlputil.infra   : experiment infrastructure (logging, JSONL/parquet I/O)
nlputil.data    : data access (PubMed parquet, Elasticsearch)
nlputil.llm     : LLM inference (vLLM batch, OpenAI/Anthropic API, JSON parsing)
nlputil.tuning  : HuggingFace fine-tuning helpers (NER, sequence classification)
nlputil.eval    : evaluation metrics (span F1, classification, bootstrap CI)
nlputil.text    : text preprocessing (cleaning, BIO tag conversion)
nlputil.graph   : graph algorithms (BFS subgraph construction)
nlputil.viz     : visualization (confusion matrix, model comparison, Cytoscape CSV)
"""
