# Business Entity Resolution Pipeline

This folder contains the complete, self-contained, reproducible pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.

---

## 📁 Package Layout

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration, constants, and paths
│   ├── data_loader.py      # TSV streaming and chunked loaders
│   ├── preprocessor.py     # Multilingual text normalization (US, India, France) & suffix stripping
│   ├── blocking.py         # Multi-pass inverted index blocking with country partitioning
│   ├── features.py         # Rapid pairwise string similarity & address number features
│   ├── model.py            # LightGBM matching model & threshold optimization
│   ├── evaluate.py         # Official macro-averaged F_0.5 metric with singleton handling
│   ├── pipeline.py         # End-to-end execution flow
│   └── utils.py            # Output TSV writer & timing utilities
├── README.md               # Reproduction guide (this file)
└── requirements.txt        # Pinned dependencies
```

---

## 🔧 Environment & Requirements

Install pinned dependencies using Python 3.8+:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `rapidfuzz>=3.5.0`
- `lightgbm>=4.0.0`
- `tqdm>=4.66.0`
- `joblib>=1.3.0`
- `scipy>=1.11.0`

---

## 🚀 End-to-End Reproduction

To regenerate both `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the dataset:

```bash
python -m src.pipeline
```

### Pipeline Flow:
1. **Data Ingestion**: Safely reads Source 1, Source 2, and Source 3 files using explicit tab delimiters.
2. **Preprocessing & Legal Suffix Cleaning**: Strips company suffixes (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, `SAS`, `प्रा. लि.`), expands street abbreviations, and standardizes diacritics.
3. **Country Partitioning & Inverted Index Blocking**: Builds inverted token and numeric indices partitioned by country. Retrieves a compact candidate set (<= 15 candidates per Source 1 entity). Writes `output/candidate_pairs.tsv`.
4. **Feature Engineering**: Computes Levenshtein, token sort, token set, n-gram Jaccard, and number overlap features.
5. **Precision-Heavy Matching**: Applies a calibrated decision threshold optimized for macro F_0.5 and writes final matches to `output/matching_results.tsv`.

---

## 🔍 Validation

To validate generated output files against official competition rules:

```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
Exit code `0` confirms the outputs are safe for submission.
