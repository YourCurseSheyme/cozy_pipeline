# </3

asd

---

Preparations:

```bash
pip install .
```

---

To tune:

```bash
python3.11 -m src.llm_evaluator.interface.cli tune /path/to/cozy_pipeline/data/tune.csv
```

To run:

```bash
python3.11 -m src.llm_evaluator.interface.cli eval /path/to/cozy_pipeline/data/input.csv -o /path/to/cozy_pipeline/data/results.csv
```
