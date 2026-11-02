# iML 26/27 – Exercise 1: Quantifying Interpretability

Template repository for the bonus task. **Fork this repository**, replace the example with your own
work, and add the PDF of your presentation. **You are expected to rewrite this README as well**: by
the deadline it should describe your approach, how to reproduce your results, and contain your own
[AI statement](#ai-statement).

> **Deadline: Sunday, November 15th, 2026, 23:59.** Your code and presentation must be submitted by then.

## Tasks

### Task 1 (1  point): An interpretability metric

Develop a metric that quantifies model interpretability. It has to work within one model class,
e.g. for decision trees, random forests and gradient-boosted trees. Then evaluate a few
inherently interpretable models and/or datasets and investigate the *threshold of
interpretability*: at what point does a model stop being interpretable?

- A trivial idea would be the *number of features used*. Your metric should go beyond that.
- We will check that you did not simply copy someone else's interpretability metric.
- You may use an LLM-based metric. In that case you have to provide evidence that its scores
  agree with human judgments.

### Task 2 (2 bonus points): Meta-features and interpretability

Find connections between a dataset's meta-features (e.g. number of samples, number of features,
class balance, …) and the interpretability, measured with your metric from Task 1, of several
inherently interpretable models. This repository has no code for Task 2. Structure it as you see fit.

### Presentation

Prepare a **5-minute presentation** of your metric and results across models and datasets (and, if
applicable, your Task 2 findings). Commit it as [`presentation/presentation.pdf`](presentation/).

## Setup

The repository is an installable Python package. We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

With plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The first run downloads the Titanic dataset from OpenML and caches it in `data/` (relative to the
directory you run from). You may use that dataset, but dont you want to rather look at something existing?

## Repository structure

```
iml_interpretability/
├── data.py        # Dataset container + load_titanic(); add your own datasets here
├── models.py      # get_models(): the models you compare (example: one decision tree)
├── metric.py      # InterpretabilityMetric base class + RelativeNumberOfLeaves (example)
└── evaluate.py    # evaluate(): fits all models on all datasets and computes all metrics
scripts/
└── run_experiment.py   # runs evaluate() and writes results/scores.csv
tests/                  # pytest tests
presentation/           # put presentation.pdf here
```

The repository runs end to end on a minimal example: a single scikit-learn decision tree on
Titanic, scored by its number of leaves relative to the maximum possible for its depth (`RelativeNumberOfLeaves` in
[`iml_interpretability/metric.py`](iml_interpretability/metric.py)). Replace this example with your
own metric, models and datasets. You are free to restructure everything, as long as the package
stays installable and your results stay reproducible from the scripts in this repository.

## Code style

Please keep your code clean and consistent. We recommend [ruff](https://docs.astral.sh/ruff/) as
linter and formatter. A configuration is already included in `pyproject.toml`:

## Submission

- Submit your code and presentation by **Sunday, November 15th, 2026, 23:59**.
- Your fork contains your code, the scripts that reproduce your results, and
  `presentation/presentation.pdf`.
- **Rewrite this README.** Replace the template text with a short description of your approach and
  how to reproduce your results.
- **Mandatory:** replace the [AI statement](#ai-statement) below with your own, stating how you
  used AI tools (or that you did not use any).

## AI statement

> **Mandatory.** Replace this example with your own: which AI tools you used, for what, and how you
> checked their output (or that you used none).

*Example:* I created this repository with Claude Code (Claude Opus 5.5). After clarifying the scope
with it, it generated the code, tests and README based on my own projects as style references. I
reviewed and edited the result and checked that it runs. I take full responsibility for and
ownership of the content of this repository.
