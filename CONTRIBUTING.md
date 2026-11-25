# Contributing to utrcalling

Thank you for helping. Questions, bug reports and data that utrcalling does not
handle well are as welcome as code: open an
[issue](https://github.com/AndreMacedo88/utrcalling/issues), with the command
line, the protocol and, if you can, a small file that shows the problem.

## A method of your own

New scores of peak-calling settings, peak callers, bandwidth rules and
protocols do not need changes to utrcalling: the
[guide to extending utrcalling](https://utrcalling.readthedocs.io/en/latest/extending.html)
shows how to write one and use it by name, in a notebook or as a plugin
package. Share a method that suits many data sets by adding it to utrcalling
itself, as below.

## Setting up

utrcalling is developed in a conda environment. From a clone of the
repository:

```bash
conda env create -f environment.yml   # utrcalling in editable mode, and its dependencies
conda activate utrcalling
pip install --group dev               # pytest, coverage, ruff, mypy and pre-commit, pinned
pre-commit install
```

`pre-commit` then formats and lints each commit, with the ruff version that CI
uses.

## Checks

CI runs these on every push and pull request; run them before you push:

```bash
ruff format .            # from the repository root: it formats the Markdown examples too
ruff check .
mypy                     # type hints, checked
coverage run -m pytest   # the test suite, with branch coverage
coverage combine -q && coverage report
```

- **Tests.** The suite must pass, with 100% branch coverage. Every new line
  and branch needs a test that checks its result, not just one that runs it;
  delete code that nothing can reach. Tests are `unittest.TestCase` classes
  that pytest runs, named after the module they test
  (`utrcalling/tests/test_<module>.py`, or a folder for a large module), with
  small synthetic data whose answers are known. `test_cli_workflows.py` runs
  whole command lines, among them those of the paper's workflows: keep them
  working.
- **Types.** Every function signature in `utrcalling/` has type hints, which
  mypy checks. Fix a type error rather than silencing it.
- **Lint.** Fix new findings rather than adding `noqa`.

## Code

- Read the code around yours, and write like it: names, comments, docstrings
  (Google style, where they help), f-strings, `pathlib`.
- Count matrices stay sparse: never make one dense. Avoid per-UTR DataFrames
  and `DataFrame.apply(axis=1)` on large tables; accumulate arrays and build
  tables once.
- Library code installs no warning filters at import, and uses no pandas API
  deprecated in pandas 3.
- The command line starts in well under a second: modules that it imports
  when it starts (`utrcalling/runtime.py`, `utrcalling/commands/options.py`
  and what they import) import neither numpy nor pandas; commands import them
  in their bodies.

## Where the code is

```
utrcalling/
├── utrcalling/
│   ├── runtime.py              Command line (built with Cyclopts)
│   ├── core/
│   │   ├── core.py             Pipeline orchestrator
│   │   ├── protocol.py         Sequencing protocol definitions
│   │   ├── utr_mapping.py      Read-to-UTR mapping (pysam)
│   │   ├── utr_reference.py    GTF reading, and merging the UTRs
│   │   ├── count_matrix.py     Sparse count matrix assembly
│   │   ├── peak_calling/       Peak callers (kde.py), bandwidth rules, scores of
│   │   │                       settings (scores/), and their names (settings.py)
│   │   ├── peak_calling_anndata.py  Peak calling and scoring of count matrices
│   │   ├── registry.py         Methods by name, and plugins
│   │   ├── genome.py           Where sites lie, and the genome around them
│   │   ├── anndata_processing.py  Matrices from matrices: fractions, mean sizes,
│   │   │                       length indices, pseudobulk units
│   │   ├── stats/              Differential site usage: formulas (design.py), the
│   │   │                       Dirichlet-multinomial fit (dirmult.py), permutations,
│   │   │                       effect sizes, DRIMSeq (drimseq.py, drimseq.R), the
│   │   │                       test (site_usage.py)
│   │   ├── barcode_filtering.py
│   │   ├── process_mapped_reads.py
│   │   ├── storage.py
│   │   └── ...
│   ├── commands/               One module per command, a folder per group
│   │                           (options.py: the options they share)
│   └── tests/                  Unit and integration tests
├── docs/                       Documentation (Sphinx), which Read the Docs builds
├── notebooks/                  Exploratory analyses
├── envs/
│   └── utrcalling-frozen.yml   Exact versions validated by the test suite
├── pyproject.toml              Package metadata and dependency ranges
├── environment.yml             Recommended conda environment
├── CITATION.cff                Citation metadata
└── README.md
```

## Commits and changes

- One concern per commit, with a message in the
  [Conventional Commits](https://www.conventionalcommits.org) style and a
  [gitmoji](https://gitmoji.dev), in lower case, on one line:
  `feat: :sparkles: add ...`, `fix: :bug: ...`, `docs: :memo: ...`,
  `test: :white_check_mark: ...`, `refactor: :recycle: ...`.
- Add what users notice to `CHANGELOG.md`, under `[Unreleased]`: what changed
  for them, and what they must do about it.
- Regenerate `utrcalling/tests/tests_baseline.txt`, the output of the whole
  test suite, with the change it reflects:
  `pytest -v -p no:cacheprovider > utrcalling/tests/tests_baseline.txt 2>&1`.
- Document new options in their help text, and new features in the
  documentation (`docs/`) and the README.

## Documentation

The documentation is in `docs/`, as Markdown pages that Sphinx builds into
the website (Read the Docs). The command and Python references are built from
the help texts and docstrings. To build it locally:

```bash
pip install --group docs
sphinx-build -W docs docs/_build/html   # then open docs/_build/html/index.html
```

## Releases

The maintainer releases from `main`, by pushing a tag:

1. Give the version its section in `CHANGELOG.md`, `## [X.Y.Z] - YYYY-MM-DD`:
   what changed for users, breaking changes first.
2. In one commit, set `__version__` in `utrcalling/_metadata.py` and
   `version` and `date-released` in `CITATION.cff`. Push it, and wait for CI
   to pass on it.
3. Tag it, `git tag -a vX.Y.Z -m "utrcalling X.Y.Z"`, and push the tag. The
   release workflow uploads the package to PyPI and publishes the GitHub
   release, which Zenodo archives with its own DOI.

Between releases, `__version__` carries the next version with `.dev0`.
