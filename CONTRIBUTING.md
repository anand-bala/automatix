# Contribution Guidelines

## Build, Lint, and Test Commands

The `justfile` defines a set of build, lint, and test commands, which can directly be
run as below:


**Testing** :
```bash
# Run all tests (uses --lf for last-failed)
just test
# OR
uv run --dev --frozen pytest --lf

# Run all tests verbosely without --lf
uv run --dev --frozen pytest -v

# Run a single test file
uv run --dev --frozen pytest tests/test_weight_functions.py -v

# Run a single test function
uv run --dev --frozen pytest tests/test_weight_functions.py::TestWeightFunctionBasics::test_constant_weight_function -v

# Run tests for specific package
uv run --dev --frozen pytest packages/morphata/tests/ -v
uv run --dev --frozen pytest packages/algebraic/tests/ -v

# Run with coverage
uv run --dev --frozen pytest --cov=automatix --cov-report=term
```

**Linting and Formatting** :
```bash
# Format and lint (fixes issues automatically)
just fmt
# OR
uv run --frozen ruff format
uv run --frozen ruff check --output-format concise --fix --exit-non-zero-on-fix

# Format + type check
just lint
```

**Type Checking** :
```bash
# Run all type checkers
just type-check

# Run specific type checkers
uv run --frozen mypy --strict
uv run --frozen ty check --output-format concise
uv run --frozen pyrefly check --output-format min-text
```

**Development Setup** :
```bash
# Set up dev environment (sync dependencies)
just dev
# OR
uv sync --all-packages --frozen --inexact --dev

# For CUDA support (set CUDA_VERSION=12 or 13)
CUDA_VERSION=12 just dev
```


One can also copy the included `.envrc.recommended` file to `.envrc` and use it with
`direnv` .

## Code Style Guidelines

### Type Annotations

- All code must pass `mypy --strict` . Use explicit type annotations.
- Use Python 3.12+ syntax: `type` aliases, `|` for unions, generic classes and functions
  with `[T]`
- Always specify return types, including `-> None`

### Documentation

- Use plain ASCII text in code, comments, and docs; do not use emojis or Unicode math
  symbols. For mathematical notation, use LaTeX in inline code blocks.
- Format: use Numpy-style docstrings with sections: Parameters, Returns, Notes,
  Examples...
- If using inline comments, make sure the comments say _why_ something is done, as
  opposed _what_ is being done.

## Workspace Structure

This is a monorepo with the main `automatix` package and two workspace packages:

```
automatix/
├── src/automatix/              # Main package (weighted automata)
│   ├── spec.py                 # WeightFunction protocol, AcceptanceCondition
│   ├── acc.py                  # State-set acceptance conditions (runtime checking)
│   ├── operators/
│   │   ├── __init__.py         # Re-exports MatrixOperator and PolynomialOperator
│   │   ├── matrix.py           # MatrixOperator for weighted NFAs
│   │   └── polynomial.py       # PolynomialOperator for AFAs
│   ├── automata/
│   │   ├── nfa.py              # Re-exports morphata.examples.nfa
│   │   └── strel.py            # Re-exports morphata.examples.strel
│   └── weights/
│       └── guard_weights.py    # Guard-based weight functions
│
├── packages/morphata/          # Foundation: graph-based automata (pure structural)
│   ├── src/morphata/
│   │   ├── spec.py             # Base automaton interfaces (Domain, TransitionRelation, etc.)
│   │   ├── acceptance.py       # Expression-based acceptance conditions (HOA specs)
│   │   ├── automaton.py        # Core automaton dataclass
│   │   ├── utils.py            # Utility functions
│   │   ├── examples/
│   │   │   ├── nfa.py          # NFA implementation (graph-based)
│   │   │   ├── strel.py        # STREL automaton (spatio-temporal specs)
│   │   │   └── ltl.py          # LTL/LTLf to AFA conversion
│   │   └── hoa/
│   │       ├── parser.py       # HOA format parser
│   │       ├── acc_expr.py     # HOA acceptance expressions
│   │       ├── exporter.py     # HOA format exporter
│   │       └── hoa.lark        # HOA grammar file
│   └── tests/
│       ├── examples/           # Tests for NFA, STREL, LTL
│       └── hoa/                # HOA format test files
│
├── packages/algebraic/         # Multi-backend semiring algebra and polynomials
│   ├── src/algebraic/
│   │   ├── __init__.py         # Canonical public API / re-exports
│   │   ├── ops/                # Backend-aware array operations
│   │   ├── spec.py             # Semiring and algebra interfaces
│   │   ├── semirings.py        # Concrete semiring implementations
│   │   ├── array/              # AlgebraicArray implementation
│   │   ├── kernels/            # Backend-specific kernels and primitives
│   │   └── polynomials/
│   │       ├── dok/            # PolyDict (dict-backed sparse polynomial)
│   │       ├── monomial_basis.py # MonomialBasis (dense tensor)
│   │       └── rank_decomp.py  # RankDecomposition (structured low-rank form)
│   └── tests/
│       ├── array/              # AlgebraicArray tests
│       └── polynomials/        # Polynomial tests
│
└── tests/                      # Automatix tests
    ├── test_weight_functions.py
    ├── operators/
    └── nfa/
```

