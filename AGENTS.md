# Agent Instructions

# To set up a development environment with all dependencies (including extras for development and testing),
# use uv (https://github.com/astral-sh/uv):

```python
uv sync --dev extra
uv run python
```

## Coding Philosophy

### Code Style Preferences
- **Pythonic**: Write idiomatic Python code that follows PEP 8 and Python best practices
- **Concise**: Favor brevity and clarity over verbosity
- **Minimal**: Use the simplest solution that works; avoid over-engineering
- **Functional**: Prefer functional programming patterns where appropriate

### Error Handling Philosophy
- **Fail Fast**: Let code fail quickly and visibly rather than hiding errors
- **Minimal Try/Except**: Avoid excessive try/except blocks that mask underlying issues
- **Explicit Failures**: When errors occur, they should be obvious and informative
- **No Silent Failures**: Don't catch exceptions unless you can meaningfully handle them

### Testing Strategy
- **Minimal Testing**: Write only essential tests that catch critical functionality
- **Quality over Quantity**: Focus on meaningful tests rather than high coverage numbers
- **Integration over Unit**: Prefer integration tests that test real workflows
- **Avoid Test Bloat**: Don't write tests for trivial getters/setters or obvious functionality

## Code Generation Guidelines

### What TO Do
- Use list/dict comprehensions instead of explicit loops when clearer
- Leverage Python's built-in functions and standard library
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Use pathlib instead of os.path for file operations
- Prefer f-strings over .format() or % formatting
- Use context managers (with statements) for resource management
- Handle edge cases with early returns rather than nested conditions

### What NOT To Do
- Don't wrap every operation in try/except "just in case"
- Don't write defensive code for inputs that should be validated elsewhere
- Don't add unnecessary abstraction layers
- Don't write tests for trivial operations
- Don't catch broad exceptions (Exception, BaseException) unless absolutely necessary
- Don't suppress errors with pass statements in except blocks
- Don't add configuration options for things that don't need to be configurable
- Avoid over-use of argparse and CLI for simple scripts.

## Specific Patterns

### Preferred Error Handling
```python
# Good: Let it fail fast
def process_file(filepath):
    with open(filepath) as f:  # FileNotFoundError will bubble up naturally
        return json.load(f)    # JSONDecodeError will bubble up naturally

# Avoid: Excessive defensive programming
def process_file(filepath):
    try:
        if not os.path.exists(filepath):
            # Handle missing file
        with open(filepath) as f:
            try:
                return json.load(f)
            except JSONDecodeError:
                # Handle invalid JSON
    except Exception as e:
        # Handle everything else
```

### Preferred Function Design
```python
# Good: Simple, direct, functional
def filter_valid_annotations(annotations: list[dict]) -> list[dict]:
    return [ann for ann in annotations if ann.get('x') and ann.get('y')]

# Avoid: Over-engineered with unnecessary error handling
def filter_valid_annotations(annotations: list[dict]) -> list[dict]:
    try:
        if not annotations:
            return []
        result = []
        for ann in annotations:
            try:
                if ann.get('x') is not None and ann.get('y') is not None:
                    result.append(ann)
            except (KeyError, AttributeError, TypeError):
                continue
        return result
    except Exception:
        return []
```

## Domain-Specific Guidelines

### For Data Processing Scripts
- Use pandas operations instead of manual loops
- Leverage numpy vectorization
- Use pathlib for file path operations
- Let pandas/numpy raise their own exceptions rather than catching them

### For CLI Scripts
- Use argparse for command-line interfaces
- Let missing required arguments fail naturally
- Use sys.exit() for expected failure cases
- Don't catch keyboard interrupts unless necessary

### For API Interactions
- Use requests library directly without excessive retry logic
- Let HTTP errors bubble up unless you can meaningfully handle them
- Use requests.Session for connection pooling when making multiple requests

## Comments and Documentation

### When to Document
- Complex algorithms or business logic
- Non-obvious performance considerations
- API contracts and expected input/output formats
- Workarounds for external library limitations

### When NOT to Document
- Obvious operations (x = x + 1)
- Standard library usage
- Simple getter/setter methods
- Code that is self-explanatory

## Summary

Write code that is clear, direct, and fails obviously when something goes wrong. Prefer simplicity over robustness, and trust that proper system design and monitoring will catch issues rather than trying to handle every possible error case in the code itself.

### File I/O Patterns

#### Preferred File Reading
```python
# Good: Let file operations fail fast and explicitly
def process_annotations(data_dir: str) -> pd.DataFrame:
    treetops_file = os.path.join(data_dir, 'treetops.gpkg')
    return gpd.read_file(treetops_file)  # FileNotFoundError will bubble up clearly

# Good: Use optional return types when files may legitimately not exist
def load_optional_config(config_path: str) -> Optional[dict]:
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None  # Explicit handling of expected case

# Avoid: Silent failures with existence checks
def process_annotations(data_dir: str) -> Optional[pd.DataFrame]:
    treetops_file = os.path.join(data_dir, 'treetops.gpkg')
    if os.path.exists(treetops_file):
        return gpd.read_file(treetops_file)
    return None  # Silent failure makes debugging harder
```

## Adding Datasets to MillionTrees

Follow the docs/contributing.md for adding new datasets. 

# Update the leaderboard

For reproducible update to the leaderboard, run all the scripts in docs/examples. For SLURM systems, a wrapper function is available in slurm/submit_all.sh
The logs are available in /logs and will report the performance of each model. Update the leaderboard with the top-level metrics. 