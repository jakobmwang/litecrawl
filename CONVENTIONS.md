# Python Project Conventions

## Tooling & Dependencies

### Project Definition

* Use **`pyproject.toml` as the single source of truth** for:

  * Project metadata
  * Runtime dependencies
  * Dev/test/tooling dependencies
* Do **not** maintain separate `requirements.txt`, `requirements-dev.txt`, etc.

### Dependency Management (uv)

* Use **`uv`** for all dependency and environment management.

* Synchronize your environment with the project config:

  ```bash
  uv sync
  ```

  Run this:

  * After pulling changes from main
  * After editing `pyproject.toml` manually

* Add or update dependencies:

  ```bash
  # Runtime dependency
  uv add <package>

  # Dev-only dependency (tests, linters, tools)
  uv add --dev <package>
  ```

* Remove dependencies:

  ```bash
  uv remove <package>
  ```

* Do **not** manually edit lockfiles unless you know what you’re doing; prefer `uv add/remove/sync`.

---

## Development Workflow

### Command Execution

* Use `uv run` for all script and command execution inside the project:

  ```bash
  uv run python -m my_package.some_module
  uv run my-script
  ```

### Before Committing

Always run, in this order:

```bash
uv run ruff format
uv run ruff check --fix
uv run pytest
```

* Fix formatter/linter issues before committing.
* All tests **must pass** before pushing.
* If tests fail, fix them immediately—do not commit failing tests.
* Keep a tight `.gitignore` to avoid any unnecessary clutter in the repository.

---

## Code Style

### Documentation

* Use **Google-style docstrings** for all public functions, classes, and modules.
* Avoid tutorial-style comments that explain *what* the code does.
* Comments should explain **why**, not **what**.

Example:

```python
async def process_items(items: list[Item]) -> list[Result]:
    """Process items and return results.

    Args:
        items: List of items to process.

    Returns:
        List of processed results.

    Raises:
        ValueError: If items list is empty.
    """
    if not items:
        raise ValueError("items must not be empty")

    return await batch_process(items)
```

### Type Annotations

* Fully type-annotate:

  * Function and method signatures
  * Public variables and module-level constants
* Target **Python 3.12+** syntax:

  * Use built-in generics: `list[T]`, `dict[K, V]`, `set[T]`
  * Use `X | Y` instead of `Union[X, Y]`
  * Use `X | None` instead of `Optional[X]`

Example:

```python
def fetch_data(url: str, timeout: float = 30.0) -> dict[str, Any] | None:
    ...
```

### Programming Paradigm

* Prefer **simple, composable functions** over complex classes.
* Use OOP when it models the domain clearly (not by default).
* Favor:

  * Immutability when practical
  * Pure functions for core logic
  * Composition over inheritance
* Use **`dataclasses`** or Pydantic models for structured data.

---

## Imports, Versions, and API Changes

* **Always verify the latest version and documentation** of any external package **before using, importing, or pinning** it.
* **Check import paths and module layouts**—packages frequently reorganize modules between minor versions.
* When adding code that relies on a package:
  * Confirm that the API, class names, and function signatures match the current stable docs.
  * Prefer **explicit version pinning** only after verifying compatibility.
* When upgrading a dependency:
  * Re-check imports and public APIs for breaking changes.
  * Update code accordingly and run the full quality gate (`ruff` + `pytest`).

**Goal:** Prevent stale imports, avoid deprecated APIs, and ensure builds remain reproducible across environments.

---

## Asynchronous Code

* Use `async`/`await` for I/O-bound operations.
* Be consistent in using `asyncio` patterns.
* Always properly `await` coroutines; avoid fire-and-forget unless explicitly intentional and documented.

---

## Performance

* Write straightforward code first; **profile before optimizing**.
* Choose appropriate data structures:

  * `set` for membership checks
  * `deque` for queues
  * `dict` for keyed lookups
* Prefer comprehensions over manual loops when clearer:

  ```python
  names = [user.name for user in users if user.is_active]
  ```
* For large datasets, consider generators/iterators to reduce memory usage.

---

## Testing

### Test Execution

* Run tests with:

  ```bash
  uv run pytest
  ```
* All tests must pass before merging to main.

### Test Style

* Tests follow the same style rules as production code.
* Use descriptive names that capture **scenario** and **expectation**:

  ```python
  def test_fetch_data_returns_none_on_timeout():
      ...
  ```
* For async code, use `pytest-asyncio` (or equivalent):

```python
@pytest.mark.asyncio
async def test_fetch_data_returns_valid_json():
    """fetch_data returns properly formatted JSON."""
    result = await fetch_data("https://api.example.com/data")
    assert isinstance(result, dict)
    assert "id" in result
```

---

## Code Organization

### Module Structure

* Keep modules **small and cohesive**.

* Prefer several small modules over one large “utils” module.

* Use clear, descriptive filenames.

* Import order:

  1. Standard library
  2. Third-party
  3. Local modules

  Each group separated by a blank line.

### Function Design

* Functions should:

  * Do **one thing** and do it well
  * Have descriptive names (`calculate_total_price` > `calc`)
  * Avoid long parameter lists (use dataclasses or config objects instead)
* Prefer **early returns** to reduce nesting and improve readability.

---

## Summary

* **Dependencies & envs:** managed solely via `pyproject.toml` + `uv add / uv remove / uv sync`.
* **Commands:** `uv run` for everything.
* **Quality gate:** `ruff format` → `ruff check --fix` → `pytest` before every commit.
* **Style:** modern Python (3.12+), fully typed, clear and minimal comments.
* **Design:** small, focused functions and modules; async for I/O; test everything that matters.