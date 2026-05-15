"""Smoke tests verifying all subpackages are importable."""


def test_import_backends() -> None:
    import divergence.backends  # noqa: F401


def test_import_evals() -> None:
    import divergence.evals  # noqa: F401


def test_import_runner() -> None:
    import divergence.runner  # noqa: F401


def test_import_analysis() -> None:
    import divergence.analysis  # noqa: F401


def test_import_dashboard() -> None:
    import divergence.dashboard  # noqa: F401


def test_import_cli() -> None:
    import divergence.cli  # noqa: F401
