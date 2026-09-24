"""Convenience entry point for the complete Section 1 experiment."""

from comp70049.phishing.run import main


if __name__ == "__main__":
    # Keep the repository-level command small; implementation lives in the
    # importable package where it can be tested and reused by the notebook.
    main()
