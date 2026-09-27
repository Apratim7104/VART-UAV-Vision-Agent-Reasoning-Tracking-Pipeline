# Contributing to UAV Florence-2 Multi-Agent Pipeline

Thank you for considering contributing! Here are the guidelines:

## Development Setup

```bash
git clone https://github.com/<your-username>/uav_florence2_agent.git
cd uav_florence2_agent
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Pull Request Process

1. Fork the repository and create a feature branch: `git checkout -b feature/my-feature`
2. Write or update tests in `tests/` for any new functionality
3. Ensure `pytest tests/ -v` passes
4. Commit with clear messages and open a Pull Request against `main`

## Code Style

- Follow PEP 8 and use 4-space indentation
- Type-annotate all public functions and methods
- Add docstrings to all public classes and functions

## Reporting Bugs

Please open a GitHub Issue with:
- Python version, OS, GPU info
- Exact command and error traceback
- Minimal reproducible example if possible
