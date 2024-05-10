from setuptools import setup
from pathlib import Path

if __name__ == "__main__":
    requirements_path = Path(__file__).parent / "requirements.txt"
    requirements_lines = requirements_path.read_text().strip().splitlines()
    setup(
        name="hydragen",
        version="0.0.1",
        packages=["hydragen"],
        install_requires=requirements_lines,
    )
