from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def read_file(file_path: str) -> str:
    """
    Read the contents of a file in the project.

    Use this when inspecting source code, configuration,
    requirements, notebooks, or other project files.
    """

    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    if not path.is_file():
        return f"Path is not a file: {file_path}"

    try:
        return path.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        return (
            f"Could not read {file_path}: "
            "file is not UTF-8 text."
        )

    except Exception as e:
        return (
            f"Error reading {file_path}: "
            f"{type(e).__name__}: {e}"
        )


# List Files
@tool
def list_files(directory: str = ".") -> str:
    """
    List files and directories inside a project directory.

    Use this to understand the project structure.
    """

    path = Path(directory)

    if not path.exists():
        return f"Directory not found: {directory}"

    if not path.is_dir():
        return f"Path is not a directory: {directory}"

    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".ipynb_checkpoints",
    }

    try:

        items = []

        for item in sorted(path.iterdir()):

            if item.name.startswith("."):
                continue

            if item.is_dir():
                items.append(f"[DIR]  {item.name}/")

            else:
                items.append(f"[FILE] {item.name}")

        if not items:
            return f"{directory} is empty."

        return "\n".join(items)

    except Exception as e:

        return (
            f"Error listing directory: "
            f"{type(e).__name__}: {e}"
        )


# Search Code
@tool
def search_code(
    search_term: str,
    directory: str = "."
) -> str:
    """
    Search for a text string across source code files.

    Use this to find functions, classes, variables,
    imports, error messages, or package names.
    """

    root = Path(directory)

    if not root.exists():
        return f"Directory not found: {directory}"

    if not root.is_dir():
        return f"Path is not a directory: {directory}"

    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".ipynb_checkpoints",
    }

    extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".md",
    }

    matches = []

    try:

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if any(
                part in ignored_dirs
                for part in path.parts
            ):
                continue

            if path.suffix.lower() not in extensions:
                continue

            try:

                lines = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()

                for line_number, line in enumerate(
                    lines,
                    start=1
                ):

                    if (
                        search_term.lower()
                        in line.lower()
                    ):

                        matches.append(
                            f"{path}:{line_number}: "
                            f"{line.strip()}"
                        )

            except Exception:
                continue

        if not matches:
            return (
                f"No matches found for: "
                f"{search_term}"
            )

        max_results = 50

        result = "\n".join(
            matches[:max_results]
        )

        if len(matches) > max_results:

            result += (
                f"\n\n... "
                f"{len(matches) - max_results} "
                "additional matches omitted."
            )

        return result

    except Exception as e:

        return (
            f"Search error: "
            f"{type(e).__name__}: {e}"
        )

# Installed Packages
@tool
def installed_packages() -> str:
    """
    Show Python packages installed in the
    current Python environment.
    """

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "list",
                "--format=columns"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:

            return (
                "Could not retrieve installed packages.\n"
                f"STDERR:\n{result.stderr}"
            )

        return (
            f"Python executable: "
            f"{sys.executable}\n\n"
            f"{result.stdout}"
        )

    except subprocess.TimeoutExpired:

        return (
            "Timed out while retrieving "
            "installed packages."
        )

    except Exception as e:

        return (
            f"Error: "
            f"{type(e).__name__}: {e}"
        )


# Web Search
tavily = TavilySearch()

@tool
def web_search(query: str) -> str:
    """
    Search the web for technical documentation,
    error messages, package issues, GitHub discussions,
    and troubleshooting information.

    Use this when local investigation is insufficient.
    """

    try:

        results = tavily.invoke(query)

        return str(results)

    except Exception as e:

        return (
            f"Web search error: "
            f"{type(e).__name__}: {e}"
        )

# Debug Tools
debug_tools = [
    read_file,
    list_files,
    search_code,
    installed_packages,
    web_search,
]