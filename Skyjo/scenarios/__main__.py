"""List the available scenarios: ``python -m Skyjo.scenarios``."""

import importlib
from pathlib import Path
from typing import List, Tuple

from Skyjo.scenarios.scenario import Scenario

_SCENARIO_DIR = Path(__file__).parent
_NOT_A_SCENARIO = {"scenario"}


def discover_scenarios() -> List[Tuple[str, Scenario]]:
    """Return ``(module_name, scenario)`` for every scenario file, sorted by name."""
    found = []
    for path in sorted(_SCENARIO_DIR.glob("*.py")):
        if path.stem.startswith("_") or path.stem in _NOT_A_SCENARIO:
            continue
        module = importlib.import_module(f"Skyjo.scenarios.{path.stem}")
        scenario = getattr(module, "SCENARIO", None)
        if isinstance(scenario, Scenario):
            found.append((path.stem, scenario))
    return found


def main() -> None:
    scenarios = discover_scenarios()
    print(f"{len(scenarios)} scenarios:\n")
    for module_name, scenario in scenarios:
        print(f"  {scenario.name}")
        print(f"    python -m Skyjo.scenarios.{module_name}")
    print("\nAdd --full-game to keep playing after the scenario round.")


if __name__ == "__main__":
    main()
