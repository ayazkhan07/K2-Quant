"""
Strategy frame nomenclature: load contract JSON and validate column references in code.

Expand strategy_frame_contract.json when the runner adds or renames columns.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_CONTRACT_PATH = Path(__file__).resolve().parent / "strategy_frame_contract.json"


def load_contract() -> Dict[str, Any]:
    with open(_CONTRACT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _string_keys_from_slice(slc: ast.AST) -> List[str]:
    """Extract string literal keys from a subscript slice (df['a'], df.loc[i, 'b'])."""
    if isinstance(slc, ast.Index):  # Python 3.8
        slc = slc.value
    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
        return [slc.value]
    if isinstance(slc, ast.Str):  # pragma: no cover — py<3.8
        return [slc.s]
    if isinstance(slc, ast.Tuple):
        out: List[str] = []
        for elt in slc.elts:
            out.extend(_string_keys_from_slice(elt))
        return out
    return []


def _subscript_chain_root(node: ast.AST) -> Optional[str]:
    """Leftmost Name in a subscript target (df, df_sorted, ...)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _subscript_chain_root(node.value)
    return None


def extract_dataframe_string_subscripts(tree: ast.AST, roots: Set[str]) -> Set[str]:
    """Collect string column keys used in roots[...] or roots.loc[...][...]."""
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        root = _subscript_chain_root(node.value)
        if root is None or root not in roots:
            continue
        for key in _string_keys_from_slice(node.slice):
            found.add(key)
    return found


def check_nomenclature(
    code: str, contract: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings) for column naming vs contract.

    errors  — legacy / UI aliases (must use canonical name); blocks save when merged into validate_strategy
    warnings — unknown column names (expand contract or confirm intentional); optional strict mode
    """
    if contract is None:
        contract = load_contract()

    roots = set(contract.get("dataframe_roots") or ["df", "data"])
    canonical = set(contract.get("canonical_columns") or [])
    aliases: Dict[str, str] = dict(contract.get("aliases") or {})
    strict_unknown = bool(contract.get("strict_unknown_columns"))

    errors: List[str] = []
    warnings: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []

    used = extract_dataframe_string_subscripts(tree, roots)

    for name in sorted(used):
        if name in canonical:
            continue
        if name in aliases:
            canon = aliases[name]
            errors.append(
                f"Nomenclature: use canonical column {canon!r} instead of {name!r} "
                f"(strategy frame contract / runner)."
            )
            continue
        msg = (
            f"Nomenclature: {name!r} is not listed in strategy_frame_contract.json. "
            f"If it is a valid execution-frame column, add it to canonical_columns; "
            f"if it is a legacy display name, add an aliases entry."
        )
        if strict_unknown:
            errors.append(msg)
        else:
            warnings.append(msg)

    return errors, warnings
