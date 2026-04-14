"""
Strategy code validator for K2 Quant.

Enforces structural AND quality requirements before a strategy can be saved.
Used from strategy_service.save_strategy (all save paths).

Returns (passed, errors, warnings). Warnings do not block save; errors do.
"""

import ast
from typing import List, Tuple

from k2_quant.utilities.strategy_frame_nomenclature import check_nomenclature


def validate_strategy(code: str) -> Tuple[bool, List[str], List[str]]:
    """Validate strategy code against all required rules.

    Returns (True, [], warnings) if the code passes, or (False, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Code has a syntax error: {e}")
        return (False, errors, warnings)

    _check_no_unicode(code, errors)

    calls = _extract_calls(tree)

    _check_report_header(code, calls, errors)
    _check_report_config(code, calls, errors)
    _check_report_table(code, calls, errors)
    _check_to_forecast(code, calls, errors)
    _check_forecast_documentation(code, calls, errors)

    nom_err, nom_warn = check_nomenclature(code)
    errors.extend(nom_err)
    warnings.extend(nom_warn)

    return (len(errors) == 0, errors, warnings)


def format_errors(errors: List[str]) -> str:
    """Format validation errors into a readable block."""
    lines = ["Strategy validation failed:", ""]
    for i, err in enumerate(errors, 1):
        lines.append(f"  {i}. {err}")
    lines.append("")
    lines.append("Fix these issues and try saving again.")
    return "\n".join(lines)


def format_warnings(warnings: List[str]) -> str:
    """Format non-blocking nomenclature / contract notices."""
    if not warnings:
        return ""
    lines = ["Strategy frame contract notices (save allowed):", ""]
    for i, w in enumerate(warnings, 1):
        lines.append(f"  {i}. {w}")
    lines.append("")
    lines.append("Update k2_quant/utilities/strategy_frame_contract.json if these are expected.")
    return "\n".join(lines)


# -- AST helpers ----------------------------------------------------------

def _extract_calls(tree: ast.Module) -> List[ast.Call]:
    """Walk the AST and return all Call nodes."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _check_no_unicode(code: str, errors: List[str]) -> None:
    """Reject code containing non-ASCII characters."""
    for line_num, line in enumerate(code.split('\n'), 1):
        for col, ch in enumerate(line):
            if ord(ch) > 127:
                snippet = line.strip()[:60]
                errors.append(
                    f"Non-ASCII character '{ch}' (U+{ord(ch):04X}) found on "
                    f"line {line_num}, column {col + 1}. Strategy code must "
                    f"use only ASCII characters. Line: {snippet}"
                )
                return


def _calls_to(calls: List[ast.Call], name: str) -> List[ast.Call]:
    """Filter calls to a specific function name."""
    out = []
    for c in calls:
        fn = c.func
        if isinstance(fn, ast.Name) and fn.id == name:
            out.append(c)
        elif isinstance(fn, ast.Attribute) and fn.attr == name:
            out.append(c)
    return out


def _get_string_arg(call: ast.Call, index: int) -> str:
    """Try to extract a constant string from positional arg at index."""
    if index < len(call.args):
        arg = call.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return ""


def _get_list_length(call: ast.Call, index: int) -> int:
    """Try to get the length of a list literal at positional arg index.
    Returns -1 if the arg is not a static list (e.g. a variable or comprehension)."""
    if index < len(call.args):
        arg = call.args[index]
        if isinstance(arg, ast.List):
            return len(arg.elts)
        if isinstance(arg, ast.ListComp):
            return -1  # dynamic, can't count -- assume OK
    return -1


# -- individual checks ---------------------------------------------------

def _check_report_header(code: str, calls: List[ast.Call],
                         errors: List[str]) -> None:
    header_calls = _calls_to(calls, "report_header")
    if not header_calls:
        errors.append(
            "Missing report_header() call. Every strategy must start with "
            "report_header('Strategy Name') to produce a titled report."
        )
        return

    title = _get_string_arg(header_calls[0], 0)
    if not title or len(title.strip()) < 3:
        errors.append(
            "report_header() has an empty or too-short title. "
            "Provide a meaningful strategy name (at least 3 characters)."
        )


def _check_report_config(code: str, calls: List[ast.Call],
                         errors: List[str]) -> None:
    config_calls = _calls_to(calls, "report_config")
    if not config_calls:
        errors.append(
            "Missing report_config() call. Every strategy must include "
            "report_config([(name, value, description), ...]) listing all "
            "configuration parameters and their meanings."
        )
        return

    first = config_calls[0]
    n_params = _get_list_length(first, 0)

    if n_params == 0:
        errors.append(
            "report_config() is called with an empty list. "
            "You must list at least 2 configuration parameters "
            "with (name, value, description) tuples."
        )
    elif n_params != -1 and n_params < 2:
        errors.append(
            f"report_config() has only {n_params} parameter(s). "
            "Every strategy must document at least 2 configuration parameters. "
            "Include all tunable values with meaningful descriptions."
        )

    if n_params > 0 or n_params == -1:
        _check_config_descriptions(first, errors)


def _check_config_descriptions(call: ast.Call, errors: List[str]) -> None:
    """Check that config tuples have non-empty descriptions."""
    if not call.args:
        return
    arg = call.args[0]
    if not isinstance(arg, ast.List):
        return

    placeholder_patterns = [
        "param", "placeholder", "todo", "tbd", "xxx", "desc", "description",
        "fill in", "update this",
    ]

    for elt in arg.elts:
        if isinstance(elt, ast.Tuple) and len(elt.elts) >= 3:
            desc_node = elt.elts[2]
            if isinstance(desc_node, ast.Constant) and isinstance(desc_node.value, str):
                desc = desc_node.value.strip()
                if len(desc) < 5:
                    errors.append(
                        "report_config() contains a parameter with an empty or "
                        "too-short description. Every parameter must have a "
                        "meaningful description (at least 5 characters) explaining "
                        "what it controls."
                    )
                    return
                desc_lower = desc.lower()
                for pat in placeholder_patterns:
                    if desc_lower == pat or desc_lower.startswith(pat + " "):
                        errors.append(
                            f"report_config() contains a placeholder description "
                            f"('{desc}'). Replace it with a real explanation of "
                            f"what the parameter controls."
                        )
                        return


def _check_report_table(code: str, calls: List[ast.Call],
                        errors: List[str]) -> None:
    table_calls = _calls_to(calls, "report_table")
    if not table_calls:
        errors.append(
            "Missing report_table() call. Every strategy must include at "
            "least one report_table(title, headers, rows) call to show "
            "step-by-step computed values in structured tables."
        )
        return

    for tc in table_calls:
        title = _get_string_arg(tc, 0)
        n_headers = _get_list_length(tc, 1)
        n_rows = _get_list_length(tc, 2)

        if title and n_headers == 0:
            errors.append(
                f"report_table('{title}') has an empty headers list. "
                "Every table must have at least 2 column headers."
            )
            return

        if title and n_headers != -1 and 0 < n_headers < 2:
            errors.append(
                f"report_table('{title}') has only {n_headers} column header(s). "
                "Tables must have at least 2 columns to be meaningful."
            )
            return

        if title and n_rows == 0:
            errors.append(
                f"report_table('{title}') has an empty rows list. "
                "Every table must have at least 1 data row. Use computed "
                "values from the strategy, not empty placeholders."
            )
            return


def _check_to_forecast(code: str, calls: List[ast.Call],
                       errors: List[str]) -> None:
    forecast_calls = _calls_to(calls, "to_forecast")
    if not forecast_calls:
        errors.append(
            "Missing to_forecast() call. Every strategy must write price "
            "projections to the Forecast tab using to_forecast(column_name, values)."
        )


def _check_forecast_documentation(code: str, calls: List[ast.Call],
                                  errors: List[str]) -> None:
    forecast_calls = _calls_to(calls, "to_forecast")
    if not forecast_calls:
        return

    lower = code.lower()
    has_doc = False
    for pattern in [
        "forecast column",
        "forecast output",
        "columns written",
        "writing to forecast",
        "forecast documentation",
    ]:
        if pattern in lower:
            has_doc = True
            break

    table_calls = _calls_to(calls, "report_table")
    for tc in table_calls:
        title = _get_string_arg(tc, 0).lower()
        if any(kw in title for kw in ["forecast", "column", "output", "written"]):
            has_doc = True
            break

    if not has_doc:
        errors.append(
            "Missing forecast column documentation. Include a report_table() "
            "call that lists the forecast columns being written (name and length) "
            "so the report shows what was published to the Forecast tab. "
            "Use a title containing 'Forecast' (e.g. 'Forecast Columns Written')."
        )
