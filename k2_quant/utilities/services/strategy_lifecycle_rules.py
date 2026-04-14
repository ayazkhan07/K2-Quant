"""
Strategy lifecycle rules — catalog for UI and persistence behavior.

Implementations live in ``StrategyService`` and ``AnalysisPageWidget`` /
``OutputsPanel``. When adding new strategy-related invariants, extend this
document and wire the code in the referenced locations.

---------------------------------------------------------------------------
RULE 1 — Strategy deletion cascades to run history
---------------------------------------------------------------------------
When a saved strategy is removed via ``StrategyService.delete_strategy``,
every row in ``strategy_runs`` with the same ``strategy_name`` is deleted in
the **same database transaction** as the ``strategies`` row.

Rationale: the Outputs tab lists runs by strategy name; orphaned runs would
show names that no longer exist in the Strategies panel.

---------------------------------------------------------------------------
RULE 2 — Outputs panel refresh after deletion
---------------------------------------------------------------------------
When the left pane emits ``strategy_deleted``, the analysis page calls
``OutputsPanel.handle_strategy_deleted`` so the run tree reloads from the DB
and any open report/code for that strategy is cleared without a save prompt.

When Thinkspace uses ``delete_strategy``, the table controller removes rows first;
``RightPaneWidget`` emits ``strategy_removed_remotely`` so the analysis page runs
the same outputs refresh and projection cleanup without calling
``delete_strategy`` again.

---------------------------------------------------------------------------
RULE 3 — Run records are keyed by name
---------------------------------------------------------------------------
``strategy_runs.strategy_name`` is a plain text key (not a foreign key).
Renaming a strategy does not rewrite old runs; only deletion uses the name
for bulk removal.

---------------------------------------------------------------------------
RULE 4 — Orphan run cleanup on OUTPUTS refresh
---------------------------------------------------------------------------
``OutputsPanel.refresh()`` calls ``prune_orphan_strategy_runs()`` so any run
rows left behind after an older client or a partial failure are removed.
``get_strategy_names_with_runs()`` only returns names that still exist in
``strategies`` (INNER JOIN), so the tree cannot show a deleted strategy.
"""
