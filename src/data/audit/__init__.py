"""Rule-based B-rep trust audit for CAD ground-truth solids.

Kept as its own subpackage (rather than flat under src/data/) since it is a
standalone geometry-QA tool, not part of the DXF/image dataloader that
src/data/__init__.py builds -- deliberately not re-exporting anything here so
importing e.g. ``src.data.audit.solid_checks`` does not pull in more than the
unavoidable ``src.data`` parent package init.

See gt_audit.py for the CLI entry point and README.md for how to run it.
"""
