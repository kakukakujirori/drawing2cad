class ToolFeedbackError(Exception):
    """A tool failure the model caused and can correct.

    The tool node routes on this class, so it says who the error is for rather
    than what went wrong; the message carries the latter. Anything else raised
    from a tool is a defect in the pipeline and ends the run instead of being
    handed to the model as if it were advice.

    Raise it only when the tool's success type cannot carry a failure.
    `run_shell` and `verify_output` return a status field, which keeps their
    detail structured, and so they never raise this.
    """
