"""What a value derived from a contract records about where it came from.

A derived value outlives the thing it describes. `review_plan` is measured
against one plan and one hypothesis, and the state it is kept in has no idea
when either of those is replaced -- so a reader that trusts it can be answering
a question about work that no longer exists. That is cache invalidation, and
the treatment is the usual one: let the derived value carry the identity of
what it was derived from, and let the reader check.

The identity is the content, not a stamp. A uuid would have to be put on each
artifact as it is stored, which is one more thing to remember at each new write
site -- the same discipline the check exists to remove. A fingerprint is
computed where it is needed and cannot fall out of step with the value.
"""

from hashlib import sha256

from pydantic import BaseModel


def fingerprint(value: BaseModel) -> str:
    """Identify `value` by its content.

    `model_dump_json` is stable for this: pydantic writes fields in declaration
    order and floats through `repr`, so equal models give equal text in any
    process. Two artifacts that differ anywhere give different fingerprints,
    which is the only property a reader needs.
    """
    return sha256(value.model_dump_json().encode("utf-8")).hexdigest()
