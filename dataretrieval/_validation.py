"""Argument checks shared by every adapter.

Rejecting a value that is not in a closed vocabulary is the one validation
every adapter does, and it was written eleven times: eight message phrasings
for one concept, so each new check was a coin flip on wording. That is how
:func:`~dataretrieval.waterdata.get_reference_table` came to tell callers who
passed a bad ``collection`` that their *code service* was invalid -- the check
was copied from :mod:`~dataretrieval.waterdata.samples`, message and local
variable name included, and the noun was never changed.

This module owns the wording so a new check cannot invent its own. It is a
leaf with no first-party imports: the vocabularies it validates against live
with the adapters that define them, and only the rejection is shared.

The other shared rule here is :func:`is_integral_count` -- what counts as an
integer argument. It grew the same way the vocabulary check did: the
bool-rejecting ``numbers.Integral`` test was spelled once in the OGC controls
and again in the configuration type policies, so a tightening of one (say,
NumPy handling) could not reach the other.
"""

from __future__ import annotations

import numbers
from collections.abc import Collection


def is_integral_count(value: object) -> bool:
    """True when *value* is an integer in the counting sense.

    Any :class:`numbers.Integral` qualifies -- a NumPy or pandas integer is a
    legitimate count from Python -- but ``bool`` is rejected despite being an
    ``Integral`` subtype: ``True`` is a flag that would silently read as ``1``.
    Type shape only; bounds (and the error voice) stay with the caller, whose
    policy they are.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _render(options: Collection[object]) -> str:
    """Format *options* for a message: ``'a', 'b', 'c'``.

    Renders the values rather than their container so ``dict_keys([...])`` and
    a bare tuple read the same to a caller, who never sees the container.
    """
    return ", ".join(repr(option) for option in options)


def require_one_of(
    value: object,
    options: Collection[object],
    *,
    name: str,
    context: str = "",
) -> None:
    """Raise ``ValueError`` unless *value* is one of *options*.

    Parameters
    ----------
    value
        The argument the caller supplied.
    options
        The closed vocabulary it must belong to -- typically
        ``get_args(SomeLiteral)``, a module constant, or a mapping's keys.
        Rendered in iteration order, so pass a sorted view when the source is
        unordered and the order would otherwise be arbitrary.
    name
        What the value *is*, as the caller's parameter names it (``"service"``,
        ``"collection"``). It becomes the message's subject, so it must match
        the parameter the caller actually passed.
    context
        Optional qualifier for a vocabulary that depends on another argument,
        e.g. ``context="service 'wqp'"`` when the valid profiles differ per
        service.

    Raises
    ------
    ValueError
        If *value* is not in *options*.
    """
    if isinstance(options, str):
        # ``str`` is a Collection, so this type-checks -- and then ``in``
        # silently means "substring", accepting any fragment of a valid option.
        raise TypeError(f"options must be a collection of values, not {options!r}")
    if value in options:
        return
    qualifier = f" for {context}" if context else ""
    raise ValueError(
        f"Invalid {name}: {value!r}{qualifier}. Valid options are: {_render(options)}."
    )
