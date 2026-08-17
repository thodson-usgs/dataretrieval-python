"""Deprecated alias for :mod:`dataretrieval.nwdc`.

The module was named for one subset of what the service offers. The National
Water Availability Assessment Data Companion serves ten modeled datasets, of
which the water-use models are five; the rest are hydrologic,
atmospheric-forcing, and assessment outputs. Every other adapter in this
package is named for its service -- ``ngwmn``, ``nldi``, ``wqp``,
``streamstats``, ``nwis`` -- so this one is now ``nwdc``.

Importing this module emits a :class:`DeprecationWarning` and re-exports
:mod:`dataretrieval.nwdc`'s public surface. The re-exported objects are the
*same objects*, not copies -- ``wateruse.get_wateruse is nwdc.get_wateruse``
-- so calls and identity comparisons behave identically through either
spelling.

It is an alias for reading, not a second name for the module. This is a
distinct module object holding its own references to the five public names,
so it does not forward *assignment* or private names: rebinding
``wateruse.get_wateruse`` leaves ``nwdc``'s global untouched (and so has no
effect on anything ``nwdc`` does internally), and ``wateruse._WATERUSE_HOST``
does not exist. Code that monkeypatches, or that reaches for a private, must
name :mod:`dataretrieval.nwdc` directly -- which is the point of the
deprecation.

``dataretrieval.__init__`` deliberately imports :mod:`dataretrieval.nwdc`
rather than this module, so ``import dataretrieval`` stays silent. The warning
fires only for code that names ``wateruse`` itself.
"""

from __future__ import annotations

from dataretrieval import nwdc as _nwdc
from dataretrieval._deprecation import REMOVALS, warn_deprecated
from dataretrieval.nwdc import *  # noqa: F403  (re-export the public surface)

#: When the alias may be deleted. Read from the shared horizon table rather
#: than spelled here, so it is audited and bumped with every other published
#: removal; matches the dated-removal convention :mod:`dataretrieval.nwis`
#: uses.
NWDC_RENAME_REMOVAL_DATE = REMOVALS["wateruse"]

__all__ = list(_nwdc.__all__)


def _warn(stacklevel: int = 1) -> None:
    """Emit this alias's advisory, attributed *stacklevel* frames up.

    Shared by the two ways the alias is reached, because each can attribute
    the warning honestly only from where it runs. Importing this module runs
    the call below, where the importing line is unreachable: the import
    machinery's own frames sit between it and here, and no fixed stacklevel
    crosses them across Python versions -- so it points here, which is what
    the default filters then hide outside ``__main__``. Reaching the alias as
    ``dataretrieval.wateruse`` instead runs
    :func:`dataretrieval.__getattr__`, which *is* called straight from the
    caller's frame and passes a stacklevel that lands there.
    """
    warn_deprecated(
        "`dataretrieval.wateruse`",
        replacement="`dataretrieval.nwdc`",
        removal=NWDC_RENAME_REMOVAL_DATE,
        detail="The service is the National Water Availability Assessment Data "
        "Companion, and water use is one of the ten datasets it serves.",
        stacklevel=stacklevel,
    )


_warn()
