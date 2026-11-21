"""The metadata object every getter returns alongside its DataFrame.

A dependency-free leaf on purpose (ADR 0003). This class is the second half of
the ``(DataFrame, metadata)`` return contract (ADR 0007), so nearly every
service module needs it; importing it imports nothing but ``httpx``.

``dataretrieval.utils.BaseMetadata`` remains the public import.
"""

from __future__ import annotations

from typing import Any

import httpx


class _NoSiteInfoError(AttributeError, NotImplementedError):
    """``site_info`` read on metadata that has none.

    An ``AttributeError`` so ``hasattr``, ``getattr(..., default)`` and
    tab-completion treat the attribute as absent instead of raising. Also a
    ``NotImplementedError`` because that is what ``BaseMetadata.site_info``
    raised before, and the class is public as ``utils.BaseMetadata``.
    """


class BaseMetadata:
    """Base class for metadata.

    Attributes
    ----------
    url : str
        Response url.
    query_time: datetime.timedelta
        Response elapsed time.
    header: httpx.Headers
        Response headers.

    """

    def __init__(self, response: httpx.Response) -> None:
        """Generate a standard set of metadata from the response.

        Parameters
        ----------
        response: ``httpx.Response``
            Response object from the ``httpx`` module.

        """

        # Coerce httpx.URL -> str: BaseMetadata.url has always been str.
        self.url = str(response.url)
        self.query_time = response.elapsed
        self.header = response.headers
        self.comment: str | None = None

    @property
    def site_info(self) -> Any:
        raise _NoSiteInfoError(
            "This metadata object has no site_info: the getter that produced "
            "this result does not return site descriptions alongside data. "
            "Call the getter in the same module that returns sites or "
            "monitoring locations to fetch them."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={self.url})"


# Pickles created after this class moved must remain readable by releases where
# its public import path was only ``dataretrieval.utils.BaseMetadata``.
BaseMetadata.__module__ = "dataretrieval.utils"
