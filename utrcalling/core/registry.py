"""Registries: the named methods of utrcalling's extension points.

utrcalling looks some of its methods up by name, as in `--score initiator`
or `--protocol-platform 10x`. Each kind of method has a registry, which maps
names to implementations (see docs/extending.md):

- peak_calling.settings.PEAK_CALLERS: peak-calling methods (`--peak-caller`)
- peak_calling.settings.BANDWIDTH_RULES: rules that set each UTR's bandwidth
  (`-b scott`)
- peak_calling.settings.SCORES: scores of peak-calling settings (`--score`)
- protocol.PROTOCOLS: sequencing protocols (`--protocol-platform`)

A method comes from one of three places:

1. utrcalling itself. The registry names its implementation as
   "module:attribute" and imports it only when it is used, so the command
   line starts fast.
2. Another installed package, which declares it as an entry point of the
   registry's group in its pyproject.toml::

       [project.entry-points."utrcalling.scores"]
       degradation = "my_package.scores:DegradationScore"

3. Code running in the same process, such as a notebook or a test::

       SCORES.register("degradation", DegradationScore)

Names are case-insensitive. utrcalling's own names cannot be taken over.

This module imports only the standard library, and importlib.metadata (40 ms)
only when a name is not utrcalling's own.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Generic, TypeVar, overload

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

T = TypeVar("T")


class Registry(Generic[T]):
    """The named implementations of one extension point.

    Args:
        kind: What it holds, for messages, such as "score".
        group: The entry point group through which other packages add
            implementations, such as "utrcalling.scores".
        builtins: utrcalling's own implementations, by name, as
            "module:attribute" references.
    """

    def __init__(self, kind: str, group: str, builtins: Mapping[str, str]) -> None:
        self.kind = kind
        self.group = group
        self._builtins = tuple(name.lower() for name in builtins)
        # Where to find each name's implementation until it is first used: a
        # "module:attribute" reference, or an entry point.
        self._references: dict[str, str | EntryPoint] = {
            name.lower(): reference for name, reference in builtins.items()
        }
        self._implementations: dict[str, T] = {}
        self._plugins_found = False

    @overload
    def register(self, name: str) -> Callable[[T], T]: ...

    @overload
    def register(self, name: str, implementation: T) -> T: ...

    def register(self, name: str, implementation: T | None = None) -> T | Callable[[T], T]:
        """Add an implementation under *name*, or replace one added before.

        Works as a decorator too::

            @SCORES.register("degradation")
            class DegradationScore(SiteScore): ...

        Raises:
            ValueError: If *name* is one of utrcalling's own.
        """
        if implementation is None:
            return lambda implementation: self.register(name, implementation)
        key = name.lower()
        if key in self._builtins:
            raise ValueError(f"{self.kind} {name!r} is utrcalling's own: choose another name")
        self._references.pop(key, None)
        self._implementations[key] = implementation
        return implementation

    def unregister(self, name: str) -> None:
        """Remove what :meth:`register` added under *name*.

        Raises:
            ValueError: If *name* is one of utrcalling's own, or was not registered.
        """
        key = name.lower()
        if key in self._builtins:
            raise ValueError(f"{self.kind} {name!r} is utrcalling's own: it stays")
        if self._implementations.pop(key, None) is None:
            raise ValueError(f"No {self.kind} {name!r} was registered")

    def names(self) -> tuple[str, ...]:
        """utrcalling's own names, then those of plugins and registrations, sorted."""
        self._find_plugins()
        others = sorted(self._known() - set(self._builtins))
        return (*self._builtins, *others)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if name.lower() not in self._known():
            self._find_plugins()
        return name.lower() in self._known()

    def check(self, name: str) -> str:
        """*name* as the registry knows it, in lower case.

        Raises:
            ValueError: If no implementation has that name.
        """
        if name not in self:
            raise ValueError(
                f"Unknown {self.kind} {name!r}: expected one of {', '.join(self.names())}"
            )
        return name.lower()

    def get(self, name: str) -> T:
        """The implementation of *name*, imported on first use.

        Raises:
            ValueError: If no implementation has that name.
        """
        key = self.check(name)
        if key not in self._implementations:
            reference = self._references.pop(key)
            if isinstance(reference, str):
                module, _, attribute = reference.partition(":")
                implementation = getattr(importlib.import_module(module), attribute)
            else:
                implementation = reference.load()
            self._implementations[key] = implementation
        return self._implementations[key]

    def _known(self) -> set[str]:
        return set(self._references) | set(self._implementations)

    def _find_plugins(self) -> None:
        """Add the entry points of the registry's group, once; names already taken stay."""
        if self._plugins_found:
            return
        from importlib.metadata import entry_points

        self._plugins_found = True
        known = self._known()
        for entry_point in entry_points(group=self.group):
            if entry_point.name.lower() not in known:
                self._references[entry_point.name.lower()] = entry_point
