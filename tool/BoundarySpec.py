# BoundarySpec.py — World Chunk boundary specification from @secure_expose metadata
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ActionSpec", "BoundarySpec", "WorldChunk"]


@dataclass(frozen=True)
class ActionSpec:
    """Single action (command) with STRIPS-like precondition/effect."""
    id: str
    description: str = ""
    category: str = "other"
    params: list[dict[str, Any]] = field(default_factory=list)
    precondition: dict[str, Any] = field(default_factory=dict)
    effect: dict[str, Any] = field(default_factory=dict)

    # ------ Planning helpers ------
    def propositions_read(self) -> set[str]:
        """Proposition names checked by precondition."""
        return set(self.precondition.keys())

    def propositions_written(self) -> set[str]:
        """Proposition names modified by effect."""
        return set(self.effect.keys())

    def applicable(self, state: dict[str, Any]) -> bool:
        """Check whether this action is applicable in *state*."""
        for prop, expected in self.precondition.items():
            if prop not in state:
                return False
            if state[prop] != expected:
                return False
        return True

    def apply(self, state: dict[str, Any], result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a new state after applying this action's effect.

        Values containing ``{param}`` placeholders are substituted from *result*.
        A value of ``None`` removes the proposition from state.
        """
        new_state = dict(state)
        for prop, val in self.effect.items():
            if val is None:
                new_state.pop(prop, None)
                continue
            if isinstance(val, str) and "{" in val and result:
                for k, v in result.items():
                    val = val.replace(f"{{{k}}}", str(v))
            new_state[prop] = val
        return new_state

    def to_dict(self) -> dict[str, Any]:
        """Return the action as a dict, leaving out empty fields."""
        d: dict[str, Any] = {"id": self.id}
        if self.description:
            d["description"] = self.description
        if self.category != "other":
            d["category"] = self.category
        if self.params:
            d["params"] = self.params
        if self.precondition:
            d["precondition"] = self.precondition
        if self.effect:
            d["effect"] = self.effect
        return d


@dataclass
class BoundarySpec:
    """World Chunk boundary specification — the declarative interface of a service.

    Generated from a ``buildCatalog()`` result, this captures every action's
    precondition/effect and computes aggregate proposition sets.
    """
    service_name: str
    actions: list[ActionSpec] = field(default_factory=list)

    # ------ Aggregate propositions ------
    @property
    def propositions_read(self) -> set[str]:
        """Propositions checked by any action's precondition."""
        s: set[str] = set()
        for a in self.actions:
            s |= a.propositions_read()
        return s

    @property
    def propositions_written(self) -> set[str]:
        """Propositions changed by any action's effect."""
        s: set[str] = set()
        for a in self.actions:
            s |= a.propositions_written()
        return s

    @property
    def all_propositions(self) -> set[str]:
        """Every proposition the service reads or writes."""
        return self.propositions_read | self.propositions_written

    # ------ Query helpers ------
    def actions_by_category(self) -> dict[str, list[ActionSpec]]:
        """Group the actions by their catalog category."""
        by_cat: dict[str, list[ActionSpec]] = {}
        for a in self.actions:
            by_cat.setdefault(a.category, []).append(a)
        return by_cat

    def applicable_actions(self, state: dict[str, Any]) -> list[ActionSpec]:
        """Return actions whose preconditions are satisfied by *state*."""
        return [a for a in self.actions if a.applicable(state)]

    def action(self, action_id: str) -> ActionSpec | None:
        """Return the action with this id, or None."""
        for a in self.actions:
            if a.id == action_id:
                return a
        return None

    # ------ Construction from catalog ------
    @classmethod
    def from_catalog(cls, catalog: list[dict[str, Any]], service_name: str = "") -> BoundarySpec:
        """Build a BoundarySpec from a ``buildCatalog()`` result list."""
        actions = []
        for entry in catalog:
            actions.append(ActionSpec(
                id=entry["id"],
                description=entry.get("description", ""),
                category=entry.get("category", "other"),
                params=entry.get("params", []),
                precondition=entry.get("precondition", {}),
                effect=entry.get("effect", {}),
            ))
        return cls(service_name=service_name, actions=actions)

    def to_dict(self) -> dict[str, Any]:
        """Return the service name, the sorted propositions and the actions as dicts."""
        return {
            "service_name": self.service_name,
            "propositions": sorted(self.all_propositions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class WorldChunk:
    """A World Chunk — a bounded context with its own state and boundary spec.

    This is the runtime companion to BoundarySpec: it pairs the declarative
    interface with a mutable world-state dictionary.
    """
    boundary: BoundarySpec
    state: dict[str, Any] = field(default_factory=dict)

    def applicable_actions(self) -> list[ActionSpec]:
        """Actions whose preconditions hold in the current state."""
        return self.boundary.applicable_actions(self.state)

    def apply_action(self, action_id: str, result: dict[str, Any] | None = None) -> bool:
        """Apply an action's effect to the chunk state. Returns False if not applicable."""
        action = self.boundary.action(action_id)
        if action is None or not action.applicable(self.state):
            return False
        self.state = action.apply(self.state, result)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return the boundary spec and a copy of the state."""
        return {
            "boundary": self.boundary.to_dict(),
            "state": dict(self.state),
        }
