"""
VariableMapper — General-purpose variable mapping with unit conversion
and fuzzy name matching.

Maps variables between two naming conventions (e.g. EntityNet attributes
to analysis tool parameters, DB columns to API fields, CSV headers to
model inputs). Integrates with UnitHandler (pint-backed) for automatic
unit conversion during mapping.

Three-tier name resolution:
  1. Exact match — alias string == source key (fastest, no false positives)
  2. Normalized match — lowercase + strip separators + expand abbreviations
  3. Fuzzy match — rapidfuzz WRatio scorer with configurable threshold

Usage:
    from pythonLibs.dataHandler.VariableMapper import VariableMapper

    mapper = VariableMapper()
    mapper.add("chamber_pressure",
               aliases=["Pc", "P_c", "chamber_pressure"],
               source_unit="MPa", target_unit="Pa")
    mapper.add("thrust",
               aliases=["Thrust", "F", "thrust"],
               source_unit="kN", target_unit="N",
               default=100.0)

    result = mapper.map(
        source={"Pc": "7.0", "Thrust": "100"},
        source_units={"Pc": "MPa", "Thrust": "kN"},
    )
    # {"chamber_pressure": 7000000.0, "thrust": 100000.0}

    # Fuzzy matching (enable via fuzzy_threshold):
    mapper2 = VariableMapper(fuzzy_threshold=80)
    mapper2.add("chamber_pressure", aliases=["Pc", "chamber_pressure"])
    result2 = mapper2.map({"ChamberPressure": 7.0})
    # Normalized match: "chamberpressure" matches "chamber_pressure"

    # Match report for transparency:
    result3, report = mapper2.map_with_report({"chamberPres": 7.0})
    # report[0].match_tier = "fuzzy", report[0].score = 85.7

Serialization:
    rules_dict = mapper.to_dict()   # Save as JSON/TOML
    mapper2 = VariableMapper.from_dict(rules_dict)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from pythonLibs.dataHandler.UnitHandler import UnitHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LowConfidenceMatchError(ValueError):
    """Raised when a fuzzy match score is below warn_threshold.

    Attributes:
        target: The target variable name that had a low-confidence match.
        source_key: The source key it tentatively matched.
        score: The fuzzy match score.
        warn_threshold: The threshold it failed to meet.
    """

    def __init__(self, target: str, source_key: str, score: float,
                 warn_threshold: float):
        self.target = target
        self.source_key = source_key
        self.score = score
        self.warn_threshold = warn_threshold
        super().__init__(
            f"Low confidence match: '{target}' ↔ '{source_key}' "
            f"(score={score:.1f}, required≥{warn_threshold}). "
            f"Add '{source_key}' to aliases to accept this match."
        )


class MissingSourceUnitError(ValueError):
    """Raised when source data lacks a required unit for unit conversion.

    This catches EntityNet data quality issues where a variable needs
    unit conversion (e.g. MPa→Pa) but the source data has no unit specified.
    Silently falling back to the rule's default source_unit is dangerous
    because the actual data might be in a different unit.

    Attributes:
        target: The target variable name.
        source_key: The matched source key.
        expected_unit: The unit expected (rule.source_unit).
        target_unit: The target unit for conversion.
    """

    def __init__(self, target: str, source_key: str,
                 expected_unit: str, target_unit: str):
        self.target = target
        self.source_key = source_key
        self.expected_unit = expected_unit
        self.target_unit = target_unit
        super().__init__(
            f"Missing source unit: '{target}' matched '{source_key}' "
            f"but no unit provided in source data. "
            f"Expected '{expected_unit}' for conversion to '{target_unit}'. "
            f"Add the unit to source data or set require_source_unit=False."
        )

# Optional dependencies for fuzzy matching (graceful degradation)
try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

try:
    import inflection
    _HAS_INFLECTION = True
except ImportError:
    _HAS_INFLECTION = False


# ---------------------------------------------------------------------------
# Engineering domain abbreviation map
# ---------------------------------------------------------------------------
ENGINEERING_ABBREVIATIONS: dict[str, str] = {
    # Pressure
    "pc": "chamberpressure", "pe": "exitpressure", "p0": "totalpressure",
    "pa": "ambientpressure",
    # Temperature
    "tc": "chambertemperature", "te": "exittemperature", "t0": "totaltemperature",
    "tinlet": "inlettemperature", "twall": "walltemperature",
    # Thrust & performance
    "isp": "specificimpulse", "cf": "thrustcoefficient",
    "cstar": "characteristicvelocity",
    # Flow
    "mdot": "massflowrate", "of": "mixtureratio", "ofratio": "mixtureratio",
    # Geometry
    "dt": "throatdiameter", "dc": "chamberdiameter", "de": "exitdiameter",
    "at": "throatarea", "ae": "exitarea", "ac": "chamberarea",
    "lc": "chamberlength", "eps": "expansionratio", "ar": "arearatio",
    # Mass
    "mprop": "propellantmass", "mdry": "drymass",
    # Velocity
    "ve": "exitvelocity", "vinf": "freestream velocity",
    # General
    "dia": "diameter", "temp": "temperature", "pres": "pressure",
    "vel": "velocity", "eff": "efficiency", "coeff": "coefficient",
    "len": "length", "vol": "volume", "dens": "density",
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
# Regex to split camelCase/PascalCase into parts
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def normalize_varname(
    name: str,
    abbreviations: dict[str, str] | None = None,
) -> str:
    """Normalize a variable name for comparison.

    Steps:
      1. Split camelCase/PascalCase → space-separated
      2. Replace separators (_, -, .) with space
      3. Lowercase
      4. Strip whitespace and collapse multiple spaces
      5. Expand abbreviations (full-string match on collapsed form)
      6. Singularize via inflection (if available)
      7. Remove all spaces → single comparable token

    Args:
        name: Variable name to normalize.
        abbreviations: Optional abbreviation → expansion map.
            Keys should be lowercase, no separators.

    Returns:
        Normalized lowercase string with no separators.
    """
    # Split camelCase
    s = _CAMEL_SPLIT_RE.sub(" ", name)
    # Replace separators (including / for things like O/F)
    s = s.replace("_", " ").replace("-", " ").replace(".", " ").replace("/", " ")
    # Lowercase and collapse
    s = " ".join(s.lower().split())

    # Collapse to single token for abbreviation lookup
    collapsed = s.replace(" ", "")

    # Abbreviation expansion
    if abbreviations:
        expanded = abbreviations.get(collapsed)
        if expanded:
            return expanded

    # Singularize each word if inflection is available
    if _HAS_INFLECTION:
        words = s.split()
        words = [inflection.singularize(w) for w in words]
        return "".join(words)

    return collapsed


# ---------------------------------------------------------------------------
# MatchReport
# ---------------------------------------------------------------------------
@dataclass
class MatchReport:
    """Report of how a single variable was matched during mapping.

    Attributes:
        target: Target variable name.
        source_key: Matched source key (None if unmatched).
        match_tier: "exact", "exact_ci", "normalized", "fuzzy", or "default"/"missing".
        score: Fuzzy match score (100.0 for exact/normalized, 0.0 for unmatched).
        normalized_alias: The alias form that matched.
        raw_value: The value from source before coercion/conversion.
    """
    target: str
    source_key: str | None = None
    match_tier: str = "missing"
    score: float = 0.0
    normalized_alias: str = ""
    raw_value: Any = None


# ---------------------------------------------------------------------------
# MappingRule
# ---------------------------------------------------------------------------
@dataclass
class MappingRule:
    """A single variable mapping rule.

    Attributes:
        target: Target variable name in the destination namespace.
        aliases: Source variable name candidates (first match wins).
        source_unit: Default source unit. Overridden by per-variable
            source_units dict in map(). Empty string = no unit.
        target_unit: Required unit in target namespace. If set and
            different from source, auto-conversion is applied.
        default: Fallback value when no alias matches in source.
            None means the variable is omitted if not found.
        required: If True and no alias matches and no default,
            raise ValueError during map().
        value_type: Coerce the mapped value to this type.
            Supports float, int, str, bool. None = no coercion.
        description: Human-readable description (for documentation,
            AI agent consumption, schema generation).
        require_source_unit: If "auto" (default), require unit in source
            data when source_unit != target_unit (i.e. unit conversion needed).
            If True, always require. If False, never require (silent fallback).
    """
    target: str
    aliases: list[str] = field(default_factory=list)
    source_unit: str = ""
    target_unit: str = ""
    default: Any = None
    required: bool = False
    value_type: type | None = float
    description: str = ""
    require_source_unit: str | bool = "auto"


# ---------------------------------------------------------------------------
# VariableMapper
# ---------------------------------------------------------------------------
class VariableMapper:
    """Maps variables between two naming conventions with unit conversion.

    General-purpose: EntityNet -> Tool, DB -> API, CSV -> Model, etc.
    Uses UnitHandler (pint-backed) for unit conversion.

    Three-tier name resolution (when fuzzy_threshold > 0):
      1. Exact match — alias == source key
      2. Normalized match — normalize_varname(alias) == normalize_varname(key)
      3. Fuzzy match — rapidfuzz WRatio score >= threshold

    Confidence policy:
      - exact           → OK (silent)
      - normalized      → WARNING (proceed, log warning)
      - fuzzy ≥ warn    → WARNING (proceed, log warning)
      - fuzzy < warn    → ERROR (raise LowConfidenceMatchError)
      - < threshold     → missing (use default or skip)

    Args:
        fuzzy_threshold: Minimum score (0-100) for fuzzy match detection.
            0 = disabled (exact only, default for backward compatibility).
        warn_threshold: Minimum score (0-100) for fuzzy matches to be
            accepted with a warning. Fuzzy matches between fuzzy_threshold
            and warn_threshold raise LowConfidenceMatchError.
            None = same as fuzzy_threshold (all detected matches accepted).
        abbreviations: Domain-specific abbreviation map for normalization.
            None = use ENGINEERING_ABBREVIATIONS. Pass {} to disable.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0,
        warn_threshold: float | None = None,
        abbreviations: dict[str, str] | None = None,
        custom_map: dict[str, str | list[str]] | None = None,
    ):
        self._rules: list[MappingRule] = []
        self._fuzzy_threshold = fuzzy_threshold
        self._warn_threshold = (
            warn_threshold if warn_threshold is not None
            else fuzzy_threshold
        )
        self._abbreviations = (
            abbreviations if abbreviations is not None
            else ENGINEERING_ABBREVIATIONS
        )
        self._custom_map: dict[str, list[str]] = {}
        if custom_map:
            self.set_custom_map(custom_map)

    # ----- Rule definition (fluent API) -----

    def add(
        self,
        target: str,
        aliases: list[str] | str,
        *,
        source_unit: str = "",
        target_unit: str = "",
        default: Any = None,
        required: bool = False,
        value_type: type | None = float,
        description: str = "",
        require_source_unit: str | bool = "auto",
    ) -> VariableMapper:
        """Add a mapping rule. Returns self for chaining.

        Args:
            target: Target variable name.
            aliases: Source variable name(s) to look up.
                Single string or list. First match wins.
            source_unit: Default source unit (overridden by source_units dict).
            target_unit: Target unit. Auto-converts if different from source.
            default: Fallback value if no alias found in source.
            required: Raise error if missing and no default.
            value_type: Coerce value to this type (float, int, str, bool, None).
            description: Human-readable description.
            require_source_unit: If "auto", require unit in source data
                when source_unit != target_unit. True=always, False=never.

        Returns:
            self (for fluent chaining).
        """
        if isinstance(aliases, str):
            aliases = [aliases]
        self._rules.append(MappingRule(
            target=target,
            aliases=aliases,
            source_unit=source_unit,
            target_unit=target_unit,
            default=default,
            required=required,
            value_type=value_type,
            description=description,
            require_source_unit=require_source_unit,
        ))
        return self

    def set_custom_map(self, custom_map: dict[str, str | list[str]]) -> VariableMapper:
        """Set explicit target -> source alias overrides.

        The custom map is applied before each rule's aliases during resolution.
        Example:
            {
              "of_ratio": ["mixture_ratio", "mr"],
              "thrust_kN": "thrust_vacuum"
            }
        """
        normalized: dict[str, list[str]] = {}
        for target, aliases in custom_map.items():
            if isinstance(aliases, str):
                values = [aliases]
            else:
                values = [str(a) for a in aliases]
            values = [v for v in values if v]
            if values:
                normalized[str(target)] = values
        self._custom_map = normalized
        return self

    def update_custom_map(self, custom_map: dict[str, str | list[str]]) -> VariableMapper:
        """Merge explicit target -> source alias overrides."""
        merged = dict(self._custom_map)
        for target, aliases in custom_map.items():
            if isinstance(aliases, str):
                values = [aliases]
            else:
                values = [str(a) for a in aliases]
            values = [v for v in values if v]
            if values:
                merged[str(target)] = values
        self._custom_map = merged
        return self

    # ----- Forward mapping (source → target) -----

    def map(
        self,
        source: dict,
        source_units: dict | None = None,
    ) -> dict:
        """Map source variables to target namespace.

        Args:
            source: Source variable dict {name: value}.
            source_units: Optional per-variable unit dict {name: unit_str}.
                Overrides rule.source_unit for matched variables.

        Returns:
            Target variable dict {target_name: converted_value}.

        Raises:
            ValueError: If a required variable is missing.
        """
        result, _ = self._map_internal(source, source_units, with_report=False)
        return result

    def map_with_report(
        self,
        source: dict,
        source_units: dict | None = None,
    ) -> tuple[dict, list[MatchReport]]:
        """Map source variables and return detailed match report.

        Same as map() but also returns a MatchReport for each rule,
        showing how each variable was resolved (exact/normalized/fuzzy/default).

        Args:
            source: Source variable dict {name: value}.
            source_units: Optional per-variable unit dict {name: unit_str}.

        Returns:
            (result_dict, reports) tuple.
        """
        return self._map_internal(source, source_units, with_report=True)

    def _map_internal(
        self,
        source: dict,
        source_units: dict | None = None,
        with_report: bool = False,
    ) -> tuple[dict, list[MatchReport]]:
        """Internal mapping with optional report generation.

        Confidence policy:
          - exact           → OK
          - normalized      → WARNING (log)
          - fuzzy ≥ warn    → WARNING (log)
          - fuzzy < warn    → ERROR (raise LowConfidenceMatchError)
        """
        source_units = source_units or {}
        result: dict[str, Any] = {}
        reports: list[MatchReport] = []

        for rule in self._rules:
            lookup_aliases = self._get_lookup_aliases(rule)
            raw_value, matched_key, tier, score, matched_alias = self._resolve_value(
                rule, source
            )

            if raw_value is None and matched_key is None:
                # No match found
                if rule.required and rule.default is None:
                    raise ValueError(
                        f"Required variable '{rule.target}' not found. "
                        f"Looked for aliases: {lookup_aliases}"
                    )
                if rule.default is not None:
                    result[rule.target] = rule.default
                    if with_report:
                        reports.append(MatchReport(
                            target=rule.target, match_tier="default",
                        ))
                else:
                    if with_report:
                        reports.append(MatchReport(
                            target=rule.target, match_tier="missing",
                        ))
                continue

            # --- Confidence policy ---
            if tier == "fuzzy" and score < self._warn_threshold:
                # Low confidence → ERROR
                raise LowConfidenceMatchError(
                    target=rule.target,
                    source_key=matched_key,
                    score=score,
                    warn_threshold=self._warn_threshold,
                )

            if tier == "normalized":
                logger.warning(
                    "NORMALIZED_MATCH: '%s' ← '%s' (via normalization)",
                    rule.target, matched_key,
                )
            elif tier == "fuzzy":
                logger.warning(
                    "FUZZY_MATCH: '%s' ← '%s' (score=%.1f)",
                    rule.target, matched_key, score,
                )

            if with_report:
                reports.append(MatchReport(
                    target=rule.target, source_key=matched_key,
                    match_tier=tier, score=score,
                    normalized_alias=matched_alias,
                    raw_value=raw_value,
                ))

            # Coerce type
            value = self._coerce(raw_value, rule.value_type)
            if value is None:
                if rule.default is not None:
                    value = rule.default
                else:
                    continue

            # Unit conversion
            raw_unit = source_units.get(matched_key, "")
            # Treat placeholder units as empty
            if raw_unit and raw_unit.strip() in ("-", "–", "—", "N/A", "n/a", "none"):
                raw_unit = ""

            # Check if source unit is required but missing
            needs_unit_check = (
                rule.require_source_unit is True
                or (rule.require_source_unit == "auto"
                    and rule.source_unit and rule.target_unit
                    and rule.source_unit != rule.target_unit)
            )
            if needs_unit_check and not raw_unit:
                raise MissingSourceUnitError(
                    target=rule.target,
                    source_key=matched_key,
                    expected_unit=rule.source_unit,
                    target_unit=rule.target_unit,
                )

            actual_source_unit = raw_unit or rule.source_unit
            if actual_source_unit and rule.target_unit and actual_source_unit != rule.target_unit:
                value = UnitHandler.convertUnit(
                    value=value,
                    unitNameIn=actual_source_unit,
                    unitNameOut=rule.target_unit,
                )

            result[rule.target] = value

        return result, reports

    # ----- Reverse mapping (target → source) -----

    def reverse_map(
        self,
        target_data: dict,
        source_unit_preference: dict | None = None,
    ) -> tuple[dict, dict]:
        """Map target variables back to source namespace.

        For each rule, converts target_name → first alias.
        If target_unit and source_unit are set, converts units back.

        Args:
            target_data: Target variable dict {target_name: value}.
            source_unit_preference: Optional {alias: unit} to override
                the default source_unit from rules.

        Returns:
            (source_attrs, source_units) tuple.
            source_attrs: {source_alias: converted_value}
            source_units: {source_alias: unit_str}
        """
        source_unit_preference = source_unit_preference or {}
        attrs: dict[str, Any] = {}
        units: dict[str, str] = {}

        for rule in self._rules:
            if rule.target not in target_data:
                continue

            value = target_data[rule.target]
            source_alias = rule.aliases[0] if rule.aliases else rule.target

            # Determine target source unit
            desired_unit = source_unit_preference.get(source_alias, rule.source_unit)

            # Reverse unit conversion
            if rule.target_unit and desired_unit and rule.target_unit != desired_unit:
                value = UnitHandler.convertUnit(
                    value=value,
                    unitNameIn=rule.target_unit,
                    unitNameOut=desired_unit,
                )

            attrs[source_alias] = value
            if desired_unit:
                units[source_alias] = desired_unit

        return attrs, units

    # ----- Validation -----

    def validate(
        self,
        source: dict,
        source_units: dict | None = None,
    ) -> list[str]:
        """Check source data against mapping rules.

        Returns a list of warning/error messages. Empty = all good.
        """
        source_units = source_units or {}
        issues: list[str] = []

        for rule in self._rules:
            lookup_aliases = self._get_lookup_aliases(rule)
            _, matched_key, tier, score, _matched_alias = self._resolve_value(rule, source)

            if matched_key is None:
                if rule.required and rule.default is None:
                    issues.append(
                        f"MISSING: '{rule.target}' — no match for {lookup_aliases}"
                    )
                elif rule.default is None:
                    issues.append(
                        f"OPTIONAL_MISSING: '{rule.target}' — "
                        f"no match for {lookup_aliases}, no default"
                    )
            else:
                # Warn about non-exact matches
                if tier == "fuzzy":
                    issues.append(
                        f"FUZZY_MATCH: '{rule.target}' matched "
                        f"'{matched_key}' (score={score:.1f})"
                    )
                elif tier == "normalized":
                    issues.append(
                        f"NORMALIZED_MATCH: '{rule.target}' matched "
                        f"'{matched_key}' via normalization"
                    )

                # Check missing source unit
                raw_u = source_units.get(matched_key, "")
                if raw_u and raw_u.strip() in ("-", "–", "—", "N/A", "n/a", "none"):
                    raw_u = ""

                needs_unit = (
                    rule.require_source_unit is True
                    or (rule.require_source_unit == "auto"
                        and rule.source_unit and rule.target_unit
                        and rule.source_unit != rule.target_unit)
                )
                if needs_unit and not raw_u:
                    issues.append(
                        f"MISSING_UNIT: '{rule.target}' matched '{matched_key}' "
                        f"but no unit in source data "
                        f"(expected '{rule.source_unit}' for → '{rule.target_unit}')"
                    )

                # Check unit compatibility
                actual_unit = raw_u or rule.source_unit
                if rule.target_unit and actual_unit:
                    src_type = UnitHandler.getUnitType(actual_unit)
                    tgt_type = UnitHandler.getUnitType(rule.target_unit)
                    if src_type and tgt_type and src_type != tgt_type:
                        issues.append(
                            f"UNIT_MISMATCH: '{rule.target}' — "
                            f"source '{actual_unit}' ({src_type}) vs "
                            f"target '{rule.target_unit}' ({tgt_type})"
                        )

        return issues

    def find_missing_units(
        self,
        source: dict,
        source_units: dict | None = None,
    ) -> list[dict]:
        """Find variables that require a source unit but are missing one.

        Returns a list of dicts with keys:
          - target: Target variable name
          - source_key: Matched source key in the data
          - expected_unit: The unit the rule expects (source_unit)
          - target_unit: The conversion target unit

        Useful for automated data repair (e.g. calling FiTsZ updateAttribute
        to add units before retrying the mapping).
        """
        source_units = source_units or {}
        missing: list[dict] = []

        for rule in self._rules:
            _, matched_key, _, _ = self._resolve_value(rule, source)
            if matched_key is None:
                continue

            raw_u = source_units.get(matched_key, "")
            if raw_u and raw_u.strip() in ("-", "–", "—", "N/A", "n/a", "none"):
                raw_u = ""

            needs_unit = (
                rule.require_source_unit is True
                or (rule.require_source_unit == "auto"
                    and rule.source_unit and rule.target_unit
                    and rule.source_unit != rule.target_unit)
            )
            if needs_unit and not raw_u:
                missing.append({
                    "target": rule.target,
                    "source_key": matched_key,
                    "expected_unit": rule.source_unit,
                    "target_unit": rule.target_unit,
                })

        return missing

    # ----- Introspection -----

    def describe(self) -> str:
        """Return a human-readable description of all mapping rules.

        Useful for AI agent consumption and documentation.
        """
        fuzzy_info = ""
        if self._fuzzy_threshold > 0:
            fuzzy_info = (
                f"  (detect≥{self._fuzzy_threshold}, "
                f"accept≥{self._warn_threshold})"
            )
            if not _HAS_RAPIDFUZZ:
                fuzzy_info += " [WARNING: rapidfuzz not installed]"
        lines = [f"Variable Mapping Rules:{fuzzy_info}", ""]
        for i, rule in enumerate(self._rules, 1):
            unit_info = ""
            if rule.source_unit or rule.target_unit:
                unit_strict = ""
                if rule.require_source_unit is True:
                    unit_strict = " *UNIT REQUIRED*"
                elif (rule.require_source_unit == "auto"
                      and rule.source_unit and rule.target_unit
                      and rule.source_unit != rule.target_unit):
                    unit_strict = " *UNIT REQUIRED(auto)*"
                unit_info = f" [{rule.source_unit} -> {rule.target_unit}]{unit_strict}"
            default_info = f" (default: {rule.default})" if rule.default is not None else ""
            required_info = " *REQUIRED*" if rule.required else ""
            desc = f" — {rule.description}" if rule.description else ""
            lines.append(
                f"  {i}. {rule.target} <- {rule.aliases}"
                f"{unit_info}{default_info}{required_info}{desc}"
            )
        return "\n".join(lines)

    @staticmethod
    def describe_report(reports: list[MatchReport]) -> str:
        """Format a list of MatchReport into a human-readable string."""
        lines = ["Match Report:", ""]
        for r in reports:
            if r.match_tier == "exact":
                lines.append(f"  {r.target} <- '{r.source_key}' [exact]")
            elif r.match_tier == "exact_ci":
                lines.append(
                    f"  {r.target} <- '{r.source_key}' "
                    f"[exact_ci alias='{r.normalized_alias}']"
                )
            elif r.match_tier == "normalized":
                lines.append(
                    f"  {r.target} <- '{r.source_key}' "
                    f"[normalized alias='{r.normalized_alias}']"
                )
            elif r.match_tier == "fuzzy":
                lines.append(
                    f"  {r.target} <- '{r.source_key}' "
                    f"[fuzzy alias='{r.normalized_alias}' score={r.score:.1f}]"
                )
            elif r.match_tier == "default":
                lines.append(f"  {r.target} <- (default)")
            else:
                lines.append(f"  {r.target} <- MISSING")
        return "\n".join(lines)

    @property
    def rules(self) -> list[MappingRule]:
        """Access the list of mapping rules (read-only copy)."""
        return list(self._rules)

    @property
    def target_names(self) -> list[str]:
        """List of all target variable names."""
        return [r.target for r in self._rules]

    @property
    def all_aliases(self) -> list[str]:
        """Flat list of all source aliases across all rules."""
        result = []
        for r in self._rules:
            result.extend(r.aliases)
        return result

    # ----- Serialization -----

    def to_dict(self) -> dict:
        """Serialize to a dict (JSON/TOML-safe).

        Returns a dict with 'rules' list and 'config' section.
        value_type is stored as string name ("float", "int", "str", "bool").
        """
        rules = []
        for rule in self._rules:
            d = asdict(rule)
            if rule.value_type is not None:
                d["value_type"] = rule.value_type.__name__
            else:
                d["value_type"] = None
            rules.append(d)
        return {
            "rules": rules,
            "config": {
                "fuzzy_threshold": self._fuzzy_threshold,
                "warn_threshold": self._warn_threshold,
                "custom_map": self._custom_map,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize rules to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict | list) -> VariableMapper:
        """Deserialize from dict or list of rule dicts.

        Accepts both new format (dict with 'rules' + 'config') and
        legacy format (plain list of rule dicts) for backward compatibility.

        Args:
            data: Dict with 'rules'+'config', or list of rule dicts.

        Returns:
            New VariableMapper instance.
        """
        _TYPE_MAP = {"float": float, "int": int, "str": str, "bool": bool}

        # Handle both new and legacy formats
        if isinstance(data, list):
            rules = data
            config = {}
        else:
            rules = data.get("rules", [])
            config = data.get("config", {})

        mapper = cls(
            fuzzy_threshold=config.get("fuzzy_threshold", 0),
            warn_threshold=config.get("warn_threshold"),
            custom_map=config.get("custom_map"),
        )
        for d in rules:
            vt = d.get("value_type")
            if isinstance(vt, str):
                vt = _TYPE_MAP.get(vt, float)
            mapper.add(
                target=d["target"],
                aliases=d.get("aliases", []),
                source_unit=d.get("source_unit", ""),
                target_unit=d.get("target_unit", ""),
                default=d.get("default"),
                required=d.get("required", False),
                value_type=vt,
                description=d.get("description", ""),
                require_source_unit=d.get("require_source_unit", "auto"),
            )
        return mapper

    @classmethod
    def from_json(cls, json_str: str) -> VariableMapper:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ----- Internal helpers -----

    def _get_lookup_aliases(self, rule: MappingRule) -> list[str]:
        """Build ordered alias candidates (custom map first, then rule aliases, then target)."""
        out: list[str] = []
        seen: set[str] = set()
        for alias in self._custom_map.get(rule.target, []):
            if alias and alias not in seen:
                out.append(alias)
                seen.add(alias)
        for alias in rule.aliases:
            if alias and alias not in seen:
                out.append(alias)
                seen.add(alias)
        if rule.target and rule.target not in seen:
            out.append(rule.target)
        return out

    def _resolve_value(
        self,
        rule: MappingRule,
        source: dict,
    ) -> tuple[Any | None, str | None, str, float, str]:
        """Find the best matching source key for a rule.

        Three-tier resolution:
          1. Exact match: alias string == source key
          2. Normalized match: normalize_varname(alias) == normalize_varname(key)
          3. Fuzzy match: rapidfuzz WRatio >= threshold

        Returns:
            (value, matched_source_key, match_tier, score, matched_alias)
            match_tier: "exact", "exact_ci", "normalized", "fuzzy"
            Returns (None, None, "missing", 0.0, "") if no match.
        """
        candidates = self._get_lookup_aliases(rule)

        # --- Tier 1: Exact match ---
        for alias in candidates:
            if alias in source:
                val = source[alias]
                if val is not None and str(val).strip() not in ("", "-"):
                    return val, alias, "exact", 100.0, alias

        # --- Tier 1.5: Case-insensitive exact match ---
        source_lut_ci = {}
        for k, v in source.items():
            if v is None or str(v).strip() in ("", "-"):
                continue
            source_lut_ci.setdefault(str(k).lower(), k)
        for alias in candidates:
            k = source_lut_ci.get(alias.lower())
            if k is not None:
                return source[k], k, "exact_ci", 100.0, alias

        # Stop here if fuzzy is disabled
        if self._fuzzy_threshold <= 0:
            return None, None, "missing", 0.0, ""

        # Pre-filter: source keys with non-empty values
        valid_keys = {
            k: v for k, v in source.items()
            if v is not None and str(v).strip() not in ("", "-")
        }
        if not valid_keys:
            return None, None, "missing", 0.0, ""

        # Build normalized lookup: {normalized_source_key: [original_keys...]}
        norm_source = {}
        for k in valid_keys:
            nk = normalize_varname(k, self._abbreviations)
            norm_source.setdefault(nk, []).append(k)

        def _pick_best(norm_key: str, matched_alias: str) -> str | None:
            candidates = norm_source.get(norm_key, [])
            if not candidates:
                return None
            # 1) case-insensitive exact to alias
            alias_l = matched_alias.lower()
            for c in candidates:
                if c.lower() == alias_l:
                    return c
            # 2) shortest key wins (more canonical), then stable lexical
            return sorted(candidates, key=lambda x: (len(x), x))[0]

        # --- Tier 2: Normalized match ---
        # Also normalize the target name itself as an implicit alias
        for alias in candidates:
            na = normalize_varname(alias, self._abbreviations)
            if na in norm_source:
                orig_key = _pick_best(na, alias)
                if orig_key is not None:
                    return valid_keys[orig_key], orig_key, "normalized", 100.0, alias

        # --- Tier 3: Fuzzy match (rapidfuzz) ---
        if not _HAS_RAPIDFUZZ:
            return None, None, "missing", 0.0, ""

        # Compare normalized alias forms against normalized source keys
        norm_aliases = [
            normalize_varname(a, self._abbreviations) for a in candidates
        ]
        norm_keys_list = list(norm_source.keys())

        best_score = 0.0
        best_norm_key = None
        best_alias = ""

        for na in norm_aliases:
            matches = rf_process.extract(
                na, norm_keys_list,
                scorer=fuzz.WRatio,
                limit=1,
            )
            if matches:
                matched_norm, score, _ = matches[0]
                if score > best_score:
                    best_score = score
                    best_norm_key = matched_norm
                    best_alias = na

        if best_score >= self._fuzzy_threshold and best_norm_key is not None:
            orig_key = _pick_best(best_norm_key, best_alias)
            if orig_key is not None:
                return valid_keys[orig_key], orig_key, "fuzzy", best_score, best_alias

        return None, None, "missing", 0.0, ""

    @staticmethod
    def _coerce(value: Any, target_type: type | None) -> Any | None:
        """Coerce a value to the target type.

        Handles string-to-number conversion, range strings ("300-1800"),
        bool strings, and None passthrough.
        """
        if target_type is None or value is None:
            return value

        if isinstance(value, target_type):
            return value

        s = str(value).strip()

        if target_type == bool:
            return s.lower() in ("true", "1", "yes")

        if target_type in (float, int):
            # Handle range strings: take the first value
            if "-" in s and not s.startswith("-"):
                parts = s.split("-")
                s = parts[0].strip()
            try:
                return target_type(float(s))
            except (ValueError, TypeError):
                return None

        if target_type == str:
            return s

        return value
