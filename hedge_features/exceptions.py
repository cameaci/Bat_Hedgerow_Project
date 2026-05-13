class HedgeFeaturesError(Exception):
    """Base exception."""


class InputValidationError(HedgeFeaturesError):
    """Invalid user input or unsupported format."""


class DatasetResolutionError(HedgeFeaturesError):
    """Dataset path missing or unavailable."""


class OptionalDependencyError(HedgeFeaturesError):
    """Raised when a required optional library is not installed."""

