from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class VerticalOperatorTransformSpec(BaseModel):
    """Declarative built-in transform for applying a vertical operator."""

    kind: Literal["vertical_operator"] = Field(
        description="Discriminator identifying this transform as a vertical operator.",
    )
    mode: Literal["none", "uniform", "ak", "pwf", "ak_pwf", "integration", "tccon"] = (
        Field(
            description="Built-in vertical-operator mode used to interpret the levels and values arrays."
        )
    )
    levels: list[float] = Field(
        default_factory=list,
        description="Vertical coordinates paired with the operator values.",
    )
    values: list[float] = Field(
        default_factory=list,
        description="Operator values applied at the specified vertical levels.",
    )
    pressure_levels: list[float] = Field(
        default_factory=list,
        description="Optional pressure grid used by pressure-based operator modes.",
    )
    coordinate: str = Field(
        default="xhgt",
        description="Trajectory column used as the vertical coordinate when applying the operator.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata stored alongside the transform specification.",
    )

    @model_validator(mode="after")
    def _validate_operator_shape(self) -> Self:
        """Validate that operator level and value arrays have matching lengths."""
        if self.mode not in {"none", "uniform"}:
            if not self.levels or not self.values:
                raise ValueError(
                    "Vertical operator transforms require non-empty levels and "
                    "values unless mode is 'none' or 'uniform'."
                )
            if len(self.levels) != len(self.values):
                raise ValueError(
                    "Vertical operator transform levels and values must have the "
                    "same length."
                )
        return self


class FirstOrderLifetimeTransformSpec(BaseModel):
    """Declarative built-in transform for first-order lifetime decay."""

    kind: Literal["first_order_lifetime"] = Field(
        description="Discriminator identifying this transform as first-order lifetime decay.",
    )
    lifetime_hours: float = Field(
        gt=0,
        description="E-folding lifetime, in hours, used for exponential decay.",
    )
    time_column: str = Field(
        default="time",
        description="Trajectory column containing the elapsed transport time.",
    )
    time_unit: str = Field(
        default="min",
        description="Unit for the trajectory time column, typically minutes or hours.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata stored alongside the transform specification.",
    )


ParticleTransformSpec = Annotated[
    VerticalOperatorTransformSpec | FirstOrderLifetimeTransformSpec,
    Field(discriminator="kind"),
]
