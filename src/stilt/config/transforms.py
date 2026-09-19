from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class VerticalOperatorTransformSpec(BaseModel):
    """Declarative built-in transform for applying a vertical operator."""

    kind: Literal["vertical_operator"] = Field(
        description="Discriminator identifying this transform as a vertical operator.",
    )
    mode: Literal["none", "uniform", "ak", "pwf", "ak_pwf"] = Field(
        description=(
            "Built-in vertical-operator mode. 'ak' interpolates an averaging "
            "kernel from levels/values; 'pwf' weights particles by the air mass "
            "they represent, derived from their release pressures; 'ak_pwf' "
            "does both."
        )
    )
    levels: list[float] = Field(
        default_factory=list,
        description="Vertical coordinates of the averaging-kernel values ('ak' and 'ak_pwf' only).",
    )
    values: list[float] = Field(
        default_factory=list,
        description="Normalized averaging-kernel values at the specified levels ('ak' and 'ak_pwf' only).",
    )
    surface_pressure: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Surface pressure (hPa) closing the bottom of the column for pressure "
            "weighting. Estimated from the particles when omitted."
        ),
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
        if self.mode in {"ak", "ak_pwf"}:
            if not self.levels or not self.values:
                raise ValueError(
                    "Vertical operator transforms with an averaging kernel "
                    "('ak', 'ak_pwf') require non-empty levels and values."
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
