"""NVIDIA NIM settings (fixed values, no env config)."""

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


class NimSettings(BaseModel):
    """Fixed NVIDIA NIM settings (not configurable via env)."""

    temperature: float = Field(
        1.0, ge=0.0, le=2.0, description="Sampling temperature, must be >=0 and <=2."
    )
    top_p: float = Field(
        1.0, ge=0.0, le=1.0, description="Nucleus sampling probability. [0,1]"
    )
    top_k: int = -1
    max_tokens: int = Field(
        ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ge=1,
        description="Maximum number of tokens in output.",
    )
    reasoning_budget: int | None = Field(
        None,
        ge=1,
        description=(
            "Optional hidden reasoning token budget for NIM chat templates. "
            "Unset preserves the provider/model default."
        ),
    )
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    min_p: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum probability threshold [0,1]."
    )
    repetition_penalty: float = Field(
        1.0, ge=0.0, description="Penalty for repeated tokens. Must be >=0."
    )
    seed: int | None = None
    stop: str | None = None
    parallel_tool_calls: bool = True
    ignore_eos: bool = False
    min_tokens: int = Field(0, ge=0, description="Minimum tokens in the response.")
    chat_template: str | None = None
    request_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, value, info: ValidationInfo):
        if value is None or value == "":
            return -1
        integer_value = int(value)
        if integer_value < -1:
            raise ValueError(f"{info.field_name} must be -1 or >= 0")
        return integer_value

    @field_validator(
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        mode="before",
    )
    @classmethod
    def validate_float_fields(cls, value, info: ValidationInfo):
        field_defaults = {
            "temperature": 1.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        if value is None or value == "":
            key = info.field_name or "temperature"
            return field_defaults.get(key, 1.0)
        try:
            float_value = float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{info.field_name} must be a float. Got {type(value).__name__}."
            ) from err
        return float_value

    @field_validator("max_tokens", "min_tokens", "reasoning_budget", mode="before")
    @classmethod
    def validate_int_fields(cls, value, info: ValidationInfo):
        field_defaults = {
            "max_tokens": ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            "min_tokens": 0,
            "reasoning_budget": None,
        }
        if value is None or value == "":
            key = info.field_name or "max_tokens"
            return field_defaults.get(key, ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS)
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{info.field_name} must be an int. Got {type(value).__name__}."
            ) from err
        return integer_value

    @field_validator("seed", mode="before")
    @classmethod
    def parse_optional_int(cls, value, info: ValidationInfo):
        if value == "" or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"{info.field_name} must be an int or empty/None."
            ) from err

    @field_validator("stop", "chat_template", "request_id", mode="before")
    @classmethod
    def parse_optional_str(cls, value, info: ValidationInfo):
        if value == "":
            return None
        if value is not None and not isinstance(value, str):
            return str(value)
        return value
