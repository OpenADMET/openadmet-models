from datetime import date
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class Metadata(BaseModel):
    version: Literal["v1"] = Field(
        ..., description="The version of the metadata schema."
    )
    name: str = Field(..., description="The name of the workflow.")
    build_number: int = Field(
        ...,
        ge=0,
        description="The build number of the workflow (must be non-negative).",
    )
    description: str = Field(..., description="Description of the workflow.")
    tag: str = Field(..., description="Primary tag for the workflow.")
    authors: str = Field(..., description="Name of the authors.")
    email: EmailStr = Field(..., description="Email address of the contact person.")
    date_created: date = Field(
        ..., alias="date-created", description="Date when the workflow was created."
    )
    biotargets: list[str] = Field(
        ..., description="List of biotargets associated with the workflow."
    )
    tags: list[str] = Field(..., description="Additional tags for the workflow.")
