from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Nutrition(BaseModel):
    serving_size: Optional[str] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    sugar_g: Optional[float] = None
    fat_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    fiber_g: Optional[float] = None


class FoodAnalysis(BaseModel):
    is_food: bool = Field(description="Whether the image contains food or a food product.")
    product_name: Optional[str] = Field(default=None, description="Product or food name only if visible/inferable with confidence.")
    label_detected: bool = Field(description="Whether a nutrition facts or ingredients label is visible.")
    nutrition: Nutrition
    ingredients: List[str] = Field(default_factory=list)
    allergens_visible: List[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    missing_fields: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(default=None, description="Short note about blur, glare, crop, uncertainty, or why fields are missing.")
