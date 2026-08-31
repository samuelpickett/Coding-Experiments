from pydantic import BaseModel

class Ingredient(BaseModel):
    name: str
    amount: float
    unit: str

class Recipe(BaseModel):
    title: str
    prep_time_minutes: int
    cook_time_minutes: int
    servings:int
    ingredients: list[Ingredient]
    instruction: list[str]
    alterations: list[str]
    tips: list[str]