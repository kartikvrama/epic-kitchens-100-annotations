"""
Objects to exclude from VLM prompting (liquids and fixed entities only).
Derived from nouns_to_ignore_fixed_large_liquids_parts.md sections 1 and 3.
"""

# Fixed entities (section 1)
FIXED_ENTITY_KEYS = {
    "bin",
    "hob",
    "tap",
    "cupboard",
    "drawer",
    "top",
    "sink",
    "rack:drying",
    "kitchen",
    "floor",
    "chair",
    "ladder",
    "wall",
    "shelf",
    "stand",
    "window",
    "candle",
    "airer",
    "door:kitchen",
    "fridge"
}

WASTE_KEYS = {
    "peels",
    "skin",
    "rubbish"
}

# Liquids (section 3)
LIQUID_KEYS = {
    "liquid:washing",
    "water",
    "oil",
    "sauce",
    "coffee",
    "milk",
    "tea",
    "vinegar",
    "liquid",
    "juice",
    "wine",
    "drink",
    "beer",
    "gin",
    "syrup",
    "soda",
    "whiskey",
}

# Combined: skip prompting the VLM for these object keys
OBJECTS_TO_EXCLUDE_FROM_VLM = FIXED_ENTITY_KEYS | LIQUID_KEYS | WASTE_KEYS
