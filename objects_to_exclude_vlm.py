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
    "fridge",
}

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
    "syrup",
}

SMALL_PARTS_KEYS = {
    "lid",
    "cover",
    "button",
    "alarm",
    "cap",
    "plug",
    "knob",
    "handle",
    "wire",
    "control:remote",
    "battery",
    "cork",
}

WASTE_KEYS = {
    "peels",
    "skin",
    "rubbish"
}

OBJECTS_TO_EXCLUDE_FROM_VLM = FIXED_ENTITY_KEYS | LIQUID_KEYS | WASTE_KEYS | SMALL_PARTS_KEYS
