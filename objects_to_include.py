"""
Objects to exclude from VLM prompting (liquids and fixed entities only).
Derived from nouns_to_ignore_fixed_large_liquids_parts.md sections 1 and 3.
"""

CATEGORIES_TO_INCLUDE = {
    "containers",
    "crockery",
    "cutlery",
    "utensils",
    # "vegetables",
    "appliances",
    "cookware",
    "materials",
    # "spices and herbs and sauces",
    "cleaning equipment and material",
    # "dairy and eggs",
    # "meat and substitute",
    # "fruits and nuts",
    # "drinks"
}