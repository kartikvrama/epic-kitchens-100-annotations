LIQUID_SUBCLASSES = {
    "liquid",
    "water",
    "oil",
    "sauce",
    "coffee",
    "milk",
    "tea",
    "vinegar",
    "syrup",
    "soup",
    "curry",
    "mixture",
    "spreads",
    "extract",
}

GRANULAR_SUBCLASSES = {
    "flour",
    "rice",
    "cereal",
    "oatmeal",
    "breadcrumb",
    "lentil",
    "yeast",
    "seed",
    "powder",
}

PART_SUBCLASSES = {
    "handle",
    "end",
    "bit",
    "piece",
    "pieces",
    "slice",
    "minced",   
}

SUBCLASSES_EXCLUDED = LIQUID_SUBCLASSES | GRANULAR_SUBCLASSES | PART_SUBCLASSES


CATEGORIES_INCLUDED = {
    "containers",
    "crockery",
    "cutlery",
    "utensils",
    "cookware",
    "materials",
    "cleaning equipment and material",
    "vegetables",
    "fruits and nuts",
    "meat and substitute",
    "dairy and eggs",
    "baked goods and grains",
}
