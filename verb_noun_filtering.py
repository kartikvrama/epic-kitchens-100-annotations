from typing import List
from utils import load_noun_class_names

NOUN_CLASS_NAMES = load_noun_class_names()

VERBS_TIDY_IDLE = {
    "return-to",
    "lay-on",
    "set-down",
    "throw-away",
    "throw-for",
    "put-away",
    "turn-off",
    "return",
    "throw-into",
    "close",
    "throw",
    "screw-lid",
    "return",
    "discard",
    "throw-for"
}

VERBS_IN_USE = {
    "heat",
    "lay-on",
    "put-with",
    "switch-on",
    "wear",
    "turn-on"
}

NOUN_CATEGORIES_TIDY_IDLE = {
    "appliances",
    "furniture",
    "storage",
    "rubbish",
}


def label_verb_noun(verb: str, noun_categories: List[str] | None = None, noun_classes: List[int] | None = None):
    """Labels action as idle/tidy or in use.

    Action is labeled as idle/tidy if:
    - Verb is in VERBS_TIDY_IDLE
    - Any one noun category is in NOUN_CATEGORIES_TIDY_IDLE

    Action is labeled as in use if:
    - Verb is in VERBS_IN_USE

    Else, return "unknown".
    """
    if noun_categories is None and noun_classes is None:
        raise ValueError("Either noun_categories or noun_classes must be provided")
    if noun_categories is not None and noun_classes is not None:
        raise ValueError("Both noun_categories and noun_classes cannot be provided")
    if noun_classes is not None:
        try:
            noun_categories = [NOUN_CLASS_NAMES[int(noun_class)]["category"] for noun_class in noun_classes]
        except KeyError as e:
            print(f"Error: Some noun classes {noun_classes} not found in NOUN_CLASS_NAMES")
            import pdb; pdb.set_trace()
            raise e
    if verb in VERBS_TIDY_IDLE or any(
        noun_category in NOUN_CATEGORIES_TIDY_IDLE
        for noun_category in noun_categories
    ):
        return "idle/tidy"
    if verb in VERBS_IN_USE:
        return "in use"
    # print(f"Verb: {verb_class}, Noun categories: {noun_categories}")
    # import pdb; pdb.set_trace()
    return "unknown"
