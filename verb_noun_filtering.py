from typing import List

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

def label_verb_noun(verb: str, noun_categories: List[str]):
    """Labels action as idle/tidy or in use.

    Action is labeled as idle/tidy if:
    - Verb is in VERBS_TIDY_IDLE
    - Any one noun category is in NOUN_CATEGORIES_TIDY_IDLE

    Action is labeled as in use if:
    - Verb is in VERBS_IN_USE

    Else, return "unknown".
    """
    if verb in VERBS_TIDY_IDLE or any(
        noun_category in NOUN_CATEGORIES_TIDY_IDLE for noun_category in noun_categories
    ):
        return "idle/tidy"
    if verb in VERBS_IN_USE:
        return "in use"
    # print(f"Verb: {verb_class}, Noun categories: {noun_categories}")
    # import pdb; pdb.set_trace()
    return "unknown"
