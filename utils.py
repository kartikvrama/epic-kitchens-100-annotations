verbs_to_ignore = [
    "turn-off",
    "switch-off",
    "shut-off",
    "water-off",
    "tap-off",
    "let-go",
    "let",
    "finish",
    "end",
    "stop",
    "wait",
    "wait-for",
    "decide"
]

nouns_to_ignore = [
    'tap',
    'cupboard',
    'drawer',
    'fridge',
    'hob',
    'water',
    'bin',          # Includes recycling, trash, dustbin
    'oven',
    'sink',
    'dishwasher',
    'microwave',
    'rack:drying',
    'freezer',
    'light',       # Often a fixed switch/fixture interaction
    'machine:washing',
    'plug',        # Fixed wall socket interaction
    'kitchen',     # General environment
    'floor',
    'fan:extractor',
    'knob',        # part of fixed appliance (oven/hob)
    'ladder',      # Furniture/tool often treated as fixed context
    'wall',
    'shelf',
    'window',
    'heater',      # Fixed appliance
    'power',       # Abstract/Fixed switch
    'airer',       # Furniture
    'door:kitchen'
]