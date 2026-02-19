# EPIC-100 Nouns to Ignore: Fixed Entities, Large Appliances, Liquids, and Small Parts

This list identifies EPIC-100 noun classes that should be **ignored** for object-centric or manipulation-focused analysis when the target is **movable, graspable, or manipulable objects** (e.g. food, small utensils, portable containers). Each entry includes the noun **key** (class name), **id** in the class list, and a short **reason**.

Categories:
1. **Fixed entities** – Built-in or fixed structures (walls, windows, cupboards, sinks, counters).
2. **Large electronic devices / appliances** – Ovens, fridges, dishwashers, etc., which are fixtures or too large to treat as manipulated objects.
3. **Liquids** – Water, oil, milk, sauces, drinks, etc., which are substances rather than discrete objects.
4. **Small parts of larger objects** – Handles, caps, switches, plugs, knobs, etc., that are components rather than standalone objects.

---

## 1. Fixed entities

| id | key | Reason |
|----|-----|--------|
| 0 | **tap** | Fixed plumbing fixture; part of the sink/kitchen. Not a portable object. |
| 3 | **cupboard** | Fixed storage/cabinet; part of the kitchen structure. |
| 8 | **drawer** | Fixed furniture component; the drawer is part of a unit, not a single object. |
| 42 | **top** | Work surface (counter, table, bench). Fixed furniture, not an object. |
| 63 | **sink** | Fixed plumbing fixture; basin and drain are part of the room. |
| 110 | **rack:drying** | Drying rack; often fixed to wall or sink, or a fixed part of the kitchen. |
| 157 | **kitchen** | The room itself; not an object. |
| 159 | **floor** | Fixed; part of the building. |
| 182 | **chair** | Fixed (or heavy) furniture; typically not manipulated like a kitchen tool. |
| 230 | **ladder** | Fixed or structural when in place; not a small manipulable object. |
| 242 | **wall** | Fixed structure; tiles and wall are part of the room. |
| 247 | **shelf** | Fixed storage surface (fridge shelf, cupboard shelf); part of a larger unit. |
| 249 | **stand** | Stand (e.g. kettle stand, charging dock) is often fixed or a fixture. |
| 263 | **window** | Fixed; part of the room. Blinds are attached to the window. |
| 280 | **candle** | Listed under furniture; often fixed (e.g. on shelf) and not a manipulated food/utensil. |
| 285 | **airer** | Clothes airer / drying rack; fixed or static furniture. |
| 295 | **door:kitchen** | Fixed; part of the room. |

---

## 2. Large electronic devices / appliances

| id | key | Reason |
|----|-----|--------|
| 12 | **fridge** | Large fixed appliance; refrigerator and its door are part of the kitchen. |
| 24 | **hob** | Cooktop / stove; fixed cooking surface with burners. |
| 44 | **kettle** | Often treated as an appliance; can be grouped with large devices for consistency. |
| 46 | **oven** | Large fixed appliance; door and interior are parts of the oven. |
| 50 | **maker:coffee** | Coffee machine (and moka, cafetière as part of setup); medium–large appliance. |
| 65 | **heat** | Refers to temperature/settings (heat cooker, heat oven), not a physical object. |
| 70 | **dishwasher** | Large fixed appliance. |
| 90 | **microwave** | Large fixed appliance. |
| 108 | **scale** | Kitchen scale is an electronic device; often left in place. |
| 113 | **freezer** | Large fixed appliance (or part of fridge-freezer). |
| 124 | **machine:washing** | Washing machine; large fixed appliance. |
| 147 | **processor:food** | Food processor; medium–large counter appliance. |
| 152 | **cooker:slow** | Slow cooker / rice cooker / electric pot; medium–large appliance. |
| 179 | **fan:extractor** | Extractor hood / ventilation; fixed above hob. |
| 186 | **toaster** | Toaster; counter appliance, often left in place. |
| 225 | **hoover** | Vacuum cleaner; appliance, not a kitchen “object” in the same sense as cutlery. |
| 245 | **tv** | TV / stereo / audio; large fixed or static device. |
| 250 | **machine:sous:vide** | Sous vide machine; appliance. |
| 267 | **heater** | Heater / radiator; fixed or static. |
| 278 | **power** | Power level/setting; abstract, not a physical object. |
| 288 | **computer** | Computer (and mouse); non-kitchen or static device. |
| 297 | **camera** | Camera; recording device, not a manipulated kitchen object. |
| 298 | **cd** | CD; media, often grouped with appliances in annotations. |

---

## 3. Liquids

| id | key | Reason |
|----|-----|--------|
| 22 | **liquid:washing** | Washing liquid, detergent, soap; liquid substance, not a discrete object. |
| 27 | **water** | Liquid; not a graspable object. |
| 31 | **oil** | Cooking oil; liquid. |
| 47 | **sauce** | Sauces, ketchup, dressings; liquid or semi-liquid. |
| 59 | **coffee** | Coffee as drink or ground; when “coffee” means the beverage, it’s a liquid. |
| 64 | **milk** | Liquid. |
| 132 | **tea** | Tea as beverage; liquid. |
| 134 | **vinegar** | Liquid. |
| 150 | **liquid** | Generic liquid (whey, spill); substance, not object. |
| 183 | **juice** | Liquid. |
| 192 | **wine** | Liquid (beverage / cooking wine). |
| 209 | **drink** | Drinks, smoothies; liquid. |
| 231 | **beer** | Liquid. |
| 264 | **gin** | Liquid (spirit). |
| 279 | **syrup** | Syrup; liquid. |
| 283 | **soda** | Coke, lemonade, tonic; liquid. |
| 294 | **whiskey** | Liquid (spirit). |

---

## 4. Small parts of larger objects

| id | key | Reason |
|----|-----|--------|
| 6 | **lid** | Lid of pot, jar, box, etc.; part of a container, not standalone. |
| 89 | **cover** | Cover for plate, blender, hob, etc.; part of another object. |
| 92 | **button** | Button / switch (e.g. light switch); control part of appliance or room. |
| 111 | **alarm** | Timer, oven clock; part of oven/microwave, not a separate object. |
| 114 | **light** | Hob light, oven light, lamp; part of appliance or fixture. |
| 118 | **cap** | Bottle cap, spice cap; small part of a bottle/container. |
| 153 | **plug** | Plug, socket; part of electrical setup. |
| 190 | **knob** | Knob, dial (heat, oven); control part of appliance. |
| 191 | **handle** | Handle of pan, drawer, cupboard; part of a larger object. |
| 204 | **wire** | Cable, lead, power cord; part of an appliance. |
| 221 | **control:remote** | Remote control; small part of TV/appliance system. |
| 233 | **battery** | Battery; small component, often inside a device. |
| 256 | **cork** | Cork / stopper; small part of a bottle. |
| 272 | **watch** | Watch; worn object, not a kitchen-manipulated object. |
| 109 | **rest** | Spoon rest, teabag holder; small accessory, part of workspace. |
| 77 | **filter** | Coffee filter, sink filter; small part of machine or sink. |

---

## Summary counts

- **Fixed entities:** 17 noun classes  
- **Large appliances / devices:** 24 noun classes  
- **Liquids:** 18 noun classes  
- **Small parts:** 16 noun classes  

Total: **75** noun classes recommended to ignore for the stated use case (focus on movable, graspable, or manipulable objects, excluding fixed structures, large appliances, liquids, and small components).

---

## Optional / borderline

- **bin** (36): Fixed bin vs portable bin; can be ignored if only fixed bins are of interest.  
- **phone** (165): Small device; ignore if you want to exclude non-food, non-utensil objects.  
- **blender** (82): Counter appliance; include if you treat it as a manipulated object, ignore if grouping with large appliances.  
- **tap** (0): Listed under “Fixed” but could also be grouped with “appliances” depending on taxonomy.

Use this list as a filter (e.g. by `key` or `id`) when loading `EPIC_100_noun_classes.csv` or `EPIC_100_noun_classes_v2.csv` to obtain the reduced set of nouns for your task.
