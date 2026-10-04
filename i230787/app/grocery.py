"""Deterministic grocery logic: catalog loading, unit normalization, ingredient merging,
categorization, external recipe parsing and plan rendering. No model calls happen here."""
import json
import math
import re
from fractions import Fraction
from functools import lru_cache
from app.config import ROOT
from app.models import Category, GroceryItem, GroceryPlan, Ingredient, PlannedRecipe, Recipe

CATEGORY_ORDER: list[Category] = ['Produce', 'Meat & Seafood', 'Dairy & Eggs', 'Bakery', 'Pantry', 'Frozen', 'Other']

# raw unit spelling -> (canonical unit, factor to base unit, base unit)
_MASS = {'g': 1, 'gram': 1, 'grams': 1, 'kg': 1000, 'kilogram': 1000, 'kilograms': 1000,
         'oz': 28.35, 'ounce': 28.35, 'ounces': 28.35, 'lb': 453.6, 'lbs': 453.6, 'pound': 453.6, 'pounds': 453.6}
_VOLUME = {'ml': 1, 'milliliter': 1, 'milliliters': 1, 'l': 1000, 'liter': 1000, 'liters': 1000, 'litre': 1000,
           'litres': 1000, 'tsp': 5, 'teaspoon': 5, 'teaspoons': 5, 'tbsp': 15, 'tablespoon': 15,
           'tablespoons': 15, 'cup': 240, 'cups': 240}
_COUNT = {'pc': 'pc', 'pcs': 'pc', 'piece': 'pc', 'pieces': 'pc', 'clove': 'clove', 'cloves': 'clove',
          'can': 'can', 'cans': 'can', 'tin': 'can', 'tins': 'can', 'bunch': 'bunch', 'bunches': 'bunch',
          'slice': 'slice', 'slices': 'slice', 'pinch': 'pinch', 'pinches': 'pinch', 'head': 'head',
          'heads': 'head', 'stalk': 'stalk', 'stalks': 'stalk', 'sprig': 'sprig', 'sprigs': 'sprig',
          'package': 'package', 'packages': 'package', 'pack': 'package', 'packs': 'package'}
KNOWN_UNITS = set(_MASS) | set(_VOLUME) | set(_COUNT)

_DESCRIPTORS = {'fresh', 'freshly', 'large', 'small', 'medium', 'chopped', 'diced', 'minced', 'sliced', 'grated',
                'finely', 'roughly', 'peeled', 'ripe', 'boneless', 'skinless', 'cooked', 'raw', 'organic',
                'extra', 'virgin', 'unsalted', 'salted', 'whole', 'shredded', 'softened', 'melted', 'cold',
                'warm', 'room', 'temperature', 'optional', 'about', 'approximately', 'of', 'the'}
_SYNONYMS = {
    'garlic clove': 'garlic', 'clove garlic': 'garlic', 'scallion': 'green onion', 'spring onion': 'green onion',
    'all-purpose flour': 'flour', 'all purpose flour': 'flour', 'plain flour': 'flour', 'caster sugar': 'sugar',
    'granulated sugar': 'sugar', 'white sugar': 'sugar', 'pepper': 'black pepper',
    'ground black pepper': 'black pepper', 'heavy whipping cream': 'heavy cream', 'double cream': 'heavy cream',
    'whipping cream': 'heavy cream', 'minced beef': 'ground beef', 'beef mince': 'ground beef',
    'coriander': 'cilantro', 'coriander leaf': 'cilantro', 'parmesan cheese': 'parmesan',
    'parmigiano reggiano': 'parmesan', 'cheddar cheese': 'cheddar', 'mozzarella cheese': 'mozzarella',
    'feta cheese': 'feta', 'vegetable stock': 'vegetable broth', 'chicken stock': 'chicken broth',
    'kosher salt': 'salt', 'sea salt': 'salt', 'table salt': 'salt', 'egg yolk': 'egg', 'chicken breast fillet': 'chicken breast',
}
_KEEP_PLURAL = ('us', 'ss', 'is')

# Ordered keyword rules; longer/more specific phrases are checked first.
_CATEGORY_KEYWORDS: dict[str, Category] = {
    'coconut milk': 'Pantry', 'peanut butter': 'Pantry', 'black pepper': 'Pantry', 'crushed tomato': 'Pantry',
    'canned tomato': 'Pantry', 'tomato paste': 'Pantry', 'tomato sauce': 'Pantry', 'soy sauce': 'Pantry',
    'baking powder': 'Pantry', 'baking soda': 'Pantry', 'bell pepper': 'Produce', 'green onion': 'Produce',
    'sour cream': 'Dairy & Eggs', 'ice cream': 'Frozen', 'frozen': 'Frozen', 'pizza dough': 'Bakery',
    'broth': 'Pantry', 'stock': 'Pantry', 'oil': 'Pantry', 'vinegar': 'Pantry', 'flour': 'Pantry', 'sugar': 'Pantry',
    'salt': 'Pantry', 'rice': 'Pantry', 'pasta': 'Pantry', 'spaghetti': 'Pantry', 'penne': 'Pantry',
    'noodle': 'Pantry', 'lentil': 'Pantry', 'bean': 'Pantry', 'chickpea': 'Pantry', 'honey': 'Pantry',
    'spice': 'Pantry', 'cumin': 'Pantry', 'paprika': 'Pantry', 'oregano': 'Pantry', 'cinnamon': 'Pantry',
    'curry': 'Pantry', 'seasoning': 'Pantry', 'sauce': 'Pantry', 'dressing': 'Pantry', 'olive': 'Pantry',
    'oat': 'Pantry', 'yeast': 'Pantry', 'vanilla': 'Pantry', 'chocolate': 'Pantry', 'nut': 'Pantry',
    'chicken': 'Meat & Seafood', 'beef': 'Meat & Seafood', 'pork': 'Meat & Seafood', 'bacon': 'Meat & Seafood',
    'turkey': 'Meat & Seafood', 'lamb': 'Meat & Seafood', 'sausage': 'Meat & Seafood', 'ham': 'Meat & Seafood',
    'shrimp': 'Meat & Seafood', 'prawn': 'Meat & Seafood', 'salmon': 'Meat & Seafood', 'tuna': 'Meat & Seafood',
    'fish': 'Meat & Seafood', 'milk': 'Dairy & Eggs', 'cream': 'Dairy & Eggs', 'butter': 'Dairy & Eggs',
    'cheese': 'Dairy & Eggs', 'cheddar': 'Dairy & Eggs', 'parmesan': 'Dairy & Eggs', 'mozzarella': 'Dairy & Eggs',
    'feta': 'Dairy & Eggs', 'yogurt': 'Dairy & Eggs', 'yoghurt': 'Dairy & Eggs', 'egg': 'Dairy & Eggs',
    'bread': 'Bakery', 'bun': 'Bakery', 'tortilla': 'Bakery', 'pita': 'Bakery', 'crouton': 'Bakery',
    'bagel': 'Bakery', 'taco shell': 'Bakery',
    'onion': 'Produce', 'garlic': 'Produce', 'tomato': 'Produce', 'lettuce': 'Produce', 'carrot': 'Produce',
    'potato': 'Produce', 'lemon': 'Produce', 'lime': 'Produce', 'avocado': 'Produce', 'cilantro': 'Produce',
    'basil': 'Produce', 'parsley': 'Produce', 'spinach': 'Produce', 'cucumber': 'Produce', 'mushroom': 'Produce',
    'ginger': 'Produce', 'celery': 'Produce', 'broccoli': 'Produce', 'zucchini': 'Produce', 'apple': 'Produce',
    'banana': 'Produce', 'berry': 'Produce', 'pepper': 'Produce', 'jalapeno': 'Produce', 'kale': 'Produce',
    'cabbage': 'Produce', 'herb': 'Produce', 'mint': 'Produce', 'pea': 'Produce', 'corn': 'Produce',
}
_CATEGORY_RULES = sorted(_CATEGORY_KEYWORDS.items(), key=lambda kv: -len(kv[0]))

_NUMBER_WORDS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'dozen': 12}
_UNICODE_FRACTIONS = {'½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4', '⅛': '1/8'}
_QTY = r'(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?|(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen)(?![a-z]))'
_LINE = re.compile(rf'^(?P<qty>{_QTY})\s*(?P<unit>[a-zA-Z]+\.?)?\s+(?P<name>.+)$', re.IGNORECASE)
_LIST_MARKER = re.compile(r'^\s*(?:[-*•·]+|\d+[.)]|\(\d+\))\s+')
_SERVINGS = re.compile(r'\b(?:serves|servings?|yield|yields|makes|portions?)\b\s*[:=]?\s*(?:about\s+)?(\d{1,2})', re.IGNORECASE)
_SECTION_INGREDIENTS = re.compile(r'^\W*ingredients?\b', re.IGNORECASE)
_SECTION_OTHER = re.compile(r'^\W*(instructions?|method|directions?|steps?|preparation|notes?)\b', re.IGNORECASE)
# Used only to *flag* instruction-like text in untrusted data for observability. It is not the defence:
# untrusted text never reaches the model as instructions and can never authorize an action.
INSTRUCTION_LIKE = re.compile(
    r'\b(ignore|disregard|override|forget|bypass)\b.{0,40}\b(instruction|rule|prompt|polic|previous|above)'
    r'|\bsystem\s*(prompt|message|notice|instruction)|\byou are (now|an?)\b|\b(assistant|ai|agent|model)\s*[,:]'
    r'|\breveal\b|\bjailbreak|\b(place|submit) an? order\b|\bcredit card\b|\bexecute\b|\bdeveloper mode\b',
    re.IGNORECASE)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Recipe]:
    records = json.loads((ROOT / 'data/sample_data.json').read_text(encoding='utf-8'))['records']
    return {r['id']: Recipe.model_validate(r) for r in records}


def normalize_text(text: str) -> str:
    text = text.lower().replace("'", '')
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text).split())


def _singular(word: str) -> str:
    if len(word) <= 3 or word.endswith(_KEEP_PLURAL):
        return word
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('oes'):
        return word[:-2]
    if word.endswith('s'):
        return word[:-1]
    return word


def canonical_name(raw: str) -> str:
    """Lower-case, drop preparation notes and descriptors, singularize, apply synonyms."""
    name = re.sub(r'\(.*?\)', ' ', raw.lower())
    name = re.split(r'[,;]| for | to taste', name)[0]
    words = [w for w in re.sub(r'[^a-z\- ]+', ' ', name).split() if w not in _DESCRIPTORS]
    name = ' '.join(_singular(w) for w in words).strip()
    return _SYNONYMS.get(name, name) or raw.strip().lower()[:60]


def categorize(name: str) -> Category:
    padded = f' {name} '
    for keyword, category in _CATEGORY_RULES:
        if re.search(rf'\b{re.escape(keyword)}', padded):
            return category
    return 'Other'


def to_base(quantity: float | None, unit: str | None) -> tuple[float | None, str | None]:
    """Convert to grams, millilitres or a count unit so that duplicates can be summed."""
    if quantity is None:
        return None, None
    u = (unit or 'pc').lower().rstrip('.')
    if u in _MASS:
        return quantity * _MASS[u], 'g'
    if u in _VOLUME:
        return quantity * _VOLUME[u], 'ml'
    return quantity, _COUNT.get(u, 'pc')


def _parse_quantity(text: str) -> float:
    text = text.strip().lower()
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    if '-' in text:  # ranges such as "2-3": plan for the upper bound
        text = text.split('-')[-1].strip()
    return float(sum(Fraction(part) for part in text.split()))


def parse_ingredient_line(line: str, allow_word_quantity: bool = True) -> Ingredient | None:
    line = _LIST_MARKER.sub('', line).strip()
    for symbol, fraction in _UNICODE_FRACTIONS.items():
        line = line.replace(symbol, f' {fraction}')
    line = ' '.join(line.split())
    match = _LINE.match(line)
    if not match:
        return None
    unit = (match['unit'] or '').lower().rstrip('.')
    name_part = match['name']
    if unit and unit not in KNOWN_UNITS:  # the "unit" was actually the first word of the name
        name_part, unit = f"{match['unit']} {name_part}", ''
    if not unit and not allow_word_quantity and not match['qty'][0].isdigit():
        return None  # "A hearty soup" is a title, not one soup
    name_part = re.sub(r'^of\s+', '', name_part, flags=re.IGNORECASE)
    name = canonical_name(name_part)
    if not name or len(name.split()) > 5:
        return None
    if unit in ('', 'pc', 'pcs') and re.search(r'\bcloves?\b', name_part, re.IGNORECASE):
        unit = 'clove'
    unit = _COUNT.get(unit, unit)
    try:
        quantity = _parse_quantity(match['qty'])
    except (ValueError, ZeroDivisionError):
        return None
    if quantity <= 0:
        return None
    return Ingredient(name=name, quantity=quantity, unit=unit or 'pc', category=categorize(name))


def parse_external_recipe(content: str, recipe_id: str, fallback_name: str) -> tuple[Recipe | None, int, int]:
    """Parse untrusted free text into a sandbox recipe. Returns (recipe, ignored_lines, instruction_like_lines).
    Only quantity-bearing ingredient lines become data; everything else is ignored."""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    title, servings, ingredients = None, None, []
    ignored = instruction_like = 0
    has_sections = any(_SECTION_INGREDIENTS.match(l) for l in lines)
    in_ingredients = not has_sections
    for line in lines:
        if INSTRUCTION_LIKE.search(line):
            instruction_like += 1
        if servings is None and (m := _SERVINGS.search(line)):
            servings = int(m.group(1))
        if _SECTION_INGREDIENTS.match(line):
            in_ingredients = True
            continue
        if _SECTION_OTHER.match(line):
            in_ingredients = False
            continue
        parsed = parse_ingredient_line(line, allow_word_quantity=has_sections) if in_ingredients else None
        if parsed:
            ingredients.append(parsed)
            continue
        if title is None and not ingredients and len(line) <= 80 and not INSTRUCTION_LIKE.search(line):
            title = re.sub(r'^(#+|recipe\s*:)\s*', '', line, flags=re.IGNORECASE).strip(' :') or None
            if title and _SERVINGS.search(title):
                title = _SERVINGS.split(title)[0].strip(' -–:(,') or None
            if title:
                continue
        ignored += 1
    if not ingredients:
        return None, ignored, instruction_like
    recipe = Recipe(id=recipe_id, name=(title or fallback_name)[:80], base_servings=min(servings or 1, 50),
                    ingredients=ingredients[:60], source='external',
                    notes='' if servings else 'Serving count not stated in source; treated as 1.')
    return recipe, ignored, instruction_like


def _round_quantity(quantity: float, unit: str) -> float:
    if unit in ('g', 'ml'):
        return round(quantity, 1)
    return float(math.ceil(quantity - 1e-9))  # whole items are bought, so round counts up


def build_plan(selected: dict[str, int], recipes: dict[str, Recipe], exclude: list[str]) -> GroceryPlan:
    planned, merged, notes = [], {}, []
    for recipe_id, servings in selected.items():
        recipe = recipes[recipe_id]
        planned.append(PlannedRecipe(recipe_id=recipe_id, name=recipe.name, servings=servings))
        factor = servings / recipe.base_servings
        for ingredient in recipe.ingredients:
            name = canonical_name(ingredient.name)
            scaled = None if ingredient.quantity is None else ingredient.quantity * factor
            quantity, unit = to_base(scaled, ingredient.unit)
            entry = merged.setdefault((name, unit), {'quantity': None, 'recipes': [],
                                                    'category': ingredient.category if ingredient.category != 'Other' else categorize(name)})
            if quantity is not None:
                entry['quantity'] = (entry['quantity'] or 0) + quantity
            if recipe.name not in entry['recipes']:
                entry['recipes'].append(recipe.name)
    excluded_names = [canonical_name(e) for e in exclude]
    items, excluded_hits = [], set()
    for (name, unit), entry in merged.items():
        hit = next((e for e in excluded_names if e and (e == name or re.search(rf'\b{re.escape(e)}\b', name))), None)
        if hit:
            excluded_hits.add(hit)
            continue
        quantity = None if entry['quantity'] is None else _round_quantity(entry['quantity'], unit)
        items.append(GroceryItem(name=name, quantity=quantity or None, unit=unit if quantity else None,
                                 category=entry['category'], recipes=entry['recipes']))
    names = [n for n, _ in merged]
    for name in sorted({n for n in names if names.count(n) > 1}):
        notes.append(f'{name} appears in units that cannot be converted, so it is listed more than once.')
    for missing in [e for e in excluded_names if e not in excluded_hits]:
        notes.append(f'"{missing}" was not in the plan, so nothing was excluded for it.')
    items.sort(key=lambda i: (CATEGORY_ORDER.index(i.category), i.name))
    if not items:
        raise ValueError('every ingredient was excluded; nothing left to buy')
    return GroceryPlan(recipes=planned, items=items, excluded_items=sorted(excluded_hits), notes=notes)


def format_quantity(quantity: float | None, unit: str | None) -> str:
    if quantity is None:
        return 'as needed'
    if unit == 'g':
        return f'{quantity / 1000:g} kg' if quantity >= 1000 else f'{quantity:g} g'
    if unit == 'ml':
        if quantity >= 1000:
            return f'{quantity / 1000:g} L'
        if quantity < 15:
            return f'{round(quantity / 5 * 4) / 4:g} tsp'
        if quantity < 60:
            return f'{round(quantity / 15 * 2) / 2:g} tbsp'
        return f'{quantity:g} ml'
    if unit in (None, 'pc'):
        return f'{quantity:g}'
    return f'{quantity:g} {unit}{"s" if quantity > 1 and unit not in ("pinch",) else ""}'


def render_plan(plan: GroceryPlan, max_chars: int = 1700) -> str:
    header = 'Grocery plan for ' + ', '.join(f'{r.name} ({r.servings} servings)' for r in plan.recipes) + ':'
    lines = [header]
    for category in CATEGORY_ORDER:
        group = [i for i in plan.items if i.category == category]
        if group:
            lines.append(f'{category}: ' + '; '.join(f'{i.name} {format_quantity(i.quantity, i.unit)}' for i in group))
    if plan.excluded_items:
        lines.append('Excluded (already have): ' + ', '.join(plan.excluded_items))
    lines.extend(f'Note: {n}' for n in plan.notes)
    text = '\n'.join(lines)
    if len(text) > max_chars:
        text = text[:max_chars - 40].rsplit('\n', 1)[0] + f'\n… ({len(plan.items)} items in total)'
    return text
