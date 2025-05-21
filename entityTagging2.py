import pandas as pd
import re

# Load input CSV
df = pd.read_csv('./cleaned_openfoodfacts.csv', nrows=10000, sep=',', encoding='utf-8')

# --- Helper functions ---

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

flavors = {'beef', 'cheese', 'chicken', 'duck', 'lamb', 'salmon', 'tuna', 'turkey', 'fish', 'oceanfish', 'shrimp', 'crab', 'mackerel', 'venison', 'rabbit', 'pork'}
nutrients = {'protein', 'fat', 'fiber', 'carbohydrate', 'vitamin', 'calcium', 'iron', 'omega', 'sodium', 'phosphorus', 'magnesium', 'potassium', 'zinc', 'taurine', 'choline', 'biotin', 'folate', 'niacin', 'pantothenic acid', 'riboflavin', 'thiamin', 'vitamin a', 'vitamin b1', 'vitamin b2', 'vitamin b6', 'vitamin b9', 'vitamin b12', 'vitamin c', 'vitamin d', 'vitamin e', 'vitamin k'}
additives = {'bha', 'bht', 'preservatives', 'colorant', 'emulsifier', 'stabilizer', 'flavoring', 'antioxidant', 'carrageenan', 'guar gum', 'xanthan gum', 'locust bean gum', 'sodium tripolyphosphate', 'potassium sorbate', 'sodium benzoate', 'citric acid', 'lecithin', 'cellulose', 'maltodextrin', 'monosodium glutamate', 'disodium inosinate', 'disodium guanylate'}
preservatives = {'sodium nitrite', 'calcium propionate', 'sorbic acid', 'sodium bisulfite', 'potassium metabisulfite'}
sweeteners = {'sucralose', 'stevia', 'fructose', 'glucose syrup', 'maltodextrin', 'aspartame'}
colors = {'caramel color', 'paprika extract', 'annatto', 'curcumin', 'beet extract', 'turmeric'}
minerals = {'chloride', 'sulfate', 'iodide', 'selenium', 'manganese', 'copper'}
probiotics = {'bacillus coagulans', 'lactobacillus acidophilus', 'bifidobacterium', 'saccharomyces boulardii'}
units = {'g', 'mg', 'mcg', '%', 'UI', 'kg', 'ml'}
thickeners = {'pectin', 'agar', 'alginate', 'gum acacia', 'carboxymethylcellulose'}
enzymes = {'protease', 'amylase', 'lipase', 'lactase', 'papain'}
acidifiers = {'lactic acid', 'malic acid', 'tartaric acid', 'phosphoric acid'}
humectants = {'glycerol', 'sorbitol', 'propylene glycol'}

quantity_patterns = [r"\d+(\.\d+)?\s?(mg|g|kg|mcg|ml|l|%)"]

def classify_token(word):
    lw = word.lower()
    if any(re.fullmatch(pat, lw) for pat in quantity_patterns):
        return "QTY"
    elif lw in nutrients:
        return "NUT"
    elif lw in flavors:
        return "FLA"
    elif lw in additives:
        return "ADD"
    elif lw in preservatives:
        return "PRE"
    elif lw in sweeteners:
        return "SWE"
    elif lw in colors:
        return "COL"
    elif lw in minerals:
        return "MIN"
    elif lw in probiotics:
        return "PRO"
    elif lw in units:
        return "UNI"
    elif lw in thickeners:
        return "THI"
    elif lw in enzymes:
        return "ENZ"
    elif lw in acidifiers:
        return "ACI"
    elif lw in humectants:
        return "HUM"
    else:
        return "ING"

def auto_tag_multi(text):
    tokens = []
    tags = []
    word_tag_pairs = []
    ingredients = str(text).split(",")
    for ingredient in ingredients:
        words = tokenize(ingredient)
        for word in words:
            tokens.append(word)
            tag = classify_token(word)
            tags.append(tag)
            word_tag_pairs.append(f"{word}:{tag}")
    return tokens, tags, word_tag_pairs

# --- Apply tagging ---
df['tokens'], df['tags'], df['word_tag_pairs'] = zip(*df['ingredients_text'].map(auto_tag_multi))

# Save the result
df[['product_name', 'ingredients_text', 'tokens', 'tags', 'word_tag_pairs']].to_csv("./genData/tagged_ingredient_output_with_pairs.csv", index=False)
