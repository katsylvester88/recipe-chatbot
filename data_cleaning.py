import pandas as pd
import json 
import isodate 

""" Cut down recipe df (if desired) and get number of and avg rating(s) 
    For food.com recipes (parquet format) 
"""
def get_slim_data_foodcom(recipe_path, nrecipes=None): 

    recipes = pd.read_parquet(recipe_path, engine="pyarrow")

    print(f"Number of recipes: {recipes.shape}")
    print("Recipe dataset columns: ")
    print(recipes.columns)

    # why not just take the most popular recipes
    if nrecipes is not None: 
        recipes_sorted = recipes.sort_values(
            by=['ReviewCount', 'AggregatedRating'], 
            ascending=[False, False])
        recipes = recipes_sorted.head(nrecipes) 

    # minor touchups 
    recipes['ReviewCount'] = recipes['ReviewCount'].fillna(0)

    return recipes

"""Cut down recipe df (if desired)  
   For food network recipes. No rating, so a random sample 
"""
def get_slim_data_foodnetwork(recipe_path, nrecipes=None): 
    return pd.read_csv(recipe_path, sep='\t').\
        sample(n=nrecipes, random_state=42).\
        fillna('')

"""
Formatting goal: 
{
  "title": "Spicy Tofu Stir Fry",
  "category": "Vegetable",
  "description": "...",
  "ingredients": ["tofu", "soy sauce", ...],
  "steps": ["Cut tofu", "Heat oil", ...],
  "tags": ["vegetarian", "quick", "asian"],
  "total_time_in_mins": 30, 
  "total_time_in_hours": 0.5, 
  "num_reviews": 205, 
  "average_rating": 4.25
}
"""
def get_json_row_foodcom(row): 
    return {
        "title": row['Name'], 
        "category": row['RecipeCategory'],
        "description": row['Description'], 
        "ingredients": row['RecipeIngredientParts'].tolist(), 
        "steps": row['RecipeInstructions'].tolist(), 
        "tags": row['Keywords'].tolist(),
        "total_time_in_mins": round(iso_duration_to_mins(row['TotalTime'])), 
        "total_time_in_hours": round(iso_duration_to_hours(row['TotalTime']), 2), 
        "num_reviews": int(row['ReviewCount']), 
        "average_rating": round(row['AggregatedRating'], 2) 
    }

"""
Note I'm now moving to a different row format than above -- partly due to input 
data differences, partly from having learned some lessons (e.g. don't need total 
time in hrs) 
"""
def get_json_row_foodnetwork(row): 
    return {
        "title": row['title'], 
        "description": str(row['description']) , 
        "level": row['level'],
        "total_time_in_mins": row['totalTime'], 
        "ingredients": row['ingredients'].split('__'), 
        "directions": row['directions'].split('__'), 
        "url": row['url']
    }

def iso_duration_to_mins(iso_duration):
    """Converts an ISO 8601 duration string to minutes."""
    duration = isodate.parse_duration(iso_duration)
    total_seconds = duration.total_seconds()
    return total_seconds / 60

def iso_duration_to_hours(iso_duration):
    """Converts an ISO 8601 duration string to minutes."""
    duration = isodate.parse_duration(iso_duration)
    total_seconds = duration.total_seconds()
    return total_seconds / 3600

if __name__ == '__main__': 

    # TO PROCESS FOOD NETWORK RECIPES (preferred)
    read_path = 'data/foodnetwork/raw_recipes.tsv' 
    write_path = 'data/foodnetwork/formatted_recipes.json' 
    recipes = get_slim_data_foodnetwork(read_path, 10000)

    ex = get_json_row_foodnetwork(recipes.iloc[0])
    print("Example row: ", ex)

    recipe_docs = [get_json_row_foodnetwork(row) for _, row in recipes.iterrows()]

    # TO PROCESS FOOD.COM RECIPES
    # read_path = 'data/food.com/recipes.parquet'
    # write_path = 'data/food.com/formatted_recipes.json'
    
    # recipes = get_slim_data_foodcom(read_path, 10000)
    # ex = get_json_row_foodcom(recipes.iloc[0])
    # print("Example row: ", ex)

    # recipe_docs = [get_json_row_foodcom(row) for _, row in recipes.iterrows()]


    with open(write_path, "w") as f:
        json.dump(recipe_docs, f)


