import pandas as pd
import json 
import isodate 

""" Cut down recipe df (if desired) and get number of and avg rating(s) """
def get_slim_data(nrecipes=None): 

    recipes = pd.read_parquet("data/recipes.parquet", engine="pyarrow")

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
def get_json_row(row): 
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
    recipes = get_slim_data(10000)

    ex = get_json_row(recipes.iloc[0])
    print("Example row: ", ex)

    recipe_docs = [get_json_row(row) for _, row in recipes.iterrows()]

    with open("data/formatted_recipes.json", "w") as f:
        json.dump(recipe_docs, f)


