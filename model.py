import json
import numpy as np 
from sentence_transformers import SentenceTransformer
import faiss 

def get_recipe_emb(recipe):
    title = recipe["title"]
    category = recipe.get("category", "")
    tags = ", ".join(recipe["tags"])
    desc = recipe.get("description", "")
    ingredients = ", ".join(recipe["ingredients"])

    search_text = f"{title} - {category} - {tags} - {desc} - {ingredients}"
    return model.encode(search_text) 

def get_recipe_context(recipe): 

    title = recipe["title"]
    tags = ", ".join(recipe["tags"])
    desc = recipe.get("description", "")
    ingredients = ", ".join(recipe["ingredients"])
    steps = ", ".join(recipe["steps"])

    return f"Title: {title}\nDescription: {desc}\
        \nTags: {tags}\nIngredients: {ingredients}\nSteps: {steps}"

if __name__ == '__main__': 

    with open("data/formatted_recipes.json") as f:
        recipes = json.load(f)

    print(recipes[0])
    model = SentenceTransformer("all-MiniLM-L6-v2")

    recipe_ids = []
    search_vecs = []
    context_texts = []
    for idx, recipe in enumerate(recipes): 
        recipe_ids.append(idx)
        search_vecs.append(get_recipe_emb(recipe))
        context_texts.append(get_recipe_context(recipe))

    emb_matrix = np.array(search_vecs).astype("float32")
    index = faiss.IndexFlatL2(emb_matrix.shape[1])  # 384 dimensions
    index.add(emb_matrix)
    faiss.write_index(index, "recipe_index.faiss")

    with open("data/context_texts.json", "w") as f:
        json.dump(context_texts, f)

    with open("data/recipe_ids.json", "w") as f:
        json.dump(recipe_ids, f)