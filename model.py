import json
import numpy as np 
from sentence_transformers import SentenceTransformer
import faiss 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
from huggingface_hub import login
from transformers import pipeline
import os
login(token=os.environ.get('hf_token'))

""" Given a recipe, get an embedding that describes it for search """
def get_recipe_emb(recipe, model, mode='foodnetwork'):
    if mode == 'foodcom': 
        title = recipe["title"]
        category = recipe.get("category", "")
        tags = ", ".join(recipe["tags"]) if recipe["tags"][0] else "" # some recipes don't have tags
        desc = recipe.get("description", "")
        ingredients = ", ".join(recipe["ingredients"])

        search_text = f"{title} - {category} - {tags} - {desc} - {ingredients}"

    elif mode == 'foodnetwork':
        title = recipe["title"]
        desc = recipe["description"]
        level = recipe["level"]
        ingredients = recipe["ingredients"]

        search_text = f"{title} - {desc} - {level} - {ingredients}"

    else: 
        print('Unsupported mode')
        return 
    
    return model.encode(search_text) 

""" Given a recipe, get the context that will be returned if chosen by RAG """
def get_recipe_context(recipe, mode='foodnetwork'): 

    if mode == 'foodcom': 
        title = recipe["title"]
        tags = ", ".join(recipe["tags"]) if recipe["tags"][0] else ""
        desc = recipe.get("description", "")
        ingredients = ", ".join(recipe["ingredients"])
        steps = ", ".join(recipe["steps"])
        context_text = f"Title: {title}\nDescription: {desc}\
        \nTags: {tags}\nIngredients: {ingredients}\nSteps: {steps}"

    elif mode == 'foodnetwork': 
        title = recipe["title"]
        desc = recipe["description"]
        level = recipe["level"]
        ingredients = recipe["ingredients"]
        directions = recipe["directions"]
        url = recipe["url"]
        context_text = f"Title: {title}\nDescription: {desc}\nLevel: {level}\
        \nIngredients: {ingredients}\nDirections: {directions}\nURL: {url}"

    else: 
        print("Unsupported mode")
        return 

    return context_text

""" Given a model and dataset of recipes, save the recipes' IDs, 
    context texts, and indexed search vectors (embeddings) """
def save_embeddings(model, recipes_path, context_path, ids_path, faiss_idx_path): 
    
    with open(recipes_path) as f:
        recipes = json.load(f)

    print(recipes[0])

    recipe_ids = []
    search_vecs = []
    context_texts = []
    for idx, recipe in enumerate(recipes): 
        recipe_ids.append(idx)
        search_vecs.append(get_recipe_emb(recipe, model))
        context_texts.append(get_recipe_context(recipe))

    emb_matrix = np.array(search_vecs).astype("float32")
    index = faiss.IndexFlatL2(emb_matrix.shape[1])  # 384 dimensions
    index.add(emb_matrix)
    faiss.write_index(index, faiss_idx_path)

    with open(context_path, "w") as f:
        json.dump(context_texts, f)

    with open(ids_path, "w") as f:
        json.dump(recipe_ids, f)

def get_top_recipes(model, user_query, faiss_idx_path, context_path): 
    encoded_query = model.encode([user_query]) 

    index = faiss.read_index(faiss_idx_path)
    D, I = index.search(np.array(encoded_query, dtype="float32"), k=3)

    with open(context_path) as f: 
        context_texts = json.load(f) 

    top_contexts = []
    for i in I[0]: 
        context = context_texts[i]

        top_contexts.append(context)

    return top_contexts

def create_prompt(user_query, top_contexts): 

    text = f"""You are a friendly cooking assistant that helps people find great recipes based on their needs.\n
    A user has requested:\n
    "{user_query}"\n
    Here is one recipe you can consider: 
    ===RECIPE START===\n
    {top_contexts[0]}\n
    ===RECIPE END===\n
    Now write a helpful recommendation explaining why this recipe matches the user's request. Do not repeat the recipe formatting.\n
    Response:
    """
    return text

if __name__ == '__main__': 

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    recipes_path = 'data/foodnetwork/formatted_recipes.json'
    context_path = 'data/foodnetwork/context_texts.json'
    ids_path = 'data/foodnetwork/recipe_ids.json'
    faiss_idx_path = 'data/foodnetwork/recipe_index.faiss'
    save_embeddings(embedding_model, recipes_path, context_path, ids_path, faiss_idx_path)

    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map={"": "cpu"},
    #     torch_dtype=torch.float32  # Don't use float16 on CPU
    # )

    # prompt = """<s>[INST] I'd like something with mushrooms - what do you recommend? [/INST]"""

    # inputs = tokenizer(prompt, return_tensors="pt")

    # output = model.generate(
    #     **inputs,
    #     max_new_tokens=150,
    #     do_sample=True,
    #     top_p=0.9,
    #     temperature=0.7,
    #     repetition_penalty=1.1
    # )

    # print(tokenizer.decode(output[0], skip_special_tokens=True))
