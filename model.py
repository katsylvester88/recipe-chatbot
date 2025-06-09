import json
import numpy as np 
from sentence_transformers import SentenceTransformer
import faiss 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
from huggingface_hub import login
from transformers import pipeline

login(token="token")

""" Given a recipe, get an embedding that describes it for search """
def get_recipe_emb(recipe, model):
    title = recipe["title"]
    category = recipe.get("category", "")
    tags = ", ".join(recipe["tags"]) if recipe["tags"][0] else "" # some recipes don't have tags
    desc = recipe.get("description", "")
    ingredients = ", ".join(recipe["ingredients"])

    search_text = f"{title} - {category} - {tags} - {desc} - {ingredients}"
    return model.encode(search_text) 

""" Given a recipe, get the context that will be returned if chosen by RAG """
def get_recipe_context(recipe): 

    title = recipe["title"]
    tags = ", ".join(recipe["tags"]) if recipe["tags"][0] else ""
    desc = recipe.get("description", "")
    ingredients = ", ".join(recipe["ingredients"])
    steps = ", ".join(recipe["steps"])

    return f"Title: {title}\nDescription: {desc}\
        \nTags: {tags}\nIngredients: {ingredients}\nSteps: {steps}"

""" Given a model and dataset of recipes, save the recipes' IDs, 
    context texts, and indexed search vectors (embeddings) """
def save_embeddings(model, recipes_path="data/formatted_recipes.json"): 
    
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
    faiss.write_index(index, "recipe_index.faiss")

    with open("data/context_texts.json", "w") as f:
        json.dump(context_texts, f)

    with open("data/recipe_ids.json", "w") as f:
        json.dump(recipe_ids, f)

def get_top_recipes(model, user_query): 
    encoded_query = model.encode([user_query]) 

    index = faiss.read_index("recipe_index.faiss")
    D, I = index.search(np.array(encoded_query, dtype="float32"), k=3)

    with open("data/context_texts.json") as f: 
        context_texts = json.load(f) 
    with open("data/formatted_recipes.json") as f: 
        recipes = json.load(f) 

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

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"},
        torch_dtype=torch.float32  # Don't use float16 on CPU
    )

    prompt = """<s>[INST] I'd like something with mushrooms - what do you recommend? [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    # user_query = "I want something with mushrooms. What do you recommend?"
    
    # top_recipes = get_top_recipes(emb_model, user_query)
    
    # prompt = create_prompt(user_query, top_recipes)
    # print(prompt)

    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Force CPU and use float32 to avoid mixed precision issues
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map={"": "cpu"},
    #     torch_dtype=torch.float32  # Don't use float16 on CPU
    # )

    # prompt = """<s>[INST] I'd like something with mushrooms - what do you recommend? [/INST]"""

    # inputs = tokenizer(prompt, return_tensors="pt")

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # llm = AutoModelForCausalLM.from_pretrained("gpt2")

    # tokenizer.pad_token = tokenizer.eos_token 
    # llm.config.pad_token_id = tokenizer.eos_token_id

    # inputs = tokenizer(prompt, return_tensors="pt")
    # output = llm.generate(
    #     **inputs, 
    #     max_new_tokens = 150, 
    #     do_sample=True, 
    #     temperature=0.8, 
    #     top_p = 0.9, 
    #     top_k=50,
    #     repetition_penalty=1.2,
    #     pad_token_id = tokenizer.eos_token_id
    # )

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # chatbot_response = generated_text[len(prompt):].strip()

    # print("Chatbot response: \n")
    # print(chatbot_response)