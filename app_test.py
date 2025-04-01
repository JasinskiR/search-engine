import os
import time
os.environ['JINA_MP_START_METHOD'] = 'spawn'  # Must be first
os.environ['JINA_LOG_LEVEL'] = 'ERROR'  # Reduce verbosity

from typing import List, Dict, Optional
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from docarray import BaseDoc, DocList
from docarray.typing import AnyTensor, ImageUrl
from jina import Executor, Flow, requests

# Define a simplified Document schema
class RecipeDoc(BaseDoc):
    title: str
    url: str
    image: Optional[ImageUrl] = None
    description: str
    ingredients: List[str] = []
    instructions: List[str] = []
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    total_time: Optional[str] = None
    servings: Optional[str] = None
    ratings: Optional[str] = None
    category: Optional[List[str]] = None
    cuisine: Optional[List[str]] = None
    embedding: Optional[AnyTensor] = None

    @property
    def text(self):
        return f"{self.title} {self.description} {' '.join(self.ingredients)}"

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_recipes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        recipes_data = json.load(f)
    
    recipes = []
    for recipe_data in recipes_data:
        try:
            instructions = recipe_data.get('instructions') or []
            ingredients = recipe_data.get('ingredients') or []
            
            recipes.append(RecipeDoc(
                title=recipe_data.get('title', ''),
                url=recipe_data.get('url', ''),
                image=recipe_data.get('image'),
                description=recipe_data.get('description', ''),
                ingredients=ingredients,
                instructions=instructions,
                prep_time=recipe_data.get('prep_time'),
                cook_time=recipe_data.get('cook_time'),
                total_time=recipe_data.get('total_time'),
                servings=recipe_data.get('servings'),
                ratings=recipe_data.get('ratings'),
                category=recipe_data.get('category'),
                cuisine=recipe_data.get('cuisine')
            ))
        except Exception as e:
            print(f"Skipping invalid recipe: {e}")
    
    return DocList[RecipeDoc](recipes)

class RecipeEncoder(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @requests
    def encode(self, docs: DocList[RecipeDoc], **kwargs):
        texts = [doc.text for doc in docs]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb.astype(np.float32)
        print(f"Generated embeddings for {len(docs)} documents")  # Debug statement

class RecipeIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = None
        self.docs = DocList[RecipeDoc]()

    @requests(on='/index')
    def index(self, docs: DocList[RecipeDoc], **kwargs):
        if not docs:
            print("No documents to index")  # Debug statement
            return
            
        embeddings = np.stack([doc.embedding for doc in docs if doc.embedding is not None])
        if embeddings.size == 0:
            print("No embeddings to add to the index")  # Debug statement
            return
            
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings)
        self.docs.extend(docs)
        print(f"Added {len(docs)} documents to the index")  # Debug statement

    @requests(on='/size')
    def get_size(self, **kwargs):
        """Returns the FAISS index size in KB"""
        if self.index is None:
            print("Index is empty")  # Debug statement
            return {'index_size_kb': 0}

        num_vectors = self.index.ntotal  # Number of stored vectors
        vector_dim = self.index.d        # Vector dimensionality
        size_bytes = num_vectors * vector_dim * 4  # Each float32 is 4 bytes
        size_kb = size_bytes / 1024  # Convert bytes to KB

        print(f"Index contains {num_vectors} vectors, size: {size_kb:.2f} KB")  # Debug statement
        return {'index_size_kb': size_kb}


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.json.get('query', '')
        if not query:
            return jsonify({'error': 'Empty query'}), 400
            
        query_doc = RecipeDoc(title='', url='', description='', ingredients=[], instructions=[])
        query_doc.text = query  
        
        flow = (
            Flow(protocol='http', port=12345, cors=True)
            .add(uses=RecipeEncoder, name='encoder')
            .add(uses=RecipeIndexer, name='indexer')
        )
        
        with flow:
            response = flow.post(on='/search', inputs=DocList[RecipeDoc]([query_doc]), return_results=True)
            
            if not response or not response[0].docs:
                return jsonify([])
                
            results = []
            for match in response[0].docs[0].matches:
                results.append({
                    'title': match.title,
                    'url': match.url,
                    'image': match.image,
                    'description': match.description,
                    'ingredients': match.ingredients,
                    'instructions': match.instructions,
                    'prep_time': match.prep_time,
                    'cook_time': match.cook_time,
                    'total_time': match.total_time,
                    'servings': match.servings,
                    'ratings': match.ratings
                })
            
            return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def index_recipes():
    print("Loading recipes...")
    recipes = load_recipes('recipes.json')
    half_size = len(recipes)
    recipes = recipes[:half_size]
    
    print(f"Loaded {len(recipes)} recipes (half of the dataset)")

    flow = (
        Flow()
        .add(uses=RecipeEncoder, name='encoder')
        .add(uses=RecipeIndexer, name='indexer')
    )

    start_time = time.time()
    with flow:
        print("Indexing recipes...")
        batch_size = 50
        for i in range(0, len(recipes), batch_size):
            batch = recipes[i:i + batch_size]
            flow.post(on='/index', inputs=batch)
        end_time = time.time()

        # ✅ Retrieve index size in KB
        response = flow.post(on='/size', return_results=True)

        index_size_kb = response[0].data.docs[0].tags.get('index_size_kb', 0) if response else 0

        print(f"Indexing complete in {end_time - start_time:.2f} seconds")
        print(f"Total documents indexed: {len(recipes)}")
        print(f"Index size: {index_size_kb:.2f} KB")
        
if __name__ == '__main__':
    index_recipes()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5025)
