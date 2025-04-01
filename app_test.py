# import os
# os.environ['JINA_MP_START_METHOD'] = 'spawn'  # Must be first
# os.environ['JINA_LOG_LEVEL'] = 'ERROR'  # Reduce verbosity

# from typing import List, Dict, Optional
# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify, render_template
# from docarray import BaseDoc, DocList
# from docarray.typing import AnyTensor, ImageUrl
# from jina import Executor, Flow, requests

# # Define a simplified Document schema
# class RecipeDoc(BaseDoc):
#     title: str
#     url: str
#     image: Optional[ImageUrl] = None
#     description: str
#     ingredients: List[str] = []
#     instructions: List[str] = []
#     prep_time: Optional[str] = None
#     cook_time: Optional[str] = None
#     total_time: Optional[str] = None
#     servings: Optional[str] = None
#     ratings: Optional[str] = None
#     category: Optional[List[str]] = None
#     cuisine: Optional[List[str]] = None
#     embedding: Optional[AnyTensor] = None

#     @property
#     def text(self):
#         return f"{self.title} {self.description} {' '.join(self.ingredients)}"

# # Initialize model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def load_recipes(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         recipes_data = json.load(f)
    
#     recipes = []
#     for recipe_data in recipes_data:
#         try:
#             # Handle missing or None values
#             instructions = recipe_data.get('instructions') or []
#             ingredients = recipe_data.get('ingredients') or []
            
#             recipes.append(RecipeDoc(
#                 title=recipe_data.get('title', ''),
#                 url=recipe_data.get('url', ''),
#                 image=recipe_data.get('image'),
#                 description=recipe_data.get('description', ''),
#                 ingredients=ingredients,
#                 instructions=instructions,
#                 prep_time=recipe_data.get('prep_time'),
#                 cook_time=recipe_data.get('cook_time'),
#                 total_time=recipe_data.get('total_time'),
#                 servings=recipe_data.get('servings'),
#                 ratings=recipe_data.get('ratings'),
#                 category=recipe_data.get('category'),
#                 cuisine=recipe_data.get('cuisine')
#             ))
#         except Exception as e:
#             print(f"Skipping invalid recipe: {e}")
    
#     return DocList[RecipeDoc](recipes)

# class RecipeEncoder(Executor):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model

#     @requests
#     def encode(self, docs: DocList[RecipeDoc], **kwargs):
#         texts = [doc.text for doc in docs]
#         embeddings = self.model.encode(texts, convert_to_numpy=True)
#         for doc, emb in zip(docs, embeddings):
#             doc.embedding = emb.astype(np.float32)

# class RecipeIndexer(Executor):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.index = None
#         self.docs = DocList[RecipeDoc]()

#     @requests(on='/index')
#     def index(self, docs: DocList[RecipeDoc], **kwargs):
#         if not docs:
#             return
            
#         embeddings = np.stack([doc.embedding for doc in docs if doc.embedding is not None])
#         if embeddings.size == 0:
#             return
            
#         if self.index is None:
#             self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
#         self.index.add(embeddings)
#         self.docs.extend(docs)

#     @requests(on='/search')
#     def search(self, docs: DocList[RecipeDoc], **kwargs):
#         if self.index is None or len(self.docs) == 0:
#             return
            
#         query_embeddings = np.stack([doc.embedding for doc in docs if doc.embedding is not None])
#         if query_embeddings.size == 0:
#             return
            
#         distances, indices = self.index.search(query_embeddings, k=3)
#         for doc, idxs in zip(docs, indices):
#             doc.matches = [self.docs[i] for i in idxs if i < len(self.docs)]

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/search', methods=['POST'])
# def search():
#     try:
#         query = request.json.get('query', '')
#         if not query:
#             return jsonify({'error': 'Empty query'}), 400
            
#         # Create query doc
#         query_doc = RecipeDoc(
#             title='',
#             url='',
#             description='',
#             ingredients=[],
#             instructions=[]
#         )
#         query_doc.text = query  # Set the search text
        
#         # Initialize flow
#         flow = (
#             Flow(protocol='http', port=12345, cors=True)
#             .add(uses=RecipeEncoder, name='encoder')
#             .add(uses=RecipeIndexer, name='indexer')
#         )
        
#         with flow:
#             response = flow.post(
#                 on='/search',
#                 inputs=DocList[RecipeDoc]([query_doc]),
#                 return_results=True
#             )
            
#             if not response or not response[0].docs:
#                 return jsonify([])
                
#             results = []
#             for match in response[0].docs[0].matches:
#                 results.append({
#                     'title': match.title,
#                     'url': match.url,
#                     'image': match.image,
#                     'description': match.description,
#                     'ingredients': match.ingredients,
#                     'instructions': match.instructions,
#                     'prep_time': match.prep_time,
#                     'cook_time': match.cook_time,
#                     'total_time': match.total_time,
#                     'servings': match.servings,
#                     'ratings': match.ratings
#                 })
            
#             return jsonify(results)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def index_recipes():
#     print("Loading recipes...")
#     recipes = load_recipes('recipes.json')
#     print(f"Loaded {len(recipes)} recipes")
    
#     flow = (
#         Flow()
#         .add(uses=RecipeEncoder, name='encoder')
#         .add(uses=RecipeIndexer, name='indexer')
#     )
    
#     with flow:
#         print("Indexing recipes...")
#         batch_size = 50
#         for i in range(0, len(recipes), batch_size):
#             batch = recipes[i:i + batch_size]
#             flow.post(on='/index', inputs=batch)
#         print("Indexing complete")

# if __name__ == '__main__':
#     # First index the recipes
#     index_recipes()
    
#     # Then start the Flask app
#     print("Starting Flask server...")
#     app.run(host='0.0.0.0', port=5007)



# import os
# import time
# import json
# import numpy as np
# import faiss
# from datetime import datetime
# from typing import List, Dict, Optional
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify, render_template
# from docarray import BaseDoc, DocList
# from docarray.typing import AnyTensor, ImageUrl
# from jina import Executor, Flow, requests

# # Configuration
# os.environ['JINA_MP_START_METHOD'] = 'spawn'
# os.environ['JINA_LOG_LEVEL'] = 'ERROR'
# INDEX_FILE = 'recipe_index.faiss'
# DOCS_FILE = 'recipe_docs.json'
# BATCH_SIZE = 50

# # Document schema
# class RecipeDoc(BaseDoc):
#     title: str
#     url: str
#     image: Optional[ImageUrl] = None
#     description: str
#     ingredients: List[str] = []
#     instructions: List[str] = []
#     prep_time: Optional[str] = None
#     cook_time: Optional[str] = None
#     total_time: Optional[str] = None
#     servings: Optional[str] = None
#     ratings: Optional[str] = None
#     category: Optional[List[str]] = None
#     cuisine: Optional[List[str]] = None
#     embedding: Optional[AnyTensor] = None
#     last_updated: datetime = datetime.now()

#     @property
#     def text(self):
#         return f"{self.title} {self.description} {' '.join(self.ingredients)}"

# # Initialize model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# class IndexStats:
#     def __init__(self):
#         self.total_docs = 0
#         self.index_size_bytes = 0
#         self.index_time_seconds = 0
#         self.last_index_time = None
#         self.batch_times = []
    
#     def update(self, num_docs, index_size, batch_time):
#         self.total_docs += num_docs
#         self.index_size_bytes = index_size
#         self.index_time_seconds += batch_time
#         self.batch_times.append(batch_time)
#         self.last_index_time = datetime.now()
    
#     def get_stats(self):
#         avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
#         bytes_per_doc = self.index_size_bytes / self.total_docs if self.total_docs else 0
#         docs_per_second = self.total_docs / self.index_time_seconds if self.index_time_seconds else 0
        
#         return {
#             'total_docs': self.total_docs,
#             'index_size_mb': round(self.index_size_bytes / (1024 * 1024), 2),
#             'total_index_time_seconds': round(self.index_time_seconds, 2),
#             'avg_batch_time_seconds': round(avg_batch_time, 2),
#             'bytes_per_doc': round(bytes_per_doc, 2),
#             'docs_per_second': round(docs_per_second, 2),
#             'last_index_time': self.last_index_time.isoformat() if self.last_index_time else None
#         }

# def load_recipes(file_path):
#     """Load recipes with validation and error handling"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         recipes_data = json.load(f)
    
#     recipes = []
#     for recipe_data in recipes_data:
#         try:
#             recipes.append(RecipeDoc(
#                 title=recipe_data.get('title', ''),
#                 url=recipe_data.get('url', ''),
#                 image=recipe_data.get('image'),
#                 description=recipe_data.get('description', ''),
#                 ingredients=recipe_data.get('ingredients', []),
#                 instructions=recipe_data.get('instructions', []),
#                 prep_time=recipe_data.get('prep_time'),
#                 cook_time=recipe_data.get('cook_time'),
#                 total_time=recipe_data.get('total_time'),
#                 servings=recipe_data.get('servings'),
#                 ratings=recipe_data.get('ratings'),
#                 category=recipe_data.get('category'),
#                 cuisine=recipe_data.get('cuisine')
#             ))
#         except Exception as e:
#             print(f"Skipping invalid recipe: {e}")
    
#     return DocList[RecipeDoc](recipes)

# class RecipeEncoder(Executor):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model

#     @requests
#     def encode(self, docs: DocList[RecipeDoc], **kwargs):
#         texts = [doc.text for doc in docs]
#         embeddings = self.model.encode(texts, convert_to_numpy=True)
#         for doc, emb in zip(docs, embeddings):
#             doc.embedding = emb.astype(np.float32)

# class RecipeIndexer(Executor):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.index = None
#         self.docs = DocList[RecipeDoc]()
#         self.stats = IndexStats()
#         self.load_existing_index()

#     def load_existing_index(self):
#         """Load existing index if available"""
#         if os.path.exists(INDEX_FILE):
#             self.index = faiss.read_index(INDEX_FILE)
#             print(f"Loaded existing index with {self.index.ntotal} vectors")
        
#         if os.path.exists(DOCS_FILE):
#             self.docs = DocList[RecipeDoc].from_json(DOCS_FILE)
#             print(f"Loaded {len(self.docs)} existing documents")

#     def save_index(self):
#         """Save current index to disk"""
#         if self.index:
#             faiss.write_index(self.index, INDEX_FILE)
#             self.docs.to_json(DOCS_FILE)
#             print(f"Saved index with {len(self.docs)} documents")

#     @requests(on='/index')
#     def index(self, docs: DocList[RecipeDoc], **kwargs):
#         start_time = time.time()
        
#         if not docs:
#             return
            
#         # Filter out documents that already exist (update instead of duplicate)
#         new_docs = [doc for doc in docs if doc.url not in {d.url for d in self.docs}]
        
#         if not new_docs:
#             print("No new documents to index")
#             return
            
#         # Encode documents (if not already encoded)
#         if any(doc.embedding is None for doc in new_docs):
#             self.encode(new_docs)
            
#         embeddings = np.stack([doc.embedding for doc in new_docs])
        
#         if self.index is None:
#             self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
#         self.index.add(embeddings)
#         self.docs.extend(new_docs)
        
#         # Update stats
#         batch_time = time.time() - start_time
#         self.stats.update(
#             num_docs=len(new_docs),
#             index_size=self.index.ntotal * embeddings.shape[1] * 4,  # 4 bytes per float32
#             batch_time=batch_time
#         )
        
#         # Save index periodically
#         if len(self.docs) % (BATCH_SIZE * 10) == 0:
#             self.save_index()

#     @requests(on='/search')
#     def search(self, docs: DocList[RecipeDoc], **kwargs):
#         if self.index is None or len(self.docs) == 0:
#             return
            
#         query_embeddings = np.stack([doc.embedding for doc in docs if doc.embedding is not None])
#         if query_embeddings.size == 0:
#             return
            
#         distances, indices = self.index.search(query_embeddings, k=3)
#         for doc, idxs in zip(docs, indices):
#             doc.matches = [self.docs[i] for i in idxs if i < len(self.docs)]

#     @requests(on='/stats')
#     def get_stats(self, **kwargs):
#         return self.stats.get_stats()

#     @requests(on='/remove')
#     def remove_docs(self, docs: DocList[RecipeDoc], **kwargs):
#         """Remove documents by URL"""
#         urls_to_remove = {doc.url for doc in docs}
#         before_count = len(self.docs)
        
#         # Remove from docs list
#         self.docs = DocList[RecipeDoc]([d for d in self.docs if d.url not in urls_to_remove])
        
#         # Rebuild index completely (FAISS doesn't support efficient removal)
#         if len(self.docs) > 0:
#             embeddings = np.stack([doc.embedding for doc in self.docs])
#             self.index = faiss.IndexFlatL2(embeddings.shape[1])
#             self.index.add(embeddings)
        
#         print(f"Removed {before_count - len(self.docs)} documents")
#         self.save_index()

# app = Flask(__name__)
# app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
# app.config['TEMPLATES_AUTO_RELOAD'] = True

# # Simple cache management
# query_cache = {}
# CACHE_SIZE = 1000

# def clear_cache():
#     global query_cache
#     query_cache = {}
#     print("Cache cleared")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/search', methods=['POST'])
# def search():
#     try:
#         query = request.json.get('query', '').strip()
#         if not query:
#             return jsonify({'error': 'Empty query'}), 400
        
#         # Check cache first
#         cache_key = hash(query)
#         if cache_key in query_cache:
#             print("Returning cached results")
#             return jsonify(query_cache[cache_key])
            
#         # Create query doc
#         query_doc = RecipeDoc(
#             title='',
#             url='',
#             description='',
#             ingredients=[],
#             instructions=[]
#         )
#         query_doc.text = query
        
#         # Initialize flow
#         flow = (
#             Flow(protocol='http', port=12345, cors=True)
#             .add(uses=RecipeEncoder, name='encoder')
#             .add(uses=RecipeIndexer, name='indexer')
#         )
        
#         with flow:
#             # Encode query
#             flow.post(on='/encode', inputs=DocList[RecipeDoc]([query_doc]))
            
#             # Search
#             response = flow.post(
#                 on='/search',
#                 inputs=DocList[RecipeDoc]([query_doc]),
#                 return_results=True
#             )
            
#             if not response or not response[0].docs:
#                 return jsonify([])
                
#             # Prepare results
#             results = []
#             for match in response[0].docs[0].matches:
#                 results.append({
#                     'title': match.title,
#                     'url': match.url,
#                     'image': match.image,
#                     'description': match.description,
#                     'ingredients': match.ingredients,
#                     'instructions': match.instructions,
#                     'prep_time': match.prep_time,
#                     'cook_time': match.cook_time,
#                     'total_time': match.total_time,
#                     'servings': match.servings,
#                     'ratings': match.ratings
#                 })
            
#             # Update cache
#             if len(query_cache) >= CACHE_SIZE:
#                 clear_cache()
#             query_cache[cache_key] = results
            
#             return jsonify(results)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/stats', methods=['GET'])
# def get_stats():
#     flow = Flow().add(uses=RecipeIndexer)
#     with flow:
#         response = flow.post(on='/stats', return_results=True)
#         return jsonify(response[0].parameters['__results__'])

# def index_recipes():
#     print("Loading recipes...")
#     recipes = load_recipes('recipes.json')
#     print(f"Loaded {len(recipes)} valid recipes")
    
#     # Initialize flow
#     flow = (
#         Flow()
#         .add(uses=RecipeEncoder, name='encoder')
#         .add(uses=RecipeIndexer, name='indexer')
#     )
    
#     with flow:
#         print("\nIndexing recipes...")
#         print(f"Total batches to process: {len(recipes) // BATCH_SIZE + 1}")
#         print("=" * 50)
        
#         for i in range(0, len(recipes), BATCH_SIZE):
#             batch = recipes[i:i + BATCH_SIZE]
#             start_time = time.time()
            
#             # Print progress before processing
#             print(f"Processing batch {i//BATCH_SIZE + 1} (docs {i} to {min(i+BATCH_SIZE, len(recipes))-1})", end='', flush=True)
            
#             # Process batch
#             flow.post(on='/index', inputs=batch)
            
#             # Print timing after processing
#             batch_time = time.time() - start_time
#             print(f" - Completed in {batch_time:.2f}s")
            
#             # Print memory usage periodically
#             if (i // BATCH_SIZE) % 10 == 0:
#                 import psutil
#                 process = psutil.Process(os.getpid())
#                 print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        
#         # Final stats
#         response = flow.post(on='/stats', return_results=True)
#         stats = response[0].parameters['__results__']
        
#         print("\n" + "=" * 50)
#         print("Indexing complete! Final statistics:")
#         print(json.dumps(stats, indent=2))
#         print("=" * 50)
        
#         # Save index
#         flow.post(on='/save_index')

# if __name__ == '__main__':
#     try:
#         # First index the recipes
#         index_recipes()
        
#         # Then start the Flask app
#         print("\nStarting Flask server...")
#         app.run(host='0.0.0.0', port=5009, threaded=True)
#     except KeyboardInterrupt:
#         print("\nProcess interrupted by user")
#     except Exception as e:
#         print(f"\nError occurred: {str(e)}")


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
