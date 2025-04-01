from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

app = Flask(__name__)

@dataclass
class Recipe:
    title: str
    url: str
    image: str
    description: str
    prep_time: str
    cook_time: str
    total_time: str
    servings: str
    ingredients: List[str]
    instructions: List[str]
    rating: Optional[str] = None
    category: Optional[str] = None
    cousine: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class RecipeSearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
         # Start timing
        start_time = time.time()
        self.recipes = self.load_recipes('recipes_old.json')
        # End timing and calculate duration
        end_time = time.time()
        self.indexing_time = end_time - start_time
        print(f"Indexing completed in {self.indexing_time:.2f} seconds")
        
    def load_recipes(self, file_path: str) -> List[Recipe]:
        with open(file_path) as f:
            data = json.load(f)
        
        recipes = []
        for item in data:
            search_text = (
                f"{item.get('title', '')} "
                f"{item.get('description', '')} "
                f"Ingredients: {' '.join(item.get('ingredients', []))} "
                f"Instructions: {' '.join(item.get('instructions', []) or [])} "
                f"Category: {item.get('category', '')} "
                f"Cousine: {item.get('cousine', '')}"
            )
            
            recipe = Recipe(
                title=item.get('title', ''),
                url=item.get('url', ''),
                image=item.get('image', ''),
                description=item.get('description', ''),
                prep_time=item.get('prep_time', ''),
                cook_time=item.get('cook_time', ''),
                total_time=item.get('total_time', ''),
                servings=item.get('servings', ''),
                ingredients=item.get('ingredients', []),
                instructions=item.get('instructions', []),
                rating=item.get('ratings', ''),
                category=item.get('category', ''),
                cousine=item.get('cousine', ''),
                embedding=self.model.encode(search_text)
            )
            recipes.append(recipe)
        return recipes
    
    def analyze_query(self, query: str) -> dict:
        """More sophisticated query analysis with fuzzy matching"""
        query = query.lower()
        analysis = {
            'meal_type': [],
            'modifiers': {
                'time': [],
                'difficulty': [],
                'diet': [],
                'equipment': []
            },
            'ingredients': [],
            'exclude_ingredients': []
        }

        # Expanded meal types with synonyms
        meal_types = {
            'breakfast': ['breakfast', 'morning', 'brunch', 'pancake', 'waffle', 'omelet'],
            'lunch': ['lunch', 'midday', 'sandwich', 'salad', 'soup'],
            'dinner': ['dinner', 'supper', 'evening', 'main course', 'entree'],
            'dessert': ['dessert', 'sweet', 'cake', 'pie', 'cookie', 'pastry']
        }

        # Expanded modifiers
        modifiers = {
            'time': ['quick', 'fast', 'easy', 'simple', 'minute', 'hour', 
                    'slow', 'long', 'overnight'],
            'difficulty': ['beginner', 'easy', 'simple', 'advanced', 'complex'],
            'diet': ['vegetarian', 'vegan', 'keto', 'low carb', 'gluten free',
                    'dairy free', 'healthy', 'light'],
            'equipment': ['air fryer', 'instant pot', 'slow cooker', 'grill']
        }

        # Detect ingredients to include/exclude
        if 'without' in query or 'no ' in query:
            parts = query.split('without') if 'without' in query else query.split('no ')
            if len(parts) > 1:
                analysis['exclude_ingredients'] = self._extract_ingredients(parts[1])

        # Always look for ingredients to include
        analysis['ingredients'] = self._extract_ingredients(query)

        # Detect meal type using fuzzy matching
        for meal, keywords in meal_types.items():
            if any(keyword in query for keyword in keywords):
                analysis['meal_type'] = meal
                break

        # Detect all modifier types
        for mod_type, mod_list in modifiers.items():
            for mod in mod_list:
                if mod in query:
                    analysis['modifiers'][mod_type].append(mod)

        return analysis

    def _extract_ingredients(self, text: str) -> List[str]:
        """Simple ingredient extraction from query"""
        common_ingredients = ['chicken', 'beef', 'pasta', 'rice', 'egg', 
                            'cheese', 'tomato', 'potato', 'onion']
        found = []
        for ing in common_ingredients:
            if ing in text:
                found.append(ing)
        return found
    
    def matches_analysis(self, recipe: Recipe, analysis: dict) -> bool:
        """More flexible matching with partial matches and scoring"""
        score = 1.0  # Base score
        
        # Meal type matching
        if analysis['meal_type']:
            # Could add more sophisticated meal type detection here
            score *= 1.2  # Boost for matching meal type
        
        # Time constraints
        if analysis['modifiers']['time']:
            total_minutes = self._parse_time(recipe.total_time)
            if 'quick' in analysis['modifiers']['time'] and total_minutes > 30:
                return False
            if 'slow' in analysis['modifiers']['time'] and total_minutes < 60:
                return False
        
        # Ingredient requirements
        if analysis['ingredients']:
            recipe_ings = ' '.join(recipe.ingredients).lower()
            for ing in analysis['ingredients']:
                if ing not in recipe_ings:
                    return False
                score *= 1.1  # Small boost for each matching ingredient
        
        # Excluded ingredients
        if analysis['exclude_ingredients']:
            recipe_ings = ' '.join(recipe.ingredients).lower()
            for ing in analysis['exclude_ingredients']:
                if ing in recipe_ings:
                    return False
        
        # Difficulty level
        if analysis['modifiers']['difficulty']:
            # Simple implementation - could be enhanced with actual difficulty data
            prep_time = self._parse_time(recipe.prep_time)
            if 'easy' in analysis['modifiers']['difficulty'] and prep_time > 20:
                return False
        
        return score  # Return score instead of boolean for flexible matching

    def _parse_time(self, time_str: str) -> int:
        """Parse time strings into minutes"""
        if not time_str:
            return 0
        try:
            if 'min' in time_str:
                return int(time_str.split(' ')[0])
            elif 'hour' in time_str:
                parts = time_str.split(' ')
                hours = int(parts[0])
                if len(parts) > 2 and 'min' in parts[2]:
                    mins = int(parts[2])
                    return hours * 60 + mins
                return hours * 60
        except:
            return 0
        return 0
    
    def search(self, query: str, top_k: int = 12) -> List[Dict]:
        query_analysis = self.analyze_query(query)
        query_embedding = self.model.encode(query)
        
        scored_recipes = []
        for recipe in self.recipes:
            match_score = self.matches_analysis(recipe, query_analysis)
            if not match_score:
                continue
                
            # Calculate semantic similarity
            semantic_sim = np.dot(recipe.embedding, query_embedding) / (
                np.linalg.norm(recipe.embedding) * np.linalg.norm(query_embedding))
            
            # Combine scores
            combined_score = semantic_sim * match_score
            
            scored_recipes.append((combined_score, recipe))
        
        # Sort by combined score
        scored_recipes.sort(reverse=True, key=lambda x: x[0])
        
        # Prepare results
        results = []
        for score, recipe in scored_recipes[:top_k]:
            results.append({
                'title': recipe.title,
                'url': recipe.url,
                'image': recipe.image,
                'description': recipe.description,
                'prep_time': recipe.prep_time,
                'cook_time': recipe.cook_time,
                'total_time': recipe.total_time,
                'servings': recipe.servings,
                'ingredients': recipe.ingredients,
                'instructions': recipe.instructions,
                'rating': recipe.rating,
                'score': float(score),
                'match_details': self._get_match_details(recipe, query_analysis)
            })
        
        return results

    def _get_match_details(self, recipe: Recipe, analysis: dict) -> str:
        """Generate human-readable match explanation"""
        details = []
        
        if analysis['meal_type']:
            details.append(f"Meal type: {analysis['meal_type']}")
        
        for mod_type, mods in analysis['modifiers'].items():
            if mods:
                details.append(f"{mod_type}: {', '.join(mods)}")
        
        if analysis['ingredients']:
            details.append(f"Includes: {', '.join(analysis['ingredients'])}")
        
        if analysis['exclude_ingredients']:
            details.append(f"Excludes: {', '.join(analysis['exclude_ingredients'])}")
        
        return ' | '.join(details)

search_engine = RecipeSearchEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    results = search_engine.search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)