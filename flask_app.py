# flask_app.py
import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Paths
META_PATH = "recipes_meta.csv"
VEC_PATH = "vectorizer.pkl"
NN_PATH = "nn_model.pkl"
IMAGE_DIR = os.path.join(os.getcwd(), "Food Images")

print(f"🔍 Current working directory: {os.getcwd()}")
print(f"🔍 Looking for files:")
print(f"   - {META_PATH}: {os.path.exists(META_PATH)}")
print(f"   - {VEC_PATH}: {os.path.exists(VEC_PATH)}")
print(f"   - {NN_PATH}: {os.path.exists(NN_PATH)}")
print(f"   - Image directory: {IMAGE_DIR} exists: {os.path.exists(IMAGE_DIR)}")

# Load artifacts
try:
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded successfully")
    
    with open(NN_PATH, "rb") as f:
        nn_model = pickle.load(f)
    print("✅ NN model loaded successfully")
    
    df = pd.read_csv(META_PATH)
    print(f"✅ Dataset loaded: {len(df)} recipes")
    
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    raise

def get_image_path(image_name):
    """Get the relative path to an image for web serving"""
    if pd.isna(image_name):
        return None
    
    image_name = str(image_name)
    
    for ext in ["", ".jpg", ".png", ".jpeg"]:
        filename = f"{image_name}{ext}"
        full_path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(full_path):
            # Return relative path for web serving
            return f"images/{filename}"
    return None

def search_recipes(query, n_results=10):
    """Search for recipes based on query"""
    if not query.strip():
        return []
    
    try:
        # Vectorize query
        q_vec = vectorizer.transform([query])
        distances, indices = nn_model.kneighbors(q_vec, n_neighbors=n_results)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(df):
                continue
                
            row = df.iloc[idx]
            
            # Get image path
            img_path = get_image_path(row["Image_Name"])
            
            # Create result dictionary
            result = {
                'title': str(row['Title']) if pd.notna(row['Title']) else "Unknown Recipe",
                'tags': str(row['Tags']) if pd.notna(row['Tags']) else "No tags",
                'ingredients': str(row['Ingredients']) if pd.notna(row['Ingredients']) else "No ingredients",
                'instructions': str(row['Instructions']) if pd.notna(row['Instructions']) else "No instructions",
                'image_path': img_path,
                'confidence': round((1-dist)*100, 1)
            }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in search_recipes: {e}")
        return []

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        query = request.json.get('query', '')
        print(f"🔍 Search query: '{query}'")
        
        results = search_recipes(query)
        print(f"✅ Found {len(results)} results")
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the Food Images directory"""
    from flask import send_from_directory
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, host='127.0.0.1', port=5000)