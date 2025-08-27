# Recipe Search Engine (Scikit-learn + Flask)

## Project Overview
This project demonstrates how to build an intelligent Recipe Search Engine using Scikit-learn for machine learning and Flask for web deployment.

* **Dataset**: Tagged Food Recipes Dataset (CSV format)
* **Machine Learning Pipeline**:
   * Text preprocessing and cleaning
   * Feature extraction using TF-IDF Vectorization
   * Similarity search with Nearest Neighbors algorithm
   * Model persistence with pickle
* **Flask** provides a web interface for recipe search and discovery
## Source
- The source of the dataset is https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images
- It includes the Dataset and the Images 
## Features
- **Smart Search**: Find recipes by ingredients, tags, or keywords
- **Similarity Matching**: Uses cosine similarity to find the most relevant recipes
- **Image Display**: Shows recipe images alongside results
- **Detailed View**: Complete recipe information including ingredients and instructions
- **Match Confidence**: Shows how closely recipes match your search query

## Project Structure
```
recipe-search-engine/
├── Tagged_Food_Recipes.csv    # Raw dataset
├── train_and_save.py          # Model training script
├── flask_app.py               # Flask web application
├── templates/                 # HTML templates
├── Food Images/               # Recipe images directory
├── vectorizer.pkl             # Trained TF-IDF vectorizer (generated)
├── nn_model.pkl              # Trained Nearest Neighbors model (generated)
├── recipes_meta.csv          # Processed dataset (generated)
└── requirements.txt          # Python dependencies
```

## Machine Learning Approach
1. **Data Processing**: Combines recipe title, ingredients, instructions, and tags into a unified text field
2. **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numerical features
3. **Model Training**: Implements K-Nearest Neighbors with cosine similarity for finding similar recipes
4. **Model Persistence**: Saves trained vectorizer and model using pickle for fast loading

## Steps to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
- Place your `Tagged_Food_Recipes.csv` file in the project directory
- Ensure the `Food Images/` folder contains recipe images

### 3. Train the Model
```bash
python train_and_save.py
```
This will generate:
- `vectorizer.pkl` - TF-IDF vectorizer
- `nn_model.pkl` - Nearest Neighbors model  
- `recipes_meta.csv` - Processed dataset

### 4. Run the Application
```bash
python flask_app.py
```

### 5. Access the Web App
Open your browser and navigate to `http://localhost:5000`

## Usage
1. Enter an ingredient, cuisine type, or keyword in the search box
2. View search results with images and match confidence scores
3. Click on recipes for detailed information
4. Try searches like:
   - "chicken pasta"
   - "vegan dessert"
   - "spicy curry"
   - "chocolate cake"

## Technical Details
- **Text Processing**: Handles encoding issues with `latin1` encoding
- **Search Algorithm**: Cosine similarity for semantic recipe matching
- **Performance**: TF-IDF limited to 5000 features for optimal speed
- **Image Handling**: Supports multiple image formats (.jpg, .png, .jpeg)

## Requirements
- Python 3.7+
- pandas
- scikit-learn
- flask
- pickle (built-in)

## Dataset Format
The CSV should contain these columns:
- `Title`: Recipe name
- `Ingredients`: List of ingredients
- `Instructions`: Cooking steps
- `Tags`: Recipe categories/tags
- `Image_Name`: Associated image filename

## Future Enhancements
- Add recipe ratings and reviews
- Implement dietary filters (vegetarian, gluten-free, etc.)
- Include nutritional information
- Add recipe recommendation system
- Mobile-responsive design improvements
