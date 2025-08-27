README - Recipe Recommender (TF-IDF + NearestNeighbors)

Files included:
- train_and_save.py     : trains TF-IDF + NN and saves artifacts to 'artifacts/'.
- flask.py                      : it allows the presenting of the data and hosting of the website
- requirements.txt      : Python packages to install.

How to use (on your PC):
1. Place these files into your project folder where 'Tagged_Food_Recipes.csv' and the folder 'Food Images' already live.
2. Open terminal and `cd` into the project folder.
3. (Optional but recommended) create venv:
   python -m venv venv
   Windows: .\venv\Scripts\Activate.ps1  (or activate.bat)
   macOS/Linux: source venv/bin/activate
4. Install packages:
   pip install -r requirements.txt
5. Train (creates 'artifacts/'):
   python train_and_save.py
6. Run flask:
       run flask_app.py

Notes:
- If you run into memory issues while TF-IDF fits, edit MAX_FEATURES in train_and_save.py to a smaller value (e.g., 5000).
- If images do not show, verify that the Image_Name column matches filenames in 'Food Images' exactly (including extensions).
