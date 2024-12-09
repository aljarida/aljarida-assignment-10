from flask import Flask, request, render_template_string, send_from_directory
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import numpy as np
import open_clip
from PIL import Image as PILImage
import tempfile

# Setup the model and data
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
df = pd.read_pickle('./image_embeddings.pickle')
image_folder_path = './coco_images_resized'

def cosine_similarity(q, i):
    q_norm, i_norm = np.linalg.norm(q), np.linalg.norm(i)
    return np.dot(q, i) / (q_norm * i_norm)

def get_image_embedding(image_path):
    image = preprocess(PILImage.open(image_path)).unsqueeze(0)  # Converts image to tensor
    query_embedding = F.normalize(model.encode_image(image))  # Calculates query embedding
    return query_embedding

def ndarray_of_embedding(embedding):
    return embedding.detach().numpy()

def get_impath(df, index):
    file_name = df.iloc[index]['file_name']
    return file_name  # Return just the relative path (without the base folder)

# Modify to return top 5 results
def get_top_n_similar(df, query_embedding, n=5):
    np_query = ndarray_of_embedding(query_embedding)
    # Ensure the 'similarity' column is numeric
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(np_query, x)).astype(float)
    # Get the indices of the top n most similar items
    top_n_indices = df['similarity'].nlargest(n).index
    # Extract filenames and similarity values
    top_n_files = df.iloc[top_n_indices]['file_name'].tolist()
    top_n_similarities = df.iloc[top_n_indices]['similarity'].tolist()
    return list(zip(top_n_files, top_n_similarities))  # Return tuples of (filename, similarity)

def use_image_query(image_path):
    query_embedding = get_image_embedding(image_path)
    return get_top_n_similar(df, query_embedding)

def get_text_embedding(string):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([string])
    query_embedding = F.normalize(model.encode_text(text))
    return query_embedding

def use_text_query(text):
    query_embedding = get_text_embedding(text)
    return get_top_n_similar(df, query_embedding)

def use_hybrid_query(file_name, text, lam):
    image_embedding = get_image_embedding(file_name)
    text_embedding = get_text_embedding(text)
    query_embedding = F.normalize(lam * text_embedding + (1.0 - lam) * image_embedding)
    return get_top_n_similar(df, query_embedding)

# Flask app setup
app = Flask(__name__)

# Configure file upload
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to serve images from the image folder
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(image_folder_path, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form.get('text_query')
        image_file = request.files.get('image_file')
        lambda_value = request.form.get('lambda')

        lambda_value = float(lambda_value) if lambda_value else 0.5  # Default lambda is 0.5

        if image_file and allowed_file(image_file.filename):
            # Save the uploaded file to a temporary location
            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(temp_image_path)

            # Process the uploaded image
            if text_query:
                file_names = use_hybrid_query(temp_image_path, text_query, lambda_value)
            else:
                file_names = use_image_query(temp_image_path)
        elif text_query:
            file_names = use_text_query(text_query)
        else:
            file_names = None

        return render_template_string("""
            <h1>Query Results</h1>
            {% if results %}
                <p>Top 5 Most Relevant Images:</p>
                <ul>
                    {% for file_name, similarity in results %}
                        <li>
                            <img src="{{ url_for('serve_image', filename=file_name) }}" alt="Result Image" width="100">
                            <p>Cosine Similarity: {{ similarity }}</p>
                        </li>
                    {% endfor %}
                </ul>
            {% else %}
                <p>No valid input provided.</p>
            {% endif %}
            <br><br>
            <a href="/">Go Back</a>
        """, results=file_names)

    return render_template_string("""
        <h1>Text & Image Query Interface</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="text_query">Text Query:</label><br>
            <input type="text" id="text_query" name="text_query"><br><br>
            
            <label for="image_file">Upload Image:</label><br>
            <input type="file" id="image_file" name="image_file"><br><br>
            
            <label for="lambda">Lambda (0-1):</label><br>
            <input type="number" id="lambda" name="lambda" min="0" max="1" step="0.01"><br><br>
            
            <input type="submit" value="Submit">
        </form>
    """)

if __name__ == '__main__':
    app.run(debug=True)

