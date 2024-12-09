from flask import Flask, request, render_template_string, send_from_directory
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import numpy as np
import open_clip
from PIL import Image as PILImage
from open_clip import create_model_and_transforms, tokenizer

# Setup the model and data
df = pd.read_pickle('./image_embeddings.pickle')
image_folder_path = './coco_images_resized'
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
def cosine_similarity(q, i):
    q_norm, i_norm = np.linalg.norm(q), np.linalg.norm(i)
    return np.dot(q, i) / (q_norm * i_norm)

def get_image_embedding(image_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
    image = preprocess(PILImage.open(image_path)).unsqueeze(0)  # Converts image to tensor
    query_embedding = F.normalize(model.encode_image(image))  # Calculates query embedding
    return query_embedding

def ndarray_of_embedding(embedding):
    return embedding.detach().numpy()

def get_impath(df, index):
    file_name = df.iloc[index]['file_name']
    impath = os.path.join(image_folder_path, file_name)
    return file_name  # Return just the relative path (without the base folder)

def use_image_query(image_path):
    query_embedding = get_image_embedding(image_path)
    np_query = ndarray_of_embedding(query_embedding)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(np_query, x))
    max_index = df['similarity'].idxmax()
    file_name = get_impath(df, max_index)
    return file_name

def get_text_embedding(string):
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([string])
    query_embedding = F.normalize(model.encode_text(text))
    return query_embedding

def use_text_query(text):
    query_embedding = get_text_embedding(text)
    np_query = ndarray_of_embedding(query_embedding)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(np_query, x))
    max_index = df['similarity'].idxmax()
    file_name = get_impath(df, max_index)
    return file_name

def use_hybrid_query(file_name, text, lam):
    image_embedding = get_image_embedding(file_name)
    text_embedding = get_text_embedding(text)
    query_embedding = F.normalize(lam * text_embedding + (1.0 - lam) * image_embedding)
    np_query = ndarray_of_embedding(query_embedding)
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(np_query, x))
    max_index = df['similarity'].idxmax()
    file_name = get_impath(df, max_index)
    return file_name

# Flask app setup
app = Flask(__name__)

# Route to serve images from the image folder
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(image_folder_path, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form.get('text_query')
        image_path_query = request.form.get('image_path_query')
        lambda_value = request.form.get('lambda')

        lambda_value = float(lambda_value) if lambda_value else 0.5  # Default lambda is 0.5
        
        if text_query and image_path_query:
            file_name = use_hybrid_query(image_path_query, text_query, lambda_value)
        elif text_query:
            file_name = use_text_query(text_query)
        elif image_path_query:
            file_name = use_image_query(image_path_query)
        else:
            file_name = None

        return render_template_string("""
            <h1>Query Result</h1>
            {% if file_name %}
                <p>Resulting Image: </p>
                <img src="{{ url_for('serve_image', filename=file_name) }}" alt="Result Image">
            {% else %}
                <p>No valid input provided.</p>
            {% endif %}
            <br><br>
            <a href="/">Go Back</a>
        """, file_name=file_name)

    return render_template_string("""
        <h1>Text & Image Query Interface</h1>
        <form method="post">
            <label for="text_query">Text Query:</label><br>
            <input type="text" id="text_query" name="text_query"><br><br>
            
            <label for="image_path_query">Image Path Query:</label><br>
            <input type="text" id="image_path_query" name="image_path_query"><br><br>
            
            <label for="lambda">Lambda (0-1):</label><br>
            <input type="number" id="lambda" name="lambda" min="0" max="1" step="0.01"><br><br>
            
            <input type="submit" value="Submit">
        </form>
    """)

if __name__ == '__main__':
    app.run(debug=True)

