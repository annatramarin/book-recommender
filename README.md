# book-recommender
A semantic book recommendation system built with sentence-transformers and cosine similarity. Input a natural language description of what you're in the mood for and get the closest matching books from a Goodreads dataset.

## How It Works
 
1. Book descriptions from a Goodreads dataset are converted into vector embeddings using a pretrained sentence-transformer model
2. When a user inputs a natural language query (e.g. *"a dark mystery set in Victorian London"*), it is embedded into the same vector space
3. Cosine similarity is computed between the query vector and all book vectors
4. The closest matching books are returned ranked by similarity
   
## Example
 
```
Input:  "a dark mystery set in Victorian London"
 
Output:
I recommend the following books:
 
- 'The Crimes of Jack The Ripper', which is a pictorial excursion into the dark underbelly of Victorian London...
- 'Oh! Where are Bloody Mary's Earrings?', which is a mystery set in the court of Queen Victoria...
- 'The Shadow of William Quest', which is a Victorian thriller set in the gaslit alleys of London...
```
 
## Stack
 
- `sentence-transformers`: generates semantic embeddings using `all-MiniLM-L6-v2`
- `scikit-learn`: cosine similarity computation
- `pandas`: data loading and manipulation
- `numpy`: saving and loading embeddings
- `HuggingFace datasets`: loading the [Goodreads book descriptions dataset](https://huggingface.co/datasets/booksouls/goodreads-book-descriptions)
  
## Setup
 
**1. Install dependencies**
```bash
pip install sentence-transformers scikit-learn pandas numpy datasets
```
 
**2. Load the dataset and generate embeddings**
```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
 
ds = load_dataset("booksouls/goodreads-book-descriptions")
df = ds["train"].to_pandas()
 
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)
np.save("embeddings.npy", embeddings)
```
 
**3. Run a recommendation**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
embeddings = np.load("embeddings.npy")
 
def recommend(query, top_n=5):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices][["title", "description"]]
    return results
 
results = recommend("a dark mystery set in Victorian London")
 
print("I recommend the following books:\n")
for _, row in results.iterrows():
    print(f"- '{row['title']}', which is a {row['description']}\n")
```
 
## Dataset
 
[booksouls/goodreads-book-descriptions](https://huggingface.co/datasets/booksouls/goodreads-book-descriptions) via HuggingFace — contains book titles and descriptions sourced from Goodreads.
 
## Notes
 
- Embeddings are generated once and saved to `embeddings.npy` to avoid recomputing on every run
- For faster embedding generation, enable GPU in your environment (e.g. Google Colab: Runtime → Change runtime type → T4 GPU)
- The model `all-MiniLM-L6-v2` runs fine on CPU for small subsets (~500 books) but a GPU is recommended for the full dataset
 
