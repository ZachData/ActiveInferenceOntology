import json
import torch
import pandas as pd
from tqdm import tqdm

def load_embeddings(filename):
    """Load embeddings from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert embeddings list back to tensor
    embeddings = torch.tensor(data['embeddings'])
    paper_ids = data['paper_ids']
    
    return embeddings, paper_ids

def calculate_similarities(embeddings):
    """Calculate cosine similarity between all pairs of embeddings."""
    # Normalize embeddings for cosine similarity
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # Calculate similarities using matrix multiplication
    similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    return similarities

def create_similarity_df(similarities, paper_ids):
    """Create a DataFrame with all pairwise similarities."""
    rows = []
    n_papers = len(paper_ids)
    
    # Create all pairwise combinations
    for i in tqdm(range(n_papers), desc="Processing similarities"):
        for j in range(n_papers):
            # Include all pairs
            rows.append({
                'paper1_id': paper_ids[i],
                'paper2_id': paper_ids[j],
                'similarity_score': similarities[i, j].item()
            })
    
    return pd.DataFrame(rows)

def main():
    print("Loading embeddings...")
    embeddings, paper_ids = load_embeddings('paper_embeddings.json')
    
    print("Calculating similarities...")
    similarities = calculate_similarities(embeddings)
    
    print("Creating similarity DataFrame...")
    df = create_similarity_df(similarities, paper_ids)
    
    print("Saving results...")
    # Save as CSV
    df.to_csv('all_similarities.csv', index=False)
    
    print(f"\nSaved {len(df)} similarities to all_similarities.csv")
    print("\nSample of similarities:")
    print(df.head())

if __name__ == "__main__":
    main()