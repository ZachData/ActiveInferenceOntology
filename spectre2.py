import pandas as pd
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def load_papers(filename):
    """Load papers from JSON file into pandas DataFrame."""
    # Read JSON lines into list
    with open(filename, 'r') as f:
        papers = [json.loads(line) for line in f]
    
    # Convert to DataFrame and select required columns
    df = pd.DataFrame(papers)
    return df[['id', 'title', 'abstract']]

def get_embeddings(texts, model, tokenizer, device):
    """Get embeddings for a list of texts using Spectre2 base."""
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the pooled output and move to CPU
        embeddings = outputs.pooler_output.cpu()
        
    return embeddings

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModel.from_pretrained("allenai/specter2_base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    
    # Load papers
    print("Loading papers...")
    df = load_papers('actinf_papers.json')
    print(f"Loaded {len(df)} papers")
    
    # Combine title and abstract for embedding
    texts = []
    for _, row in df.iterrows():
        text = f"[Title]: {row['title']} [Abstract]: {row['abstract']}"
        texts.append(text)
    
    # Get embeddings in batches
    batch_size = 8
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = get_embeddings(batch_texts, model, tokenizer, device)
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save results
    output = {
        'paper_ids': df['id'].tolist(),
        'embeddings': embeddings.tolist()  # Convert tensor to list
    }
    
    with open('paper_embeddings.json', 'w') as f:
        json.dump(output, f)
    
    print("Done! Embeddings saved to paper_embeddings.json")

if __name__ == "__main__":
    main()