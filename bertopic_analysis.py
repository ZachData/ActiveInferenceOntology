import pandas as pd
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN

def load_papers(filepath):
    """Load papers from JSON file and combine title and abstract."""
    papers = []
    with open(filepath, 'r') as file:
        for line in file:
            papers.append(json.loads(line))
    
    df = pd.DataFrame(papers)
    df['text'] = df['title'] + " " + df['abstract']
    return df

def create_topic_model(sentence_model):
    """Create BERTopic model with custom parameters."""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2)
    )
    
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,  # Make sure to calculate probabilities
        verbose=True
    )
    
    return topic_model

def create_topic_distribution_csv(df, topic_model, output_file='paper_topics.csv'):
    """Create a CSV file with papers and their topic distributions."""
    # Get documents ready for transformation
    docs = df['text'].tolist()
    
    # Get fresh topic distributions using transform
    topics, probs = topic_model.transform(docs)
    
    # Get topic info and create labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:  # Skip outlier topic
            topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:3]]
            topic_labels[topic_id] = f"Topic {topic_id}: {', '.join(topic_words)}"
    
    # Create base DataFrame
    result_df = pd.DataFrame({
        'title': df['title'],
        'id': df['id']
    })
    
    # Create a mapping of topic IDs to their position in the probability array
    topic_positions = {topic: idx for idx, topic in enumerate(sorted(topic_labels.keys()))}
    
    # Add probability for each topic
    for topic_id, label in topic_labels.items():
        pos = topic_positions[topic_id]
        result_df[label] = probs[:, pos]
    
    # Round probabilities to 4 decimal places
    prob_cols = [col for col in result_df.columns if col.startswith('Topic')]
    result_df[prob_cols] = result_df[prob_cols].round(4)
    
    # Create summary of top topics for each paper
    result_df['Top Topics'] = result_df[prob_cols].apply(
        lambda x: ' | '.join([f"{col}({x[col]*100:.1f}%)" 
                            for col in x.nlargest(3).index]), axis=1)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    return result_df

def main():
    # Load papers
    print("Loading papers...")
    df = load_papers('actinf_papers.json')
    
    # Initialize the sentence transformer model
    print("Initializing embedding model...")
    sentence_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    
    # Create and train topic model
    print("Creating and training topic model...")
    topic_model = create_topic_model(sentence_model)
    
    # Fit the model
    topics, probs = topic_model.fit_transform(df['text'].tolist())
    
    # Create topic distribution CSV
    print("Creating topic distribution CSV...")
    paper_topics_df = create_topic_distribution_csv(df, topic_model)
    
    # Print sample of results
    print("\nSample of paper-topic distributions:")
    print(paper_topics_df[['title', 'Top Topics']].head())
    print(f"\nFull results saved to paper_topics.csv")

if __name__ == "__main__":
    main()