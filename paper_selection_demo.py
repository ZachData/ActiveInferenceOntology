import sys
from active_paper_selector import ActivePaperSelector
import random
import pandas as pd
import numpy as np

def find_candidate_paper(selector: ActivePaperSelector, 
                        start_id: str,
                        min_sim: float = 0.1,
                        max_sim: float = 0.9,
                        attempts: int = 3) -> str:
    """
    Find a suitable candidate paper with increasingly relaxed similarity constraints.
    
    Args:
        selector: ActivePaperSelector instance
        start_id: Starting paper ID
        min_sim: Initial minimum similarity threshold
        max_sim: Initial maximum similarity threshold
        attempts: Number of attempts with relaxed constraints
        
    Returns:
        Selected paper ID
    """
    similarities = selector.papers[start_id].similarities
    paper_ids = list(selector.papers.keys())
    
    for i in range(attempts):
        # Relax constraints with each attempt
        current_min = min_sim * (0.5 ** i)
        current_max = max_sim + (1 - max_sim) * (i / attempts)
        
        candidates = [pid for pid, sim in similarities.items() 
                     if current_min <= sim <= current_max and pid != start_id]
        
        if candidates:
            return random.choice(candidates)
    
    # If still no candidates, just choose a random paper
    other_papers = [pid for pid in paper_ids if pid != start_id]
    return random.choice(other_papers)

def demonstrate_paper_path(start_id: str = None, end_id: str = None):
    """
    Demonstrate path finding between two papers using active selection.
    
    Args:
        start_id: Optional starting paper ID (random if None)
        end_id: Optional target paper ID (random if None)
    """
    # Initialize selector
    print("Initializing paper selector...")
    selector = ActivePaperSelector(temperature=0.1)
    selector.load_data(
        'paper_embeddings.json',
        'all_similarities.csv',
        'paper_topics.csv'
    )
    
    paper_ids = list(selector.papers.keys())
    print(f"Loaded {len(paper_ids)} papers")
    
    # Select random papers if not provided
    if start_id is None:
        start_id = random.choice(paper_ids)
        print("Randomly selected start paper")
    
    if end_id is None:
        end_id = find_candidate_paper(selector, start_id)
        print("Selected end paper based on similarity criteria")
    
    start_paper = selector.papers[start_id]
    end_paper = selector.papers[end_id]
    
    print("\nStarting paper:")
    print(f"ID: {start_id}")
    print(f"Title: {start_paper.title}")
    print("Top topics:")
    for topic, weight in sorted(start_paper.topics.items(), 
                              key=lambda x: x[1], reverse=True)[:3]:
        print(f"- {topic}: {weight:.3f}")
        
    print("\nTarget paper:")
    print(f"ID: {end_id}")
    print(f"Title: {end_paper.title}")
    print("Top topics:")
    for topic, weight in sorted(end_paper.topics.items(), 
                              key=lambda x: x[1], reverse=True)[:3]:
        print(f"- {topic}: {weight:.3f}")
    
    print(f"\nDirect similarity: {start_paper.similarities[end_id]:.3f}")
    
    try:
        # Find path using active selection
        print("\nFinding optimal path...")
        path = selector.find_optimal_path(start_id, end_id, min_papers=3)
        
        print("\nSelected path:")
        total_info_gain = 0
        current_papers = []
        
        for i, pid in enumerate(path):
            paper = selector.papers[pid]
            current_papers.append(pid)
            
            # Calculate information gain
            if i > 0:
                info_gain = selector.compute_information_gain(
                    current_papers[:-1], pid)
                total_info_gain += info_gain
            else:
                info_gain = 0
                
            # Print paper details
            print(f"\nStep {i+1}:")
            print(f"ID: {pid}")
            print(f"Title: {paper.title}")
            if i > 0:
                prev_similarity = paper.similarities[path[i-1]]
                print(f"Similarity to previous: {prev_similarity:.3f}")
                print(f"Information gain: {info_gain:.3f}")
            
            # Print top topics that are new or enhanced
            if i > 0:
                prev_coverage = {topic: 0.0 for topic in paper.topics}
                for prev_pid in path[:i]:
                    prev_paper = selector.papers[prev_pid]
                    for topic, weight in prev_paper.topics.items():
                        prev_coverage[topic] = max(prev_coverage[topic], weight)
                
                new_topics = []
                for topic, weight in paper.topics.items():
                    if weight > prev_coverage[topic]:
                        improvement = weight - prev_coverage[topic]
                        new_topics.append((topic, improvement))
                
                if new_topics:
                    print("New/enhanced topics:")
                    for topic, imp in sorted(new_topics, 
                                          key=lambda x: x[1], reverse=True)[:2]:
                        print(f"- {topic}: +{imp:.3f}")
        
        print(f"\nTotal information gain: {total_info_gain:.3f}")
        print(f"Path length: {len(path)} papers")
        
    except nx.NetworkXNoPath:
        print("\nError: No path found between papers. They may be disconnected in the similarity graph.")
    except Exception as e:
        print(f"\nError finding path: {str(e)}")

if __name__ == "__main__":
    # Can be run with specific paper IDs or random selection
    start_id = sys.argv[1] if len(sys.argv) > 1 else None
    end_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    demonstrate_paper_path(start_id, end_id)