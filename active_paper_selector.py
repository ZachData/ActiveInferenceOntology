import numpy as np
from scipy.special import softmax
import torch
import json
from typing import List, Tuple, Dict
import networkx as nx
from dataclasses import dataclass
import pandas as pd

@dataclass
class PaperNode:
    """Represents a paper with its associated metadata and embeddings"""
    paper_id: str
    title: str
    embedding: np.ndarray
    topics: Dict[str, float]  # Topic distribution
    similarities: Dict[str, float]  # Cached similarities to other papers

class ActivePaperSelector:
    def __init__(self, temperature: float = 1.0):
        """Initialize the active paper selector
        
        Args:
            temperature: Temperature parameter for softmax selection
        """
        self.temperature = temperature
        self.papers: Dict[str, PaperNode] = {}
        self.similarity_graph = nx.Graph()
        
    def load_data(self, 
                  embeddings_file: str,
                  similarities_file: str, 
                  topics_file: str):
        """Load paper data from preprocessed files
        
        Args:
            embeddings_file: JSON file containing paper embeddings
            similarities_file: CSV file containing pairwise similarities
            topics_file: CSV file containing topic distributions
        """
        # Load embeddings
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
            embeddings = torch.tensor(data['embeddings'])
            paper_ids = data['paper_ids']
            
        # Load similarities
        similarities_df = pd.read_csv(similarities_file)
        
        # Load topics
        topics_df = pd.read_csv(topics_file)
        
        # Create paper nodes and build graph
        for i, paper_id in enumerate(paper_ids):
            # Get topic distribution for this paper
            paper_topics = topics_df[topics_df['id'] == paper_id]
            topic_dist = {col: paper_topics[col].iloc[0] 
                         for col in paper_topics.columns 
                         if col.startswith('Topic')}
            
            # Get similarities for this paper
            paper_sims = similarities_df[similarities_df['paper1_id'] == paper_id]
            sim_dict = dict(zip(paper_sims['paper2_id'], 
                              paper_sims['similarity_score']))
            
            # Create paper node
            self.papers[paper_id] = PaperNode(
                paper_id=paper_id,
                title=topics_df[topics_df['id'] == paper_id]['title'].iloc[0],
                embedding=embeddings[i].numpy(),
                topics=topic_dist,
                similarities=sim_dict
            )
            
            # Add to similarity graph
            for other_id, sim in sim_dict.items():
                if sim > 0.5:  # Only add edges for sufficiently similar papers
                    self.similarity_graph.add_edge(paper_id, other_id, weight=sim)
    
    def compute_information_gain(self, 
                               current_papers: List[str], 
                               candidate_paper: str) -> float:
        """Compute expected information gain for a candidate paper
        
        Args:
            current_papers: List of currently selected paper IDs
            candidate_paper: ID of paper being considered
            
        Returns:
            Expected information gain score
        """
        candidate = self.papers[candidate_paper]
        
        # Compute topic coverage improvement
        if not current_papers:
            topic_gain = sum(candidate.topics.values())
        else:
            # Get current topic coverage
            current_coverage = {topic: 0.0 for topic in candidate.topics}
            for paper_id in current_papers:
                paper = self.papers[paper_id]
                for topic, weight in paper.topics.items():
                    current_coverage[topic] = max(current_coverage[topic], weight)
            
            # Compute improvement in coverage
            topic_gain = 0
            for topic, weight in candidate.topics.items():
                if weight > current_coverage[topic]:
                    topic_gain += (weight - current_coverage[topic])
        
        # Compute novelty based on similarity
        if not current_papers:
            novelty = 1.0
        else:
            similarities = [candidate.similarities[pid] for pid in current_papers]
            novelty = 1.0 - max(similarities)
        
        # Combine metrics
        info_gain = 0.7 * topic_gain + 0.3 * novelty
        return info_gain
    
    def select_next_paper(self, 
                         current_papers: List[str],
                         candidate_pool: List[str] = None) -> str:
        """Select the next paper to read using active selection
        
        Args:
            current_papers: List of already selected paper IDs
            candidate_pool: Optional list of paper IDs to choose from
                          (if None, uses all papers)
        
        Returns:
            ID of the selected paper
        """
        if candidate_pool is None:
            candidate_pool = list(self.papers.keys())
        
        # Remove already selected papers from pool
        candidate_pool = [p for p in candidate_pool 
                         if p not in current_papers]
        
        # Compute information gain for each candidate
        gains = [self.compute_information_gain(current_papers, pid) 
                for pid in candidate_pool]
        
        # Use softmax selection
        probs = softmax(np.array(gains) / self.temperature)
        selected_idx = np.random.choice(len(candidate_pool), p=probs)
        
        return candidate_pool[selected_idx]
    
    def find_optimal_path(self, 
                         start_paper: str, 
                         end_paper: str,
                         min_papers: int = 3,
                         max_papers: int = 5) -> List[str]:
        """Find optimal path between two papers with minimal intermediate papers
        
        Args:
            start_paper: Starting paper ID
            end_paper: Target paper ID
            max_papers: Maximum number of intermediate papers
            
        Returns:
            List of paper IDs forming the path
        """
        # First try direct shortest path using similarities
        shortest_path = nx.shortest_path(
            self.similarity_graph,
            start_paper,
            end_paper,
            weight='weight'
        )
        
        # If path is too short, need to force intermediate papers
        if len(shortest_path) < min_papers:
            path = [start_paper]
            remaining_steps = min_papers - 2  # -2 for start and end papers
            
            # Force selection of intermediate papers
            while remaining_steps > 0:
                # Get papers more similar to target than current paper
                current_sim = self.papers[path[-1]].similarities[end_paper]
                candidates = [
                    pid for pid, sim in self.papers[end_paper].similarities.items()
                    if sim > current_sim * 0.7 and pid not in path and pid != end_paper
                ]
                
                if not candidates:
                    # If no good candidates, relax similarity requirement
                    candidates = [pid for pid in self.papers.keys() 
                               if pid not in path and pid != end_paper]
                
                next_paper = self.select_next_paper(path, candidates)
                path.append(next_paper)
                remaining_steps -= 1
            
            path.append(end_paper)
            return path
        
        # If path length is acceptable, use it
        if len(shortest_path) <= max_papers + 2:
            return shortest_path
        
        # If path is too long, try active selection approach
        path = [start_paper]
        current_paper = start_paper
        remaining_papers = max_papers
        
        while remaining_papers > 0:
            # Get papers that are more similar to target than current paper
            current_sim = self.papers[current_paper].similarities[end_paper]
            candidates = [
                pid for pid, sim in self.papers[end_paper].similarities.items()
                if sim > current_sim and pid not in path
            ]
            
            if not candidates:
                break
                
            # Select next paper using active selection
            next_paper = self.select_next_paper(path, candidates)
            path.append(next_paper)
            current_paper = next_paper
            remaining_papers -= 1
            
            # Check if we've reached the target
            if next_paper == end_paper:
                break
        
        # Add target paper if not reached
        if path[-1] != end_paper:
            path.append(end_paper)
            
        return path

def main():
    # Example usage
    selector = ActivePaperSelector(temperature=0.1)
    
    # Load data
    selector.load_data(
        'paper_embeddings.json',
        'all_similarities.csv',
        'paper_topics.csv'
    )
    
    # Example: Select sequence of papers
    papers = []
    for _ in range(5):
        next_paper = selector.select_next_paper(papers)
        papers.append(next_paper)
        paper = selector.papers[next_paper]
        print(f"\nSelected paper: {paper.title}")
        print(f"Top topics: {sorted(paper.topics.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Example: Find path between papers
    start = papers[0]
    end = papers[-1]
    path = selector.find_optimal_path(start, end)
    print(f"\nPath from {selector.papers[start].title} to {selector.papers[end].title}:")
    for pid in path:
        print(f"-> {selector.papers[pid].title}")

if __name__ == "__main__":
    main()