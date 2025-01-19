Project Overview: 

"Active Inference Research Navigator" - A system that applies active inference principles to intelligently guide researchers through scientific literature. Designed to maximize information gain and minimize time taken for an individual to learn about a topic or understand the breadth of the field by using the methods given in “Active Data Selection and Information Seeking.” 

Current Implementation:
1. Data Processing Pipeline:
- Uses SPECTER2 for paper embeddings
- BERTopic for topic modeling
- These create a semantic ‘paper space’ which can be interpreted as an ontology. 

2. Active Selection Algorithm:
- Implements information gain algorithm based on the paper ‘Active Data Selection’
- Finds optimal paths between papers by maximizing information gain while minimizing amount of reading needed

3. Core Features:
- Intelligent paper selection maximizing information gain
- Topic progression tracking
- Ability to more easily discover the cutting edge of the field
- Ability to more easily craft the ‘related works’ section of one’s paper to aid researchers

Planned Upgrades:
1. Semantic Scholar API Integration:
- Add citation network data
- Include reference relationships
- Expand available metadata
- Access larger paper corpus

2. Fine-tune SPECTRE2 on active inference data
- Would produce a richer semantic space (ontology)
- Improves output of all areas

3. Implement tool as a website via AWS
- Use EC2/S3 to set up 
- Semantic Scholar api allows ~1 request/s, so a good sum of time will be needed to build the search space 

Research Value:
1. For Active Inference Institute:
- Demonstrates practical application of active inference principles
- Provides tool for efficiently navigating research literature
- Supports knowledge discovery and synthesis

2. Novel Contributions:
- Applies active inference to literature navigation
- Integrates multiple information sources for paper selection
- Implements practical path finding in research space
- Creates reusable framework for research exploration

Future Potential:
- Could help identify research gaps
- Support systematic literature reviews
- Guide research direction decisions
- Facilitate interdisciplinary connections
- Assist in research front identification
