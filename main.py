"""
Peezy RAG API
=============

A simple FastAPI service that queries the Peezy knowledge graph.
Deployed to Railway, called by Firebase Functions before building prompts.

Endpoints:
- GET /health - Health check
- POST /query - Query the knowledge graph
- GET /stats - Graph statistics
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Set
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import networkx as nx

# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Peezy RAG API",
    description="Knowledge graph query service for Peezy moving assistant",
    version="1.0.0"
)

# Allow CORS for Firebase Functions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================================

# Global graph instance
G: nx.DiGraph = None
node_index: Dict[str, Set[str]] = {}  # keyword -> node_ids

def load_graph():
    """Load the knowledge graph from JSON file."""
    global G, node_index
    
    graph_path = os.getenv("GRAPH_PATH", "peezy_knowledge_graph.json")
    
    if not os.path.exists(graph_path):
        print(f"⚠️ Graph file not found at {graph_path}")
        G = nx.DiGraph()
        return
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in data.get('nodes', []):
        node_id = node.get('id', '')
        node_type = node.get('type', 'Unknown')
        G.add_node(node_id, type=node_type)
    
    # Add edges
    for edge in data.get('edges', []):
        source = edge.get('source', '')
        target = edge.get('target', '')
        relation = edge.get('relation', 'RELATES_TO')
        description = edge.get('description', '')
        
        if source and target:
            G.add_edge(source, target, relation=relation, description=description)
    
    # Build keyword index for fast lookup
    build_keyword_index()
    
    print(f"✅ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def build_keyword_index():
    """Build an inverted index from keywords to node IDs for fast lookup."""
    global node_index
    node_index = {}
    
    for node_id in G.nodes():
        # Extract keywords from node ID
        keywords = extract_keywords(node_id)
        
        for kw in keywords:
            if kw not in node_index:
                node_index[kw] = set()
            node_index[kw].add(node_id)
        
        # Also index by node type
        node_type = G.nodes[node_id].get('type', '').lower()
        if node_type:
            if node_type not in node_index:
                node_index[node_type] = set()
            node_index[node_type].add(node_id)

def extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    if not text:
        return set()
    
    # Lowercase and extract words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Remove common stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and', 'not',
        'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'but', 'also', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
        'their', 'them', 'we', 'our', 'us', 'you', 'your', 'he', 'his', 'him',
        'she', 'her', 'i', 'me', 'my', 'what', 'about', 'which', 'who', 'whom'
    }
    
    return {w for w in words if w not in stopwords and len(w) > 2}

# ============================================================================
# SYNONYM EXPANSION
# ============================================================================

SYNONYMS = {
    'movers': ['mover', 'moving', 'company', 'companies', 'crew', 'truck'],
    'mover': ['movers', 'moving', 'company'],
    'moving': ['move', 'movers', 'relocation', 'relocate'],
    'piano': ['pianos', 'grand', 'upright', 'keyboard', 'instrument'],
    'pet': ['pets', 'dog', 'cat', 'animal', 'animals'],
    'pets': ['pet', 'dog', 'cat', 'animal', 'animals'],
    'dog': ['dogs', 'pet', 'pets', 'puppy', 'canine'],
    'cat': ['cats', 'pet', 'pets', 'kitten', 'feline'],
    'apartment': ['apt', 'condo', 'flat', 'unit', 'building'],
    'elevator': ['lift', 'service'],
    'stairs': ['staircase', 'steps', 'flight', 'floor', 'floors'],
    'pack': ['packing', 'packed', 'boxes', 'box'],
    'packing': ['pack', 'boxes', 'wrapping', 'supplies'],
    'box': ['boxes', 'carton', 'container'],
    'boxes': ['box', 'cartons', 'containers'],
    'quote': ['quotes', 'estimate', 'estimates', 'bid', 'bids', 'price'],
    'estimate': ['estimates', 'quote', 'quotes', 'bid', 'binding'],
    'cost': ['costs', 'price', 'prices', 'fee', 'fees', 'expense', 'much', 'money'],
    'price': ['prices', 'cost', 'costs', 'fee', 'pricing'],
    'insurance': ['coverage', 'protection', 'insured', 'liability'],
    'damage': ['damaged', 'broken', 'hurt', 'scratched'],
    'interstate': ['state', 'states', 'cross', 'crossing', 'long distance'],
    'utilities': ['utility', 'electric', 'gas', 'water', 'power', 'internet'],
    'address': ['addresses', 'mail', 'postal', 'usps'],
    'lease': ['rental', 'rent', 'landlord', 'tenant', 'renting'],
    'deposit': ['security', 'refund'],
    'storage': ['store', 'stored', 'warehouse', 'unit'],
    'schedule': ['scheduling', 'book', 'booking', 'reserve', 'reservation'],
    'book': ['booking', 'reserve', 'schedule', 'booked'],
    'stress': ['stressed', 'overwhelmed', 'anxious', 'worried', 'panic'],
    'overwhelmed': ['stressed', 'anxious', 'worried', 'panic', 'help'],
    'first': ['start', 'begin', 'initial', 'starting'],
    'start': ['first', 'begin', 'initial', 'beginning'],
    'health': ['certificate', 'vet', 'veterinarian', 'medical'],
    'certificate': ['health', 'cert', 'document'],
    'safe': ['gun', 'firearm', 'heavy'],
    'pool': ['table', 'billiard', 'billiards'],
    'hot': ['tub', 'spa', 'jacuzzi'],
    'antique': ['antiques', 'valuable', 'heirloom', 'fragile'],
    'art': ['artwork', 'painting', 'paintings', 'frame', 'frames'],
    'wine': ['collection', 'bottles', 'cellar'],
    'clock': ['grandfather', 'antique', 'pendulum'],
    'scam': ['fraud', 'fraudulent', 'fake', 'rogue', 'dishonest'],
    'tip': ['tips', 'tipping', 'gratuity'],
    'inventory': ['list', 'items', 'belongings', 'stuff'],
    'walkthrough': ['inspection', 'check', 'final'],
}

def expand_keywords(keywords: Set[str]) -> Set[str]:
    """Expand keywords with synonyms."""
    expanded = set(keywords)
    for kw in keywords:
        if kw in SYNONYMS:
            expanded.update(SYNONYMS[kw])
    return expanded

# ============================================================================
# QUERY FUNCTIONS
# ============================================================================

def find_matching_nodes(query: str, context: Dict[str, Any] = None) -> List[str]:
    """Find nodes that match the query keywords."""
    keywords = extract_keywords(query)
    expanded = expand_keywords(keywords)
    
    matching_nodes = set()
    
    # Find nodes via keyword index
    for kw in expanded:
        if kw in node_index:
            matching_nodes.update(node_index[kw])
    
    # Also do direct substring matching on node IDs
    query_lower = query.lower()
    for node_id in G.nodes():
        node_lower = node_id.lower()
        # Check if any expanded keyword is in node name
        for kw in expanded:
            if kw in node_lower:
                matching_nodes.add(node_id)
                break
    
    # Context-based boosting
    if context:
        # If user has pets, prioritize pet-related nodes
        if context.get('has_pets'):
            pet_nodes = node_index.get('pet', set()) | node_index.get('pets', set())
            matching_nodes.update(pet_nodes)
        
        # If long distance, prioritize interstate nodes
        if context.get('move_distance') in ['long_distance', 'cross_country', 'interstate']:
            interstate_nodes = node_index.get('interstate', set()) | node_index.get('dot', set())
            matching_nodes.update(interstate_nodes)
        
        # If apartment, include elevator/stairs nodes
        if context.get('dwelling_type') == 'apartment':
            apt_nodes = node_index.get('elevator', set()) | node_index.get('stairs', set())
            matching_nodes.update(apt_nodes)
    
    return list(matching_nodes)

def get_node_context(node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
    """Get all relationships for a node (incoming and outgoing)."""
    if node_id not in G:
        return []
    
    context = []
    
    # Outgoing edges (what this node requires/triggers/etc)
    for neighbor in G.successors(node_id):
        edge_data = G.edges[node_id, neighbor]
        context.append({
            'source': node_id,
            'relation': edge_data.get('relation', 'RELATES_TO'),
            'target': neighbor,
            'description': edge_data.get('description', ''),
            'direction': 'outgoing'
        })
    
    # Incoming edges (what requires/triggers this node)
    for predecessor in G.predecessors(node_id):
        edge_data = G.edges[predecessor, node_id]
        context.append({
            'source': predecessor,
            'relation': edge_data.get('relation', 'RELATES_TO'),
            'target': node_id,
            'description': edge_data.get('description', ''),
            'direction': 'incoming'
        })
    
    return context

def query_graph(query: str, context: Dict[str, Any] = None, max_results: int = 10) -> Dict[str, Any]:
    """
    Main query function - finds relevant nodes and their relationships.
    
    Returns a structured response with:
    - matched_nodes: List of nodes that matched
    - relationships: All relevant relationships
    - formatted_context: Pre-formatted string for LLM injection
    """
    # Find matching nodes
    matching_nodes = find_matching_nodes(query, context)
    
    if not matching_nodes:
        return {
            'matched_nodes': [],
            'relationships': [],
            'formatted_context': '',
            'node_count': 0
        }
    
    # Limit to top N nodes (prioritize by number of connections)
    node_scores = []
    for node_id in matching_nodes:
        # Score = number of connections
        score = G.in_degree(node_id) + G.out_degree(node_id)
        node_scores.append((score, node_id))
    
    node_scores.sort(reverse=True)
    top_nodes = [node_id for score, node_id in node_scores[:max_results]]
    
    # Gather all relationships
    all_relationships = []
    seen_edges = set()
    
    for node_id in top_nodes:
        for rel in get_node_context(node_id):
            edge_key = (rel['source'], rel['relation'], rel['target'])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                all_relationships.append(rel)
    
    # Format for LLM injection
    formatted_lines = []
    for rel in all_relationships[:20]:  # Limit to 20 relationships
        desc = rel['description']
        if desc:
            formatted_lines.append(
                f"• {rel['source']} {rel['relation']} {rel['target']}: {desc}"
            )
        else:
            formatted_lines.append(
                f"• {rel['source']} {rel['relation']} {rel['target']}"
            )
    
    formatted_context = '\n'.join(formatted_lines) if formatted_lines else ''
    
    return {
        'matched_nodes': top_nodes,
        'relationships': all_relationships,
        'formatted_context': formatted_context,
        'node_count': len(top_nodes)
    }

# ============================================================================
# API MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    max_results: Optional[int] = 10

class QueryResponse(BaseModel):
    success: bool
    matched_nodes: List[str]
    relationships: List[Dict[str, Any]]
    formatted_context: str
    node_count: int

class HealthResponse(BaseModel):
    status: str
    graph_loaded: bool
    node_count: int
    edge_count: int

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load the graph on startup."""
    load_graph()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        graph_loaded=G is not None and G.number_of_nodes() > 0,
        node_count=G.number_of_nodes() if G else 0,
        edge_count=G.number_of_edges() if G else 0
    )

@app.get("/stats")
async def get_stats():
    """Get graph statistics."""
    if not G:
        return {"error": "Graph not loaded"}
    
    # Count nodes by type
    type_counts = {}
    for node_id in G.nodes():
        node_type = G.nodes[node_id].get('type', 'Unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    # Count edges by relation
    relation_counts = {}
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'Unknown')
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "nodes_by_type": type_counts,
        "edges_by_relation": relation_counts
    }

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """
    Query the knowledge graph.
    
    Request body:
    - query: The user's question/message
    - context: Optional context (has_pets, move_distance, dwelling_type, etc.)
    - max_results: Maximum nodes to return (default 10)
    
    Returns:
    - matched_nodes: Nodes that matched the query
    - relationships: All relevant relationships
    - formatted_context: Pre-formatted string for LLM prompt injection
    """
    if not G or G.number_of_nodes() == 0:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    
    result = query_graph(
        query=request.query,
        context=request.context,
        max_results=request.max_results or 10
    )
    
    return QueryResponse(
        success=True,
        matched_nodes=result['matched_nodes'],
        relationships=result['relationships'],
        formatted_context=result['formatted_context'],
        node_count=result['node_count']
    )

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Get a specific node and its relationships."""
    if not G:
        raise HTTPException(status_code=503, detail="Knowledge graph not loaded")
    
    if node_id not in G:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    
    node_data = G.nodes[node_id]
    relationships = get_node_context(node_id)
    
    return {
        "id": node_id,
        "type": node_data.get('type', 'Unknown'),
        "relationships": relationships
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
