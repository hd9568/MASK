import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Dict, Tuple, List, Set
from sparsene.op_gen.opgraph.graph import construct_graph_from_block_op, OpGraph

class GraphRegion:
    def __init__(self, nodes: List[str], level: int):
        self.nodes = nodes
        self.level = level
        self.entry_nodes: Set[str] = set()
        self.exit_nodes: Set[str] = set()

def add_graph_to_networkx(
    G: nx.DiGraph, 
    graph: OpGraph, 
    prefix: str = "",
    level: int = 0,
    regions: Dict[str, GraphRegion] = None
) -> Set[str]:
    """
    Recursively add nodes and edges from OpGraph to networkx graph.
    Returns the set of node IDs added at this level.
    """
    if regions is None:
        regions = {}
    
    current_region_nodes = []
    entry_nodes = set()
    exit_nodes = set()
    
    for node_id, node in graph.nodes.items():
        unique_id = f"{prefix}{node_id}"
        
        if isinstance(node, OpGraph.BlockArgNode):
            label = f"BlockArg\n{node.block_arg}"
            color = 'lightblue'
            G.add_node(unique_id, label=label, color=color, level=level)
            current_region_nodes.append(unique_id)
            if not node.inputs:
                entry_nodes.add(unique_id)
            if not node.outputs:
                exit_nodes.add(unique_id)
                
        elif isinstance(node, OpGraph.SuperNode):
            # Don't create a node for SuperNode itself
            sub_prefix = f"{prefix}sub_{node_id}_"
            sub_nodes = add_graph_to_networkx(G, node.graph, sub_prefix, level, regions)
            current_region_nodes.extend(sub_nodes)
            
            # Store region information
            region = GraphRegion(list(sub_nodes), level)
            regions[unique_id] = region
            
            # Find entry and exit nodes of the subgraph
            for sub_node in sub_nodes:
                if not list(G.predecessors(sub_node)):
                    entry_nodes.add(sub_node)
                if not list(G.successors(sub_node)):
                    exit_nodes.add(sub_node)
            
        elif isinstance(node, OpGraph.OpNode):
            label = f"Op\n{node.op}"
            color = 'lightgray'
            G.add_node(unique_id, label=label, color=color, level=level)
            current_region_nodes.append(unique_id)
            if not node.inputs:
                entry_nodes.add(unique_id)
            if not node.outputs:
                exit_nodes.add(unique_id)
    
    # Add edges
    for edge in graph.edges:
        src_id = f"{prefix}{edge.src.id}"
        dst_id = f"{prefix}{edge.dst.id}"
        
        # If source is a SuperNode, connect from its exit nodes
        if isinstance(edge.src, OpGraph.SuperNode):
            src_region_id = f"{prefix}{edge.src.id}"
            if src_region_id in regions:
                for exit_node in regions[src_region_id].exit_nodes:
                    G.add_edge(exit_node, dst_id)
        # If destination is a SuperNode, connect to its entry nodes
        elif isinstance(edge.dst, OpGraph.SuperNode):
            dst_region_id = f"{prefix}{edge.dst.id}"
            if dst_region_id in regions:
                for entry_node in regions[dst_region_id].entry_nodes:
                    G.add_edge(src_id, entry_node)
        else:
            G.add_edge(src_id, dst_id)
    
    if prefix and current_region_nodes:
        region = GraphRegion(current_region_nodes, level)
        region.entry_nodes = entry_nodes
        region.exit_nodes = exit_nodes
        regions[prefix.rstrip('_')] = region
    
    return set(current_region_nodes)

def draw_region_bounds(
    ax: plt.Axes,
    pos: Dict[str, Tuple[float, float]],
    region: GraphRegion,
    padding: float = 0.1
) -> None:
    """Draw a rounded rectangle around a group of nodes."""
    if not region.nodes:
        return
        
    xs = [pos[node][0] for node in region.nodes]
    ys = [pos[node][1] for node in region.nodes]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2
    
    rect = FancyBboxPatch(
        (min_x - padding, min_y - padding),
        width,
        height,
        boxstyle=f"round,pad={padding}",
        facecolor='none',
        edgecolor='gray',
        alpha=0.5,
        linestyle='--',
        linewidth=1
    )
    ax.add_patch(rect)

def visualize_op_graph(graph: OpGraph, filename: str) -> None:
    """
    Visualize an OpGraph using networkx and matplotlib, with hierarchical layout.
    """
    G = nx.DiGraph()
    regions = {}

    # Recursively add all nodes and edges
    add_graph_to_networkx(G, graph, regions=regions)
    
    # Create hierarchical layout (left to right)
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Adjust positions to ensure left-to-right flow and prevent overlapping
    for node in G.nodes():
        level = G.nodes[node]['level']
        pos[node] = (pos[node][0] + level * 2, pos[node][1])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Draw regions from deepest level to shallowest
    max_level = max(region.level for region in regions.values())
    for level in range(max_level, -1, -1):
        level_regions = [r for r in regions.values() if r.level == level]
        for region in level_regions:
            draw_region_bounds(ax, pos, region)
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        ax=ax,
        connectionstyle="arc3,rad=0.2"  # Add curve to edges to make them more visible
    )
    
    # Add labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Operation Graph Visualization")
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(filename)
