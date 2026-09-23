// elkLayout.js
import ELK from 'elkjs/lib/elk.bundled.js';

const elk = new ELK();

export const getLayoutedElements = async (nodes, edges, options = {}) => {
  const isHorizontal = options.direction === 'LR';

  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': isHorizontal ? 'RIGHT' : 'DOWN',
      'elk.spacing.nodeNode': '24', // Keep collapsed tables close together
      'elk.layered.spacing.nodeNodeBetweenLayers': '50', // Leave room for expanded tables
      'elk.edgeRouting': 'ORTHOGONAL', // Crisp 90-degree lines
    },
    children: nodes.map((node) => ({
      id: node.id,
      // Pass estimated dimensions to ELK so it calculates spacing accurately
      width: node.measured?.width || 220,
      height: node.data.expanded
        ? 40 + (node.data.columns?.length || 0) * 28
        : 40,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  };

  const layoutedGraph = await elk.layout(graph);

  const layoutedNodes = nodes.map((node) => {
    const elkNode = layoutedGraph.children?.find((n) => n.id === node.id);
    return {
      ...node,
      position: {
        x: elkNode?.x ?? 0,
        y: elkNode?.y ?? 0,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
};