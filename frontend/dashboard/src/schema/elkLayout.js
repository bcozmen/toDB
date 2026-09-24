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
      'elk.spacing.nodeNode': '36',
      'elk.layered.spacing.nodeNodeBetweenLayers': '80',
      'elk.edgeRouting': 'ORTHOGONAL',
    },
    children: nodes.map((node) => {
      const isExpanded = Boolean(node.data?.expanded);
      const colCount = node.data?.columns?.length || 0;
      const height = isExpanded ? 44 + colCount * 28 + 12 : 44;
      return {
        id: node.id,
        width: node.measured?.width || 260,
        height,
      };
    }),
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
        x: elkNode?.x ?? node.position?.x ?? 0,
        y: elkNode?.y ?? node.position?.y ?? 0,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
};