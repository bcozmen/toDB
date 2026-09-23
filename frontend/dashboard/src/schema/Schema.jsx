import React, { useEffect, useCallback, useRef, useState } from 'react';
// Make sure to import your api object from wherever you saved it
import {
	ReactFlow, Background, Controls, useNodesState, useEdgesState, useReactFlow, ReactFlowProvider, MarkerType
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { api } from '../api';
import { TableNode } from './TableNode';
import { getLayoutedElements } from './elkLayout';
import { normalizeSchema } from './schemaTypes';
import './schema.css';
const nodeTypes = { tableNode: TableNode };

function SchemaFlow() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
	const [tables, setTables] = useState([]);
	const [expandedTables, setExpandedTables] = useState(new Set(['encounters', 'patients']));
	const expandedTablesRef = useRef(expandedTables);
	const hasFittedInitialLayout = useRef(false);
  const { fitView } = useReactFlow();


	const toggleTable = useCallback((tableName) => {
		const table = tables.find((candidate) => candidate.name === tableName);
		if (!table) return;
		const normalizedName = tableName.toLowerCase();
		const willExpand = !expandedTablesRef.current.has(normalizedName);

		setExpandedTables((current) => {
			const next = new Set(current);
			if (willExpand) next.add(normalizedName);
			else next.delete(normalizedName);
			expandedTablesRef.current = next;
			return next;
		});

		setNodes((currentNodes) => {
			const target = currentNodes.find((node) => node.id === tableName);
			if (!target) return currentNodes;
			const verticalDelta = table.columns.length * 28;

			return currentNodes.map((node) => {
				if (node.id === tableName) {
					return {
						...node,
						data: { ...node.data, expanded: willExpand },
					};
				}

				// ELK places nodes with the same x coordinate in a vertical stack.
				// Only make room in that stack; unrelated tables keep their positions.
				const isInSameStack = Math.abs(node.position.x - target.position.x) < 2;
				const isBelowTarget = node.position.y > target.position.y;
				if (isInSameStack && isBelowTarget) {
					return {
						...node,
						position: {
							...node.position,
							y: node.position.y + (willExpand ? verticalDelta : -verticalDelta),
						},
					};
				}

				return node;
			});
		});
	}, [setNodes, tables]);

	const layoutTables = useCallback(async (tablesToLayout, expanded, fitAfterLayout = false) => {
	const rawNodes = [];
	const rawEdges = [];

	// Normalize once, then the rest of the diagram is independent of API shape.
	tablesToLayout.forEach((table) => {
	  const tableName = table.name;
	  // 1. Create Nodes
		rawNodes.push({
			id: tableName,
			type: 'tableNode',
			position: { x: 0, y: 0 }, // Temporary positions before ELK runs
			data: {
			  label: tableName,
			  columns: table.columns,
			  dictionary: table.dictionary,
			  expanded: expanded.has(tableName.toLowerCase()),
			  onToggle: () => toggleTable(tableName),
			},
		});

	  // 2. Extract Foreign Key relations to build Edges
	  table.columns.forEach((column) => {
		if (column.fk) {
		  rawEdges.push({
			id: `e-${tableName}-${column.name}-${column.fk.table}`,
			source: tableName,
			target: column.fk.table,
			animated: true,
			style: { stroke: '#007acc', strokeWidth: 2 },
			markerEnd: {
			  type: MarkerType.ArrowClosed,
			  color: '#007acc',
			},
		  });
		}
	  });
	});

	// 3. Compute ELK Automatic Layout
	if (rawNodes.length > 0) {
	  const { nodes: layoutedNodes, edges: layoutedEdges } = await getLayoutedElements(
		rawNodes,
		rawEdges,
		{ direction: 'LR' }
	  );

	  setNodes(layoutedNodes);
	  setEdges(layoutedEdges);

	  // Fit only the initial diagram. Keeping the current viewport makes
	  // subsequent expand/collapse actions feel like local interactions.
	  if (fitAfterLayout) {
		window.requestAnimationFrame(() => fitView({ padding: 0.2 }));
	  }
	}
	}, [fitView, setNodes, setEdges, toggleTable]);

  useEffect(() => {
	const fetchSchema = async () => {
	  try {
		const data = await api.getSchema();
		setTables(normalizeSchema(data));
	  } catch (err) {
		console.error('Failed to load schema for diagram:', err);
	  }
	};

	fetchSchema();
	}, []);

  useEffect(() => {
	if (tables.length > 0) {
	  const fitInitialLayout = !hasFittedInitialLayout.current;
	  hasFittedInitialLayout.current = true;
	  layoutTables(tables, expandedTables, fitInitialLayout);
	}
	}, [tables, layoutTables]);

  return (
	<div className="schema-flow">
	  <ReactFlow
		nodes={nodes}
		edges={edges}
		onNodesChange={onNodesChange}
		onEdgesChange={onEdgesChange}
		nodeTypes={nodeTypes}
	  >
		<Background color="#333" gap={16} />
		<Controls />
	  </ReactFlow>
	</div>
  );
}

// Wrapper ensures ReactFlow context works properly inside GoldenLayout
export default function Schema() {
  return (
	<ReactFlowProvider>
	  <SchemaFlow />
	</ReactFlowProvider>
  );
}