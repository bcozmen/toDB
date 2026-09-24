import React, { useEffect, useCallback, useRef, useState, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { api } from '../api';
import { TableNode } from './TableNode';
import { DictionaryMenu } from './DictionaryMenu';
import { getLayoutedElements } from './elkLayout';
import { normalizeSchema } from './schemaTypes';
import './schema.css';

const nodeTypes = { tableNode: TableNode };

function SchemaFlow() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [tables, setTables] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedTables, setExpandedTables] = useState(new Set(['encounters', 'patients']));
  const [activeDict, setActiveDict] = useState(null);

  const hasFittedInitialLayout = useRef(false);
  const { fitView } = useReactFlow();

  const relationsCount = useMemo(() => {
    return tables.reduce((acc, table) => {
      return acc + (table.columns?.filter((c) => Boolean(c.fk)).length || 0);
    }, 0);
  }, [tables]);

  const handleOpenDictionary = useCallback((tableName, dictionary, key) => {
    setActiveDict({
      tableName,
      dictionary,
      dictionaryKey: key,
    });
  }, []);

  const toggleTable = useCallback((tableName) => {
    const normalizedName = tableName.toLowerCase();
    setExpandedTables((prev) => {
      const next = new Set(prev);
      if (next.has(normalizedName)) {
        next.delete(normalizedName);
      } else {
        next.add(normalizedName);
      }
      return next;
    });
  }, []);

  const handleExpandAll = useCallback(() => {
    setExpandedTables(new Set(tables.map((t) => t.name.toLowerCase())));
  }, [tables]);

  const handleCollapseAll = useCallback(() => {
    setExpandedTables(new Set());
  }, []);

  const layoutTables = useCallback(
    async (tablesToLayout, expandedSet, fitAfterLayout = false) => {
      if (!tablesToLayout || tablesToLayout.length === 0) return;

      const rawNodes = [];
      const rawEdges = [];
      const query = searchQuery.trim().toLowerCase();

      tablesToLayout.forEach((table) => {
        const tableName = table.name;
        const isExpanded = expandedSet.has(tableName.toLowerCase());
        const isMatched =
          !query ||
          tableName.toLowerCase().includes(query) ||
          table.columns?.some((c) => c.name.toLowerCase().includes(query));

        // 1. Create Nodes
        rawNodes.push({
          id: tableName,
          type: 'tableNode',
          position: { x: 0, y: 0 },
          data: {
            label: tableName,
            columns: table.columns,
            dictionary: table.dictionary,
            expanded: isExpanded,
            highlighted: query ? isMatched : false,
            dimmed: query ? !isMatched : false,
            onToggle: () => toggleTable(tableName),
            onOpenDictionary: handleOpenDictionary,
          },
        });

        // 2. Extract Foreign Key relations to build Edges
        table.columns?.forEach((column) => {
          if (column.fk) {
            rawEdges.push({
              id: `e-${tableName}-${column.name}-${column.fk.table}`,
              source: tableName,
              target: column.fk.table,
              type: 'smoothstep',
              animated: true,
              style: { stroke: '#38bdf8', strokeWidth: 1.75 },
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: '#38bdf8',
                width: 14,
                height: 14,
              },
            });
          }
        });
      });

      const { nodes: layoutedNodes, edges: layoutedEdges } = await getLayoutedElements(
        rawNodes,
        rawEdges,
        { direction: 'LR' }
      );

      setNodes(layoutedNodes);
      setEdges(layoutedEdges);

      if (fitAfterLayout) {
        window.requestAnimationFrame(() => {
          fitView({ padding: 0.18, duration: 400 });
        });
      }
    },
    [fitView, handleOpenDictionary, searchQuery, setEdges, setNodes, toggleTable]
  );

  const handleResetLayout = useCallback(() => {
    layoutTables(tables, expandedTables, true);
  }, [layoutTables, tables, expandedTables]);

  // Load Schema on mount
  useEffect(() => {
    let isCancelled = false;

    const loadSchema = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await api.getSchema();
        if (!isCancelled) {
          const normalized = normalizeSchema(data);
          setTables(normalized);
        }
      } catch (err) {
        if (!isCancelled) {
          console.error('Failed to load schema for diagram:', err);
          setError('Failed to load database schema. Ensure API backend is online.');
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };

    loadSchema();
    return () => {
      isCancelled = true;
    };
  }, []);

  // Layout when tables or expanded set change
  useEffect(() => {
    if (tables.length > 0) {
      const isInitial = !hasFittedInitialLayout.current;
      if (isInitial) {
        hasFittedInitialLayout.current = true;
      }
      layoutTables(tables, expandedTables, isInitial);
    }
  }, [tables, expandedTables, layoutTables]);

  // Real-time highlight when search query changes
  useEffect(() => {
    const query = searchQuery.trim().toLowerCase();
    setNodes((currNodes) =>
      currNodes.map((node) => {
        const table = tables.find((t) => t.name === node.id);
        const match =
          !query ||
          node.id.toLowerCase().includes(query) ||
          table?.columns?.some((c) => c.name.toLowerCase().includes(query));

        return {
          ...node,
          data: {
            ...node.data,
            highlighted: query ? match : false,
            dimmed: query ? !match : false,
          },
        };
      })
    );
  }, [searchQuery, setNodes, tables]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    const query = searchQuery.trim().toLowerCase();
    if (!query) return;
    const match = nodes.find((n) => n.id.toLowerCase().includes(query));
    if (match) {
      fitView({ nodes: [{ id: match.id }], padding: 0.6, duration: 500 });
    }
  };

  const handleRetry = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getSchema();
      const normalized = normalizeSchema(data);
      setTables(normalized);
    } catch (err) {
      console.error('Failed to load schema for diagram:', err);
      setError('Failed to load database schema. Ensure API backend is online.');
    } finally {
      setLoading(false);
    }
  }, []);

  if (loading && tables.length === 0) {
    return (
      <div className="schema-panel schema-state-center">
        <div className="schema-spinner" />
        <span className="schema-state-text">Loading database schema…</span>
      </div>
    );
  }

  if (error && tables.length === 0) {
    return (
      <div className="schema-panel schema-state-center">
        <div className="schema-error-card">
          <p className="schema-error-msg">{error}</p>
          <button type="button" className="schema-btn-primary" onClick={handleRetry}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="schema-panel">
      {/* Top Controls / Header */}
      <header className="schema-header">
        <div className="schema-header-left">
          <span className="schema-kicker">DATABASE SCHEMA</span>
          <div className="schema-title-row">
            <h2 className="schema-main-title">Relational Graph</h2>
            <span className="schema-header-badge">
              {tables.length} tables • {relationsCount} relations
            </span>
          </div>
        </div>

        <div className="schema-header-right">
          <form className="schema-search-form" onSubmit={handleSearchSubmit}>
            <svg className="schema-search-icon" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                clipRule="evenodd"
              />
            </svg>
            <input
              type="text"
              className="schema-search-input"
              placeholder="Filter tables & columns…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="schema-search-clear-btn"
                onClick={() => setSearchQuery('')}
                aria-label="Clear table search"
              >
                ×
              </button>
            )}
          </form>

          <div className="schema-action-group">
            <button
              type="button"
              className="schema-action-btn"
              onClick={handleExpandAll}
              title="Expand all tables"
            >
              Expand All
            </button>
            <button
              type="button"
              className="schema-action-btn"
              onClick={handleCollapseAll}
              title="Collapse all tables"
            >
              Collapse All
            </button>
            <button
              type="button"
              className="schema-action-btn"
              onClick={handleResetLayout}
              title="Reset layout and center view"
            >
              Fit View
            </button>
          </div>
        </div>
      </header>

      {/* React Flow Canvas */}
      <div className="schema-flow-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitViewOptions={{ padding: 0.18 }}
          minZoom={0.2}
          maxZoom={2.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#1e2c38" gap={24} size={1.2} />
          <Controls showInteractive={false} className="schema-controls-panel" />
        </ReactFlow>
      </div>

      {/* Dictionary Modal */}
      {activeDict && (
        <DictionaryMenu
          tableName={activeDict.tableName}
          dictionary={activeDict.dictionary}
          dictionaryKey={activeDict.dictionaryKey}
          onClose={() => setActiveDict(null)}
        />
      )}
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