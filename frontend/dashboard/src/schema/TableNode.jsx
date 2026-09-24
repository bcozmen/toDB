import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

export const TableNode = memo(({ data }) => {
  const isExpanded = Boolean(data.expanded);
  const isPatient = data.label?.toLowerCase() === 'patients';
  const hasDictionary = data.dictionary && Object.keys(data.dictionary).length > 0;

  const isDictionaryColumn = (columnName) => {
    const norm = columnName.toLowerCase();
    if (isPatient) {
      return ['gender', 'race', 'ethnicity', 'birthplace', 'city'].includes(norm);
    }
    return ['code', 'description'].includes(norm) && hasDictionary;
  };

  const handleDictionaryClick = (e, key = null) => {
    e.stopPropagation();
    data.onOpenDictionary?.(data.label, data.dictionary, key);
  };

  return (
    <div
      className={`schema-table-node ${isExpanded ? 'is-expanded' : 'is-collapsed'} ${
        data.highlighted ? 'is-highlighted' : ''
      } ${data.dimmed ? 'is-dimmed' : ''}`}
    >
      {/* Handles for foreign key relationships */}
      <Handle
        type="target"
        position={Position.Left}
        className="schema-table-handle schema-handle-target"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="schema-table-handle schema-handle-source"
      />

      {/* Table Header */}
      <div
        className="schema-table-header"
        onClick={() => data.onToggle?.()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            data.onToggle?.();
          }
        }}
        aria-expanded={isExpanded}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${data.label} table`}
      >
        <div className="schema-header-left">
          <svg className="schema-table-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <ellipse cx="12" cy="5" rx="9" ry="3" />
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          </svg>
          <span className="schema-table-title">{data.label}</span>
        </div>

        <div className="schema-header-right">
          {hasDictionary && (
            <button
              type="button"
              className="schema-dict-badge-btn"
              onClick={(e) => handleDictionaryClick(e, null)}
              title={`View ${data.label} dictionary`}
              aria-label={`Open dictionary for ${data.label}`}
            >
              <svg viewBox="0 0 20 20" fill="currentColor" className="schema-dict-btn-icon">
                <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
              </svg>
              <span>dict</span>
            </button>
          )}

          <span className="schema-col-count-badge">
            {data.columns?.length || 0} cols
          </span>

          <span className={`schema-chevron ${isExpanded ? 'is-open' : ''}`} aria-hidden="true">
            ▾
          </span>
        </div>
      </div>

      {/* Expanded Columns List */}
      {isExpanded && (
        <div className="schema-table-columns">
          {data.columns?.map((col, idx) => {
            const isDict = isDictionaryColumn(col.name);
            const dictKey = isPatient ? col.name.toLowerCase() : null;

            return (
              <div
                key={idx}
                className={`schema-table-column ${isDict ? 'has-dictionary' : ''}`}
                title={col.fk ? `Foreign Key → ${col.fk.table}.${col.fk.column}` : undefined}
              >
                <div className="schema-col-left">
                  {col.pk && (
                    <span className="schema-key-badge pk" title="Primary Key">
                      PK
                    </span>
                  )}
                  {col.fk && (
                    <span
                      className="schema-key-badge fk"
                      title={`Foreign Key references ${col.fk.table}.${col.fk.column}`}
                    >
                      FK
                    </span>
                  )}

                  {isDict ? (
                    <button
                      type="button"
                      className="schema-col-name-btn"
                      onClick={(e) => handleDictionaryClick(e, dictKey)}
                      title={`Click to view ${col.name} dictionary`}
                    >
                      <span>{col.name}</span>
                      <svg className="schema-col-dict-icon" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M1 2.828c.885-.37 2.154-.769 3.388-.893 1.33-.134 2.458.063 3.112.752v9.746c-.935-.53-2.12-.603-3.213-.493-1.18.12-2.37.461-3.287.811V2.828zm7.5-.141c.654-.689 1.782-.886 3.112-.752 1.234.124 2.503.523 3.388.893v10.743c-.918-.35-2.107-.692-3.287-.81-1.094-.111-2.278-.039-3.213.492V2.687z" />
                      </svg>
                    </button>
                  ) : (
                    <span className={`schema-col-name ${col.pk ? 'is-pk' : ''} ${col.fk ? 'is-fk' : ''}`}>
                      {col.name}
                    </span>
                  )}
                </div>

                <span className="schema-col-type">{col.type}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});

TableNode.displayName = 'TableNode';