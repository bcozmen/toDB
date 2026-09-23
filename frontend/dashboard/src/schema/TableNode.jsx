// TableNode.jsx
import React, { useCallback, useRef, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { createPortal } from 'react-dom';
import { DictionaryMenu } from './DictionaryMenu';

export const TableNode = ({ data }) => {
  const isExpanded = data.expanded ?? false;
  const [dictionaryAnchors, setDictionaryAnchors] = useState([]);
  const dictionaryId = useRef(0);
  const isPatient = data.label.toLowerCase() === 'patients';

  const isDictionaryColumn = useCallback((columnName) => {
    const normalizedName = columnName.toLowerCase();
    return isPatient
      ? ['gender', 'race', 'ethnicity', 'birthplace', 'city'].includes(normalizedName)
      : ['code', 'description'].includes(normalizedName);
  }, [isPatient]);

  const openDictionary = (event) => {
    event.stopPropagation();
    setDictionaryAnchors((current) => [...current, {
      id: dictionaryId.current++,
      rect: event.currentTarget.getBoundingClientRect(),
      element: event.currentTarget,
      key: event.currentTarget.dataset.dictionaryKey,
    }]);
  };

  return (
    <div className="schema-table-node">
      {/* Handles for connections */}
      <Handle type="target" position={Position.Left} className="schema-table-handle" />
      <Handle type="source" position={Position.Right} className="schema-table-handle" />

      {/* Table Header */}
      <button
        type="button"
        className="schema-table-header"
        onClick={() => data.onToggle?.()}
        aria-expanded={isExpanded}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${data.label}`}
      >
        <span className="schema-table-title">📊 {data.label}</span>
        <span className="schema-table-chevron" aria-hidden="true">{isExpanded ? '▾' : '▸'}</span>
      </button>

      {/* Columns List */}
      {isExpanded && <div className="schema-table-columns">
        {data.columns?.map((col, idx) => (
          <div key={idx} className="schema-table-column">
            {isDictionaryColumn(col.name) ? (
              <button
                type="button"
                className="schema-column-name schema-column-link"
                data-dictionary-key={isPatient ? col.name.toLowerCase() : undefined}
                onClick={openDictionary}
              >
                {col.pk ? '🔑 ' : col.fk ? '🔗 ' : ''}{col.name}
              </button>
            ) : (
              <span className={`schema-column-name ${col.pk ? 'is-primary-key' : col.fk ? 'is-foreign-key' : ''}`}>
                {col.pk ? '🔑 ' : col.fk ? '🔗 ' : ''}{col.name}
              </span>
            )}
            <span className="schema-column-type">{col.type}</span>
          </div>
        ))}
      </div>}
      {dictionaryAnchors.map((dictionaryAnchor) => {
        const dictionaryContainer = dictionaryAnchor.element.closest('.lm_content') ?? document.body;
        return createPortal(
          <DictionaryMenu
            key={dictionaryAnchor.id}
            tableName={data.label}
            dictionary={data.dictionary}
            dictionaryKey={dictionaryAnchor.key}
            anchorElement={dictionaryAnchor.element}
            anchorRect={dictionaryAnchor.rect}
            portalContainer={dictionaryContainer}
            onClose={() => setDictionaryAnchors((current) => current.filter(({ id }) => id !== dictionaryAnchor.id))}
          />,
          dictionaryContainer,
          dictionaryAnchor.id,
        );
      })}
    </div>
  );
};