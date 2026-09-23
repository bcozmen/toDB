import React, { useEffect, useRef, useState } from 'react';

function dictionaryEntries(dictionary) {
  return Object.entries(dictionary ?? {}).flatMap(([key, value]) => {
    if (Array.isArray(value)) {
      return value.map((item) => ({ label: item, description: null }));
    }

    return [{ label: key, description: value }];
  });
}

function connectorPath(connector) {
  const horizontal = connector.side === 'left' || connector.side === 'right';
  const direction = horizontal
    ? (connector.x2 >= connector.x1 ? 1 : -1)
    : (connector.y2 >= connector.y1 ? 1 : -1);
  const bend = Math.max(40, Math.min(120, Math.abs(connector.x2 - connector.x1) * 0.45));
  if (horizontal) {
    return `M ${connector.x1} ${connector.y1} C ${connector.x1 + direction * bend} ${connector.y1}, ${connector.x2 - direction * bend} ${connector.y2}, ${connector.x2} ${connector.y2}`;
  }

  const verticalBend = Math.max(40, Math.min(120, Math.abs(connector.y2 - connector.y1) * 0.45));
  return `M ${connector.x1} ${connector.y1} C ${connector.x1} ${connector.y1 + direction * verticalBend}, ${connector.x2} ${connector.y2 - direction * verticalBend}, ${connector.x2} ${connector.y2}`;
}

export function DictionaryMenu({ tableName, dictionary, dictionaryKey, anchorElement, anchorRect, portalContainer, onClose }) {
  const menuRef = useRef(null);
  const isPatientDictionary = tableName.toLowerCase() === 'patients';
  const selectedDictionary = isPatientDictionary && dictionaryKey
    ? { [dictionaryKey]: dictionary?.[dictionaryKey] ?? [] }
    : dictionary;
  const entries = dictionaryEntries(selectedDictionary);
  const menuWidth = 320;
  const getContainerRect = () => portalContainer?.getBoundingClientRect() ?? {
    left: 0,
    top: 0,
    width: window.innerWidth,
    height: window.innerHeight,
  };
  const containerRect = getContainerRect();
  const [position, setPosition] = useState(() => ({
    left: anchorRect.left - containerRect.left,
    top: anchorRect.bottom - containerRect.top + 8,
  }));
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [connector, setConnector] = useState(null);
  const [diagramScale, setDiagramScale] = useState(1);
  const dragState = useRef(null);

  // React Flow transforms do not move the popup automatically. Follow the
  // source element while preserving any offset created by dragging the menu.
  useEffect(() => {
    let animationFrame;
    const followAnchor = () => {
      const rect = anchorElement?.getBoundingClientRect();
      if (rect) {
        const hostRect = getContainerRect();
        const nextScale = anchorElement.offsetWidth > 0
          ? rect.width / anchorElement.offsetWidth
          : 1;
        setDiagramScale(nextScale);
        if (!dragState.current) {
          setPosition({
            left: rect.left - hostRect.left + dragOffset.x,
            top: rect.bottom - hostRect.top + 8 + dragOffset.y,
          });
        }

        const menuRect = menuRef.current?.getBoundingClientRect();
        if (menuRect) {
          const source = {
            x: rect.left + rect.width / 2 - hostRect.left,
            y: rect.top + rect.height / 2 - hostRect.top,
          };
          const menuCenter = {
            x: menuRect.left + menuRect.width / 2 - hostRect.left,
            y: menuRect.top + menuRect.height / 2 - hostRect.top,
          };
          const horizontal = Math.abs(menuCenter.x - source.x) >= Math.abs(menuCenter.y - source.y);
          const side = horizontal
            ? (menuCenter.x >= source.x ? 'right' : 'left')
            : (menuCenter.y >= source.y ? 'bottom' : 'top');
          const connectorBySide = {
            left: {
              side,
              x1: rect.left - hostRect.left,
              y1: source.y,
              x2: menuRect.right - hostRect.left,
              y2: menuCenter.y,
            },
            right: {
              side,
              x1: rect.right - hostRect.left,
              y1: source.y,
              x2: menuRect.left - hostRect.left,
              y2: menuCenter.y,
            },
            top: {
              side,
              x1: source.x,
              y1: rect.top - hostRect.top,
              x2: menuCenter.x,
              y2: menuRect.bottom - hostRect.top,
            },
            bottom: {
              side,
              x1: source.x,
              y1: rect.bottom - hostRect.top,
              x2: menuCenter.x,
              y2: menuRect.top - hostRect.top,
            },
          };
          setConnector(connectorBySide[side]);
        }
      }
      animationFrame = window.requestAnimationFrame(followAnchor);
    };

    animationFrame = window.requestAnimationFrame(followAnchor);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [anchorElement, dragOffset, menuWidth]);

  useEffect(() => {
    const handlePointerDown = (event) => {
      if (event.target.closest('.schema-dictionary-menu')) return;
      if (event.target.closest('.schema-column-link')) return;
      if (!menuRef.current?.contains(event.target)) onClose();
    };
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
    };

    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  const startDragging = (event) => {
    if (event.target.closest('button')) return;
    event.preventDefault();
    dragState.current = {
      startX: event.clientX,
      startY: event.clientY,
      left: position.left,
      top: position.top,
    };

    const handlePointerMove = (moveEvent) => {
      if (!dragState.current) return;
      const nextPosition = {
        left: dragState.current.left + (moveEvent.clientX - dragState.current.startX) / diagramScale,
        top: dragState.current.top + (moveEvent.clientY - dragState.current.startY) / diagramScale,
      };
      const anchor = anchorElement?.getBoundingClientRect();
      const hostRect = getContainerRect();
      if (anchor) {
        setDragOffset({
          x: nextPosition.left - (anchor.left - hostRect.left),
          y: nextPosition.top - (anchor.bottom - hostRect.top) - 8,
        });
      }
      setPosition(nextPosition);
    };
    const stopDragging = () => {
      dragState.current = null;
      document.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerup', stopDragging);
    };

    document.addEventListener('pointermove', handlePointerMove);
    document.addEventListener('pointerup', stopDragging);
  };

  return (
    <>
      {connector && <svg className="schema-dictionary-connector" aria-hidden="true">
        <path
          d={connectorPath(connector)}
        />
      </svg>}
      <div
        ref={menuRef}
        className="schema-dictionary-menu"
        style={{
          left: position.left,
          top: position.top,
          width: menuWidth,
          transform: `scale(${diagramScale})`,
          transformOrigin: 'top left',
        }}
        role="dialog"
        aria-label={`${tableName} dictionary`}
      >
        <div className="schema-dictionary-heading" onPointerDown={startDragging}>
          <span>{isPatientDictionary ? `${dictionaryKey} values` : `${tableName} code dictionary`}</span>
          <button type="button" onClick={onClose} aria-label="Close dictionary">×</button>
        </div>
        <div className="schema-dictionary-list">
          {entries.length === 0 && <div className="schema-dictionary-empty">No values available.</div>}
          {entries.map((entry, index) => (
            <div className="schema-dictionary-entry" key={`${entry.label}-${index}`}>
              <strong>{String(entry.label)}</strong>
              {entry.description !== null && <span>{String(entry.description)}</span>}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}