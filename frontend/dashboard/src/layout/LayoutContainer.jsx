import React, { useEffect, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import { GoldenLayout } from 'golden-layout';

// Import Golden Layout essential styles
import 'golden-layout/dist/css/goldenlayout-base.css';
import 'golden-layout/dist/css/themes/goldenlayout-dark-theme.css';

import Schema from '../schema/Schema'; 
import Patient from '../patient/Patient'
import Timeline from '../timeline/Timeline';


// Map components for factory registration
const components = {
  Schema,
  Patient,
  Timeline,
};

// Define Layout Config (uses componentType instead of componentName)
const layoutConfig = {
  root: {
    type : 'column',
    content: [
      {
        type: 'row',
        content: [
          {
            type: 'component',
            componentType: 'Schema',
            title: 'Schema Diagram',
          },
          {
            type: 'column',
            content: [
              {
                type: 'component',
                componentType: 'Patient',
                title: 'Patient Details',
                //height
                
              },
              {
                type: 'component',
                componentType: 'AI',
                title: 'AI Insights',
              },
            ],
          },
        ],
      },
      {
        type: 'component',
        componentType: 'Timeline',
        title: 'Event Timeline',
        height: 70,
      },
    ],
  },
};
export default function LayoutContainer() {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Golden Layout on the container element
    const layout = new GoldenLayout(containerRef.current);

    // Register factories that create React roots on GL container elements
    Object.entries(components).forEach(([name, Component]) => {
      layout.registerComponentFactoryFunction(name, (container) => {
        const root = ReactDOM.createRoot(container.element);
        root.render(<Component />);

        // Clean up React root when panel is destroyed to prevent memory leaks
        container.addEventListener('beforeComponentRelease', () => {
          root.unmount();
        });
      });
    });

    layout.registerComponentFactoryFunction('AI', (container) => {
      const root = ReactDOM.createRoot(container.element);
      root.render(<div style={{ padding: '10px', color: '#fff' }}>AI Insights Panel (Placeholder)</div>);

      container.addEventListener('beforeComponentRelease', () => {
        root.unmount();
      });
    });

    // Load initial configuration
    layout.loadLayout(layoutConfig);

    // Handle Window Resize
    const handleResize = () => {
      if (containerRef.current) {
        layout.setSize(
          containerRef.current.clientWidth,
          containerRef.current.clientHeight
        );
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup on component unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      layout.destroy();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100vw', height: '100vh', position: 'relative' }}
    />
  );
}