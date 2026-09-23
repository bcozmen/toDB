Healthcare Data & ML Explorer
Goal

Build a React/TypeScript application for exploring Synthea-generated healthcare data, its vocabulary, and an ML system exposed through an API.

Main Features

Load and display the database schema

Tables and columns

Primary/foreign keys

Relationships

Automatic graph layout

Interactive schema visualization

Patient explorer

Demographics

Conditions

Medications

Encounters

Observations/labs

Procedures

Patient timeline/history

Vocabulary integration

Connect database codes to concepts and vocabulary information

ML integration

Send patient/cohort data to ML API

Display predictions

Compare predictions with actual outcomes

Calculate errors and model metrics

Explore prediction errors and open the corresponding patient

Dockable/resizable windows so users can arrange panels like a desktop application

Proposed UI
┌──────────────────────┬────────────────────────┐
│                      │                        │
│   Database Schema    │   Patient Explorer     │
│                      │                        │
│                      ├────────────────────────┤
│                      │   ML / Predictions     │
│                      │                        │
└──────────────────────┴────────────────────────┘


Panels should communicate with each other. For example, selecting a patient in ML results should open that patient in the Patient Explorer.

Tech Stack

React + TypeScript — frontend

Vite — build tooling

Golden Layout — dockable/resizable application windows

React Flow — interactive schema graph

ELK — automatic graph/table layout

Zustand — shared application state

TanStack Query — API requests, caching, loading/error states

ECharts or Recharts — ML charts and metrics

Tailwind CSS — styling

Zod — data validation

High-Level Architecture
React Application
│
├── Golden Layout
│   ├── Schema Viewer
│   ├── Patient Explorer
│   ├── Vocabulary
│   └── ML / Evaluation
│
├── Zustand
│   └── Shared application state
│
├── TanStack Query
│   └── API communication
│
└── Visualization
    ├── React Flow + ELK
    └── Charts


The React application should communicate with the database/vocabulary/ML through a backend API rather than accessing the database directly.

Development Order

React + TypeScript + Vite

Golden Layout shell

Schema loading and visualization

React Flow + ELK automatic layout

Patient explorer/timeline

Vocabulary integration

ML predictions

Model evaluation and error analysis

Cross-panel interactions