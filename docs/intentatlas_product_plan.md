# IntentAtlas Product Plan

## Positioning

`IntentAtlas` is the product surface.

`TDRCluster` remains the underlying engine.

External description:

- IntentAtlas turns raw query logs into an interactive intent map.
- It is local-first, analysis-oriented, and designed for unlabeled query exploration.
- The default outcome is not just clustering output, but a readable taxonomy, a browseable hierarchy, and a single-file `agent_report.md` ready for further LLM discussion.

## Primary Users

### 1. Operations and support analysts

They want to upload a CSV or Excel file and quickly answer:

- What are users asking for most?
- What are the recurring pain points?
- What is new, noisy, or ambiguous?

### 2. LLM / NLP / search engineers

They care about:

- taxonomy quality
- noise isolation
- prompt and review logic
- reproducible outputs
- downstream exportability

### 3. Product and growth teams

They want:

- an interpretable demand map
- a way to inspect long-tail themes
- a basis for FAQ, routing, policy, and workflow updates

## Product Goals

### V1 goal

Make IntentAtlas usable end-to-end for a first-time user without asking them to understand the repository structure.

Success criteria:

- user can upload a file in the UI
- system runs a preflight check automatically
- system recommends preview vs full run
- user can launch a run from the UI
- user can inspect the result in an interactive report page
- user can export a single `agent_report.md`

### V2 goal

Turn the result viewer into a lightweight curation workspace.

Success criteria:

- user can rename categories
- user can edit definitions and boundary notes
- user can merge categories
- user can move misassigned queries
- user can preserve a curated layer without overwriting raw taxonomy truth

## Core Principles

- Leaf taxonomy is the source of truth.
- Hierarchy is a browsing layer, not a rewrite layer.
- The default exported artifact should be one file, not many.
- Raw artifacts still exist for advanced users, but they should be secondary.
- Preview is a first-class product feature, not a debug mode.

## V1 Scope

### Entry and setup

- Local Streamlit app
- Sidebar settings for LLM and embedding providers
- Config can be saved back to `config/config.yaml`

### Upload and preflight

- Drag-and-drop CSV / Excel upload
- Query-column selection
- Automatic preflight checks:
  - valid row count
  - empty row count
  - duplicate ratio
  - text-length profile
  - rough embedding / LLM token estimate
- Automatic preview recommendation

### Run workspace

- Preview mode
- Full mode
- Stage progress:
  - loading
  - taxonomy induction
  - saving
  - hierarchy
  - report generation
- Live run log
- Live counters for create / merge / split / inspect

### Result workspace

- Overview metrics
- Category distribution chart
- Category cards
- Hierarchy tree view
- Residual / ambiguous cluster view
- Download button for `agent_report.md`

### Export strategy

Default export:

- one single-file `agent_report.md`

Raw artifacts remain available but secondary:

- clustering csv
- summary json
- hierarchy json

## V2 Scope

### Curation layer

- raw taxonomy remains immutable
- curated taxonomy becomes an overlay
- changes are recorded as operations, not silent rewrites

### Editing operations

- rename category
- edit definition
- edit boundary notes
- merge categories
- move query
- mark residual cluster as keep / ignore / promote

### Additional product capabilities

- run-to-run comparison
- time-slice comparison
- taxonomy continuation over new batches
- curated hierarchy editing

## Information Architecture

### 1. Upload & Check

- upload file
- choose query column
- inspect data quality
- see cost estimate
- accept preview recommendation

### 2. Run

- choose preview or full
- start run
- monitor progress and log

### 3. Report

- overview
- categories
- hierarchy
- residuals
- agent report download

### 4. Roadmap / Curation

- communicate what becomes editable in V2

## Agent Report Design

The main export is a single markdown file with two jobs:

- readable enough for a human analyst
- structured enough for a chatbot to continue the analysis

Recommended sections:

- Run Overview
- Executive Summary
- Hierarchy View
- Full Leaf Taxonomy
- Residual And Ambiguous Areas
- Suggested Actions
- Suggested Follow-up Questions For Another LLM
- Curation Log
- Structured Appendix

## Implementation Roadmap

### Phase A

- create reusable analysis service
- create preflight service
- create report service
- create runtime config helpers

### Phase B

- build Streamlit product shell
- connect upload to run pipeline
- connect report view to saved artifacts

### Phase C

- tighten onboarding
- refine progress display
- polish report interactions
- improve single-file export quality

### Phase D

- add curated taxonomy overlay
- add editing operations
- add comparison and continuation workflows

## Current Build Strategy

This repository now targets:

- V1 implemented as a usable local-first UI
- V2 documented as the next structured expansion
