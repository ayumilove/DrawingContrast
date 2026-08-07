# Product: Drawing Contrast Viewer

## Product Purpose
A specialized engineering tool for comparing two versions of mechanical engineering PDF drawings (blueprints). It detects pixel-level differences between an original drawing (B) and its derived version (A), and presents them in an interactive web viewer with a slider-based transition.

Beyond raw pixel diffing, the tool now provides three higher-level capabilities:

1. **View-Block Registration** — Drawings are segmented into individual views/blocks (three-view layouts, detail views) and each block is registered independently (feature matching + RANSAC + displacement clustering). Views that merely moved position are re-aligned block-by-block instead of flooding the diff with false positives, and their motion is reported as "moved view" annotations.
2. **VLM Semantic Interpretation** — Each clustered diff region is cropped, labeled (OLD/NEW side-by-side) and sent to a vision-language model, which returns a structured verdict: change type (dimension/geometry/annotation/moved-view/noise), a human-readable engineering description, and a noise flag. Pure rendering artifacts are filtered out automatically.
3. **Change List Sidebar** — The result viewer shows a collapsible panel beside the drawing: an AI-generated change summary, then one entry per detected change with a color-coded type badge and pixel statistics. Clicking an entry pans/zooms to the region with a pulsing highlight. VLM results stream in incrementally — the viewer opens instantly after the CV pass, and entries upgrade from "analyzing" to full descriptions as they complete.

## Users
- Mechanical engineers reviewing drawing revisions
- Quality assurance teams verifying blueprint changes
- Design managers comparing part specifications
- Used on desktop monitors (24-27"), typical office lighting, during focused review sessions

## Register
product

## Brand & Tone
- **Professional, precise, technical** — this is an engineering tool, not a consumer app
- Clean and functional — every element serves the comparison task
- Confidence-inspiring — engineers need to trust what they see
- Minimal distraction — the drawings are the focus, not the UI

## Anti-references
- Flashy SaaS dashboards with gradient cards
- Dark-mode-for-cool developer tools
- Overly playful consumer apps

## Core Interaction
1. Slider at 0-10%: Pure original drawing (B) displayed cleanly
2. Slider at 10-90%: Animated transition B→A with color-coded differences
   - Red overlay: elements added in A (not in B)
   - Blue overlay: elements removed from A (was in B)
3. Slider at 90-100%: Pure derived drawing (A) displayed cleanly
4. Pan and zoom: mouse wheel, drag, pinch gestures
5. Change list sidebar: click an entry to jump to the region; overlay shows red boxes for changes, yellow dashed boxes for moved views, with line-width compensation at high zoom levels

## Key Constraints
- Drawings are high-resolution (4763x3368px) — performance matters
- White background is essential for drawing legibility
- Red/blue diff colors must remain distinguishable and not clash with drawing content
- The tool must work entirely client-side after initial image load
