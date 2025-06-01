# UI/UX Guidelines for ComfyUI Nodes

## Purpose
This document defines the user interface and user experience standards for all ComfyUI nodes in The-Machine, ensuring simplicity, clarity, accessibility, and a consistent teal-themed visual identity.

## Core Principles
- **Simplicity:** Each node should focus on a single responsibility. Avoid crowded or overly complex UIs.
- **Clarity:** Inputs, outputs, and controls must be clearly labeled. Use tooltips and help text where needed.
- **Consistency:** All nodes should use a unified color palette (teal theme), layout, and control style.
- **Accessibility:** Ensure all controls are keyboard-navigable and color contrast meets accessibility standards.
- **Feedback:** Provide clear, immediate feedback for user actions, progress, and errors.

## Teal Color Theme
- Primary: #20B2AA (Light Sea Green)
- Accent: #008080 (Teal)
- Background: #F0F8F8 (very light teal/white)
- Use teal for buttons, progress bars, highlights, and active states.
- Use accent teal for important actions or status indicators.

## Node Layout Best Practices
- **Header:** Node name, icon, and brief description.
- **Inputs:** Clearly labeled input fields, file pickers, or dropdowns.
- **Controls:** Teal-themed buttons, sliders, or toggles for configuration.
- **Progress:** Progress bars or spinners in teal for long-running tasks.
- **Outputs:** Output file lists, previews, or summaries with clear status indicators.
- **Errors:** Error messages in high-contrast (e.g., red on white/teal) with actionable suggestions.
- **Help:** Tooltip icons or expandable help sections for advanced options.

## Example Node Layout
```
+---------------------------------------------------+
| [Node Icon]  Node Name (e.g., "Separation Node")  |
|---------------------------------------------------|
| [Input]  Select files: [File Picker]              |
| [Input]  Model: [Dropdown]                        |
| [Control] [Run] [Reset]                           |
|---------------------------------------------------|
| [Progress Bar: teal]                              |
|---------------------------------------------------|
| [Output] Output files:                            |
|   - vocals.wav  [Preview] [Open]                  |
|   - instrumental.wav  [Preview] [Open]            |
|---------------------------------------------------|
| [Error] [Error message, if any]                   |
+---------------------------------------------------+
```

## User Flows
- **Batch Processing:**
  - User selects multiple files, configures options, runs node, and sees progress/output for each file.
- **Single-File Processing:**
  - User selects one file, configures options, runs node, and sees output for that file.
- **Error Handling:**
  - Errors are displayed inline with clear messages and suggestions. User can retry or skip.
- **Manifest/Output Review:**
  - User can view manifest summaries and output previews at each stage.

## Accessibility
- All controls must be keyboard-accessible.
- Use ARIA labels for screen readers.
- Ensure color contrast ratio of at least 4.5:1 for text and controls.

## Best Practices
- Minimize required user input; use sensible defaults.
- Group related controls and outputs visually.
- Use icons and color to reinforce meaning, but never as the sole indicator.
- Provide confirmation dialogs for destructive actions (e.g., reset, delete).
- Allow users to download manifest, logs, or outputs at any stage.

## Validation
- UI tests for all nodes to ensure layout, color, and accessibility standards are met.
- User feedback collection for continuous improvement. 