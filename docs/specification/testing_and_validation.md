# Testing & Validation Specification for ComfyUI Nodes

## Purpose
This document defines the testing and validation requirements for all ComfyUI nodes in The-Machine, ensuring correctness, privacy, robustness, and a high-quality user experience.

## Core Testing Areas
- **Unit Tests:** Test individual node logic, functions, and utilities in isolation.
- **Integration Tests:** Test end-to-end data flow and manifest updates across multiple nodes.
- **UI Tests:** Test user interface elements, accessibility, and user flows for each node.

## Privacy & PII Validation
- All tests must verify that no PII or original filenames/paths are present in any output, log, or manifest.
- Privacy/PII checks must be run after every processing step and on all outputs.
- Automated tests should simulate edge cases (e.g., files with embedded PII, unusual metadata).

## Manifest Correctness
- Validate that manifest entries are correct, complete, and updated at every stage.
- Test manifest schema compliance (required fields, types, and lineage).
- Ensure manifest is written to disk after every node and is readable by downstream nodes.

## Batch & Single-File Workflow Tests
- All nodes must be tested in both batch and single-file modes.
- Validate correct indexing, grouping, and manifest updates for both workflows.
- Simulate mixed input scenarios (e.g., partial batches, incomplete tuples).

## Error Handling
- Test all error paths: corrupt files, model/API failures, missing inputs, invalid configs.
- Ensure errors are logged only with anonymized, non-PII information.
- Validate that error sections in the manifest are populated and actionable.
- UI tests must verify that errors are displayed clearly and allow user recovery (retry, skip, etc.).

## User Feedback & Accessibility
- UI tests for all controls, progress bars, and feedback messages.
- Accessibility tests: keyboard navigation, ARIA labels, color contrast.
- User feedback collection mechanisms (e.g., feedback button, error reporting).

## Regression & Continuous Integration
- Automated test suite must run on every code change (CI integration).
- Regression tests for all critical paths and privacy logic.
- Test coverage reports must be generated and reviewed regularly.

## Example Test Cases
- Ingest a batch of files with mixed PII in filenames and metadata; verify all outputs are anonymized.
- Process a single file through the entire pipeline; validate manifest lineage and outputs.
- Simulate a model/API failure in the middle of a batch; verify error handling and manifest updates.
- UI test: user configures a node, runs processing, and reviews outputs and errors.
- Accessibility test: user navigates all controls via keyboard and screen reader.

## Validation Criteria
- All tests must pass before deployment.
- Privacy/PII validation is mandatory for every release.
- User feedback must be reviewed and incorporated into future improvements. 