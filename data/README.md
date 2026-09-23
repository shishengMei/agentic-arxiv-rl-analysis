# Data provenance

`experiment.json` contains aggregate metrics extracted from the local checkout of
`Algorineko/AgenticArXiv-RL` recorded in its `source` object. It does not contain raw prompts,
model completions, model weights, or copied upstream source code.

Regenerate it with the `import` command documented in the project README. The importer records the
source Git commit and SHA-256 of every consumed summary and rejects disagreement between v5 summary
files and the experiment manifest.
