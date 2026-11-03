# Outputs and portable exports

Each input creates `<output>/<source_name>/` containing `temp/`, `semantic/`,
`logs/`, `logs/debug/`, `output/data.jsonl`, `output/benchmarks_jsonl/`, and
`output/_video_state/`. Omit source_name to use the input folder's name.
`logs/debug/*_METADATA.json` contains structured metadata and Gate C decisions.
Metadata is exported separately from instruction-task rows.

Status reports owned-process state, capability counts, data size/time,
video-state counts and the last segment event. Compare snapshots to assess
progress. Startup, completed-video scans, model loading, candidate filtering
and exhausted API budgets can each cause no writes; a single snapshot is not
a hang diagnosis. Status never starts or repairs a process.

Export accepts one source workspace or a run root with source children. It reads
combined task rows and structured metadata, not arbitrary historical text logs.
Older logs need an explicit converter when snapshots are absent. Stop the
producer for a consistent snapshot. Export detects source-content changes and
publishes `manifest.json` last. A failed export may leave a new partial directory
without a completion manifest; it is never overwritten or automatically deleted.

Dry run validates the plan without creating its destination. The destination
must be new and outside the source tree. Images must resolve inside the run or
explicit `--asset-root` directories; symlinks cannot authorize an escape.
Ambiguous relative references are rejected. Remote images are not downloaded.
Source records and images remain unchanged. Only images are copied; video paths
remain provenance references.

Package groups are `understanding.jsonl`, `edit.jsonl`, `unify.jsonl`,
`metadata.jsonl`, plus `generation.jsonl` if present. Empty groups are omitted.
Nested candidates, turns, frame metadata and evidence remain intact. Image path
fields become `images/<content_sha256>.<decoded_format>`; identical image bytes
share one copy. Producer `record_key` values must be unique per source. Missing
keys are derived from source location, row and content. `input_row` and existing
provenance stay intact; `_export_provenance` adds source file/line and hashes.

Images must exist, verify and fully decode. Hashes cover copied images and
JSONL files. After moving/copying a package, verify it:

```bash
python -c 'from vidueu.export import verify_package; print(verify_package("./moved_package"))'
```

Image paths use the package root, not the caller's cwd. Export is not semantic
review or missing-image regeneration. It never guesses timestamps, substitutes
another frame, or drops a record to make verification pass.

## Paper-schema release export

After creating a portable package, run the publication-facing second step:

```bash
./environments/dev/bin/vidueu paper-export \
  --source ./outputs/package_v1 \
  --destination ./outputs/paper_release_v1
```

This keeps `understanding.jsonl`, `edit.jsonl`, `unify.jsonl`, and
`metadata.jsonl` separate. Repaired metadata/edit files take priority over
their original counterparts; `und.jsonl` is an accepted input alias. The
export projects paper fields and explicitly retains the technical identifiers,
`frame_registry`, and `evidence_registry` needed to interpret images and
evidence. Arbitrary answer content, option labels, round order, and dialogue
order are preserved. Stored `sample_id`/`id` supplies a unique `record_key`;
the producer's grouping key is preserved in `record_lineage.jsonl`.

Images are independently copied and verified by SHA-256 and full decode.
References are relative to the package root. Local provenance/raw/audit
wrappers are excluded. A path-valued `segment.video_id` becomes an opaque
deterministic ID; an existing video ID is retained. No source videos are copied.

Strict mode refuses unresolved schema or reference issues before writing the
destination. For a local diagnostic candidate, `--allow-unresolved` retains
all rows and writes `reference_exceptions.jsonl`; `publish_ready` stays false
when issues remain. `complete: true` means materialization finished and does
not mean release acceptance. Unknown evidence is never guessed or silently
removed. Manifests use `vidueu-paper-hf-v2` and record exact source hashes.
Path-like strings quoted in original supervised text are retained and listed
in `path_text_review.jsonl`; they are not automatically rewritten. Operational
workspace/account paths are a separate acceptance check.

`--sample-limit 100` limits task rows only; it retains metadata dependencies
and may still copy many images. A successful portable export is not by itself
a successful paper-schema export. Inspect `publish_ready`, nullable-field
counts, and the exception ledger before publishing.
