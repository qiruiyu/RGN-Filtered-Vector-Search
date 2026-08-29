# Region-Guided Navigation (RGN)

This repository is the C++17 reference implementation accompanying the RGN paper. It evaluates multi-attribute hybrid vector search over a static graph index. The submitted code is intentionally narrow: it contains the construction, persistence, query generation, exact ground-truth generation, RGN/ORGN navigation, filtered search, and evaluation paths used by the paper, without dynamic deletion, replacement, update, resize, warm-start, or alternative bottom-layer neighbor-control APIs.

## 1. Problem definition

Each entity `e` contains a dense vector `v(e)` in `R^d` and a finite set of structured attributes `A(e)`. Each query `q` contains a dense query vector `v(q)` and a finite set of required attributes `A(q)`.

An entity is qualified exactly when:

```text
A(q) is a subset of A(e)
```

The implementation never replaces this condition with scalar category equality. Attribute sets are sorted and deduplicated when they are created. The shared predicate is implemented by `rgn::satisfies` in `include/rgn/attributes.hpp` and is reused by ground-truth generation, representative selection, ORGN, RGN, bottom-layer search, and result validation.

The task is to return the `k` nearest qualified entities under squared Euclidean distance.

## 2. RGN overview

RGN uses a sparse upper graph layer as a shared guidance graph and the original bottom graph for filtered nearest-neighbor search.

The query path is:

1. Canonicalize the required attribute set `A(q)`.
2. Select the first representative whose complete entity attribute set contains `A(q)`.
3. If no representative exists, keep the original query vector and continue with the normal filtered graph search.
4. Starting from the representative, expand the shared guidance graph for the configured number of RGN hops without applying an attribute filter.
5. Apply the subset predicate only after the local guidance region has been collected.
6. Rank qualified local candidates by query distance and qualified local connectivity.
7. Fuse selected candidates into a transition vector.
8. Use the rewritten vector for the configured graph phase and restore the original vector for the filtered bottom phase in the paper's default two-phase mode.
9. At the bottom layer, perform the fixed one-hop qualified entry seeding used by the reported experiments, then continue with the configured HNSW-style or ACORN-style expansion depth.
10. Validate every returned entity again with `A(q) subseteq A(e)` before recording metrics.

The representative table is keyed by the complete canonical attribute set of an entity, not by an individual attribute. A query may therefore use a strict subset of the representative's attribute set.

## 3. Submission scope and immutability

The index is build-once and query-only after construction. This matches the experimental setting of the paper and keeps the submission auditable.

The following general-purpose hnswlib capabilities are deliberately not part of this repository:

- dynamic deletion and undelete;
- replacement of deleted slots;
- in-place vector update and connection repair;
- runtime index resizing;
- duplicate-label update semantics;
- configurable warm-start or variable-depth initial query seeding;
- configurable bottom-layer neighbor completion/control;
- generic filter functors and generic stop-condition search.

Labels must be unique. Inserting a duplicate label throws an exception. The maximum entity count is fixed when the index is created. To change vectors, attributes, graph capacity, or construction parameters, rebuild the index.

The bottom-layer entry seeding depth is fixed at one because that is part of the evaluated search path. It is not a user-facing warm-start feature: there is no `--initial-hops` option and no alternative seeding depth in the submitted executable.

## 4. Repository layout

### `app/`

The executable and experiment orchestration code.

- `app/rgn.cpp` is the minimal program entry point. It parses options, loads the dataset, builds the workload, runs the experiment, and writes the CSV.
- `app/config.hpp` defines all supported command-line options and their defaults.
- `app/config.cpp` parses, validates, prints, and documents command-line options.
- `app/dataset.hpp` defines the in-memory dataset representation.
- `app/dataset.cpp` loads and validates `RGNDATA` version 2 and legacy `WIKILBL` version 1 files.
- `app/workload.hpp` defines query and exact-result records.
- `app/workload.cpp` samples deterministic queries, chooses biased attribute sets, intersects attribute posting lists, computes exact top-k results, and computes filtered-query correlation.
- `app/experiment.hpp` defines benchmark rows and experiment results.
- `app/experiment.cpp` builds or validates an index, runs every `efSearch` value, verifies returned attributes, computes metrics, and writes the CSV.

### `include/rgn/`

Public types, the graph template, and template implementation modules.

- `attributes.hpp` implements canonical multi-attribute sets and the shared subset predicate.
- `attribute_index.hpp` declares the exact posting-list attribute index.
- `core.hpp` contains the small distance-space interface and binary I/O helpers retained from the upstream foundation.
- `distance_l2.hpp` provides the squared Euclidean distance implementation.
- `visited_list_pool.hpp` provides reusable visited arrays for graph traversals.
- `region_guidance.hpp` declares RGN transition candidates and fusion strategies.
- `graph_index.hpp` contains the upstream-derived static graph construction core and composes the RGN-specific template modules below.
- `index_persistence.tpp` saves and loads the versioned RGN index, variable-length attribute sets, representatives, and validation metadata.
- `representative_table.tpp` assigns the shared guidance layer and registers one representative per complete attribute set.
- `query_rewrite_mean.tpp` implements the mean-candidate rewriting baseline.
- `orgn_navigation.tpp` implements global ORGN candidate selection.
- `rgn_navigation.tpp` implements representative lookup, unfiltered h-hop guidance expansion, local subset filtering, connectivity calculation, and RGN rewriting.
- `hybrid_search.tpp` implements the two-phase upper/bottom graph search used by the experiments.
- `rgn.hpp` is the aggregate include for users of the library.

Template implementations use `.tpp` files because `RgnIndex<dist_t>` must be visible when instantiated. Splitting them from `graph_index.hpp` makes the upstream-derived graph core and the paper-specific components easier to audit.

### `src/`

Non-template implementations compiled into `rgn_core`.

- `attribute_index.cpp` builds posting lists and computes exact intersections for all query attributes.
- `region_guidance.cpp` ranks RGN candidates and fuses their vectors.

### `tests/`

Small deterministic regression tests.

- `multi_attribute_test.cpp` checks sorting/deduplication, subset semantics, qualified results, representative fallback, duplicate-label rejection, and index save/load equivalence.
- `end_to_end_test.cpp` writes an `RGNDATA` version 2 fixture, creates two-attribute queries, builds and reloads an index, writes a CSV, rejects sparse workloads with fewer than `k` qualified entities, rejects changed data, and rejects vector-dimension mismatches.

### Root files

- `CMakeLists.txt` builds the library, executable, and tests.
- `LICENSE` contains the Apache License 2.0.
- `THIRD_PARTY_NOTICES.md` records the hnswlib version, commit, derived files, and major modifications.

## 5. Requirements

- CMake 3.16 or newer;
- a C++17 compiler;
- sufficient memory for the input vectors and graph;
- no third-party runtime libraries.

The code has been verified with MinGW GNU C++ 8.1.0 on Windows. The source is standard C++17 and can also be built with recent GCC, Clang, or MSVC toolchains.

## 6. Build

### Windows with MinGW

Run these commands from the repository root:

```text
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable is `build\rgn.exe`.

### Linux or macOS

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable is normally `build/rgn`. Generated build directories and compiler outputs are excluded from the source submission.

## 7. Run the regression tests

```text
ctest --test-dir build --output-on-failure
```

Expected test names:

```text
rgn_multi_attribute
rgn_end_to_end
```

Both tests must pass before running a large experiment.

## 8. Dataset format

RGN uses a little-endian binary format named `RGNDATA`, version `2`.

### Header

Fields are stored consecutively with one-byte packing:

| Field | Type | Meaning |
|---|---|---|
| `magic` | `char[8]` | `RGNDATA` followed by `\0` |
| `version` | `uint32_t` | Must equal `2` |
| `entity_count` | `uint64_t` | Number of vectors/entities |
| `dimension` | `uint32_t` | Number of float coordinates per vector |
| `attribute_universe_size` | `uint32_t` | Declared attribute-universe cardinality for metadata |

Attribute identifiers are arbitrary signed 32-bit integers. `attribute_universe_size` is metadata and is not interpreted as a numeric upper bound on identifiers.

### Entity record

Each of the `entity_count` records stores:

| Field | Type | Count |
|---|---|---:|
| vector coordinates | `float` | `dimension` |
| attribute count | `uint32_t` | 1 |
| attribute identifiers | `int32_t` | `attribute_count` |

Attribute order and duplicate identifiers do not affect semantics. For example, `[7, 2, 7, 5]` is stored in memory as `{2, 5, 7}`.

### Legacy one-attribute input

The loader also accepts `WIKILBL` version `1`. Each legacy scalar label is converted to a one-element `AttributeSet`, so the same subset predicate is used by both legacy and multi-attribute experiments.

The loader rejects bad magic, unsupported versions, zero-sized vector data, truncated records, unsafe attribute counts, and trailing bytes.

## 9. Quick start

Print the supported options:

```text
build\rgn.exe --help
```

Validate a configuration without loading data:

```text
build\rgn.exe --dry-run --query-attributes 2 --ef-search 50,100,200
```

Build an index, save it, run queries, and write results:

```text
build\rgn.exe \
  --data dataset.bin \
  --index experiment.rgn.index \
  --output experiment.csv \
  --M 16 \
  --ef-construction 200 \
  --ef-search 50,100,200,400 \
  --k 10 \
  --queries 100 \
  --query-attributes 2 \
  --query-seed 42 \
  --index-seed 100 \
  --bias 0.75 \
  --guidance-layer 1 \
  --guidance-ratio -1 \
  --representatives 100 \
  --enable-rewrite 1 \
  --rewrite-method 2 \
  --rewrite-hops 2 \
  --fusion 0 \
  --two-phase 1 \
  --search-hops 2 \
  --acorn-mode 0 \
  --rebuild
```

PowerShell accepts the same arguments on one line. On Linux or macOS, replace the executable path and use the shell's normal line continuation syntax.

If the index exists and all stored construction settings, vectors, and attribute sets match the input dataset, it is loaded. Use `--rebuild` to ignore an existing index and construct a new one.

## 10. Command-line options

| Option | Default | Meaning |
|---|---:|---|
| `--data <path>` | required | Input `RGNDATA` v2 or `WIKILBL` v1 file |
| `--index <path>` | `rgn.index` | Saved RGN index path |
| `--output <path>` | `rgn_results.csv` | Result CSV path |
| `--k <n>` | `10` | Number of qualified nearest neighbors and Recall@k denominator |
| `--queries <n>` | `100` | Number of deterministic query entities |
| `--query-attributes <n>` | `2` | Required attributes per query; `0` keeps the selected complete set |
| `--M <n>` | `16` | Base graph degree |
| `--ef-construction <n>` | `200` | Construction candidate width |
| `--ef-search <a,b,...>` | `50,100,200,400` | Search widths, one CSV row per value |
| `--query-seed <n>` | `42` | Query permutation seed |
| `--index-seed <n>` | `100` | Random graph-level seed |
| `--bias <0..1>` | `1.0` | Rank-based distance bias for choosing the query attribute group |
| `--guidance-layer <n>` | `1` | Shared RGN guidance layer, at least 1 |
| `--guidance-ratio <-1 or 0..1>` | `-1` | Native random levels at `-1`, otherwise target guidance-layer ratio |
| `--representatives <n>` | `1` | Minimum guidance-layer nodes per complete attribute set |
| `--rewrite-candidates <n>` | `10` | Candidate limit for mean rewriting |
| `--enable-rewrite <0 or 1>` | `1` | Disable or enable query rewriting |
| `--rewrite-method <0..2>` | `2` | `0` mean baseline, `1` ORGN, `2` RGN |
| `--rewrite-hops <n>` | `2` | Unfiltered RGN guidance expansion depth |
| `--fusion <0..4>` | `0` | RGN candidate fusion strategy |
| `--two-phase <0..3>` | `1` | Rewritten/original query-vector placement |
| `--search-hops <n>` | `2` | Bottom-layer filtered expansion depth |
| `--acorn-mode <0 or 1>` | `0` | `0` ACORN-1, `1` uncompressed ACORN-gamma |
| `--acorn-gamma <n>` | `2` | Degree multiplier when `--acorn-mode 1`; must be at least 2 |
| `--rebuild` | off | Rebuild even if the index path exists |
| `--no-save-index` | off | Do not save a newly built index |
| `--dry-run` | off | Validate and print configuration without loading data |
| `--help` | off | Print the complete option list |

### Fusion strategies

| Value | Strategy |
|---:|---|
| `0` | Product of distance rank and qualified-connectivity rank |
| `1` | Distance rank only |
| `2` | Qualified-connectivity rank only |
| `3` | Deterministic seeded random selection |
| `4` | Equal average of candidates |

### Two-phase modes

The first component is the upper-layer vector and the second is the bottom-layer vector.

| Value | Upper layers | Bottom layer |
|---:|---|---|
| `0` | original | original |
| `1` | rewritten | original |
| `2` | original | rewritten |
| `3` | rewritten | rewritten |

Mode `1` is the default paper configuration: RGN guides navigation at the upper levels while the original query vector determines the final filtered neighbors.

## 11. Reproducing the four search variants

All variants can reuse the same compatible static index. Keep all construction parameters and the index path identical.

| Variant | Key query options |
|---|---|
| HNSW-style filtered traversal | `--enable-rewrite 0 --search-hops 1` |
| ACORN-1 filtered traversal | `--enable-rewrite 0 --search-hops 2` |
| RGN + HNSW-style traversal | `--enable-rewrite 1 --rewrite-method 2 --search-hops 1` |
| RGN + ACORN-1 traversal | `--enable-rewrite 1 --rewrite-method 2 --search-hops 2` |

Use separate output CSV paths for the four commands. Query count, query seed, query-attribute count, bias, `k`, and `efSearch` values must remain identical for a valid comparison.

## 12. Workload and exact ground truth

Queries are sampled reproducibly from dataset entity labels using `std::mt19937` and `--query-seed`.

For every query:

1. Complete attribute-set centroids are computed.
2. The requested bias selects an attribute group by query-to-centroid distance rank.
3. The first requested number of canonical attributes becomes `A(q)`.
4. Posting lists for every attribute in `A(q)` are intersected.
5. Exact squared L2 distances are computed only over entities satisfying the subset predicate.
6. The exact top-k labels become the ground truth.

Queries with fewer than `k` qualified entities are skipped. If the dataset cannot provide the requested query count under this condition, the program fails with a clear error instead of reporting an invalid Recall@k.

The filtered-query correlation reported by the program is:

```text
rho(q) = (dispersion - separation) / (dispersion + separation)
```

where `dispersion` is the average squared distance of qualified entities to their centroid and `separation` is the squared distance from the query vector to that centroid.

## 13. CSV output

Each `efSearch` value produces one row.

### Primary metrics

| Column | Meaning |
|---|---|
| `ef_search` | Search candidate width |
| `recall_at_k` | Fraction of exact qualified top-k labels returned |
| `qps` | Queries per second for the approximate search loop |
| `average_nfr` | Average unused fraction of the configured working-candidate capacity |
| `result_underfill_ratio` | Fraction of queries returning fewer than `k` approximate results |
| `average_distance_computations` | Average graph-search distance evaluations |
| `average_rewrite_distance_computations` | Average distance evaluations used by rewriting |
| `average_rho` | Mean valid filtered-query correlation |
| `negative_rho_queries` | Number of valid queries with negative correlation |

### Reproducibility metadata

The CSV also records query count, query-attribute count, entity count, dimension, `M`, `efConstruction`, bias, whether the index was built or loaded, construction seconds, rewrite settings, search hops, guidance settings, ACORN settings, and random seeds.

Timing depends on hardware, compiler, build type, and machine load. Recall, seeds, parameters, and workload identity should be used for algorithmic reproducibility; wall-clock equality is not expected across machines.

## 14. Index persistence and safety checks

Saved indexes use an RGN-specific magic value and version. The file stores graph layout metadata, level-0 vector records, upper links, canonical per-entity attribute sets, representative entries, and representative mean state.

On load, the implementation validates:

- magic and format version;
- file bounds and serialized block sizes;
- graph degree and ACORN metadata;
- vector record layout against the active L2 space;
- entity count and construction options;
- every stored vector against the supplied dataset;
- every stored canonical attribute set against the supplied dataset;
- representative identifiers, levels, and attributes;
- trailing bytes.

An incompatible or stale index is rejected with an instruction to rebuild. Indexes from the earlier scalar-category implementation are intentionally not accepted.

## 15. Provenance and license

The graph foundation is derived from `nmslib/hnswlib` version `0.8.0`, commit `3f34296`:

```text
https://github.com/nmslib/hnswlib
```

The derived foundation was renamed and substantially modified for canonical multi-attribute storage, subset-based qualification, complete-attribute-set representatives, shared guidance-layer assignment, ORGN and RGN query rewriting, unfiltered local guidance expansion followed by subset filtering, ACORN-style filtered traversal, two-phase query vectors, experiment metrics, and strict versioned persistence.

The complete Apache License 2.0 text is retained in `LICENSE`. `THIRD_PARTY_NOTICES.md` identifies the upstream project, version, commit, source URL, derived files, and modifications. Short provenance notices remain at the top of upstream-derived headers; other source comments were removed for the submission as requested.

When publishing results produced by this repository, cite the accompanying RGN paper and acknowledge hnswlib as the graph implementation foundation.

## 16. Reliability checklist for reviewers

Before trusting a reported result, verify that:

1. the code builds in Release mode;
2. both CTest tests pass;
3. all compared methods use the same dataset and compatible index;
4. query seed, query count, bias, query-attribute count, and `k` are identical;
5. every CSV contains the intended `efSearch` values;
6. the printed `average_rho` and negative-query count agree across methods;
7. index loading reports `loaded` rather than silently accepting mismatched data;
8. result-underfill is considered when interpreting Recall@k;
9. timing comparisons are performed on the same machine and build.

These checks correspond to validations implemented in the executable and regression tests rather than relying only on manual experiment bookkeeping.
