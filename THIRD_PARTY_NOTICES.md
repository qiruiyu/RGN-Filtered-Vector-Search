# Third-party notices

The files `include/rgn/core.hpp`, `include/rgn/distance_l2.hpp`, `include/rgn/visited_list_pool.hpp`, and substantial portions of `include/rgn/graph_index.hpp` are derived from:

- Project: nmslib/hnswlib
- Version: v0.8.0
- Commit: `3f34296`
- Source: https://github.com/nmslib/hnswlib
- License: Apache License 2.0

The derived files have been renamed and modified. Major changes include multi-attribute entity storage, subset-based qualification, ACORN-style predicate-subgraph traversal, a representative table over complete attribute sets, ORGN/RGN query rewriting, two-stage navigation, additional metrics, an RGN-specific persistence format, and removal of general-purpose dynamic update, deletion, replacement, resize, warm-start, and alternative neighbor-control APIs from the submission build.

The repository retains the complete Apache License 2.0 text in `LICENSE`.
