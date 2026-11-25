# Comparing detectors

Choose one terrain layer and resolution mode for a batch. Select methods in
**Compare results** and run them sequentially. Inspect individual results and
unique candidates in the result selector. The table reports counts, mean lengths,
elapsed time and sampled process memory. The Validation workspace adds accuracy
and spatial diagnostics when you supply a reference.

**Algorithm Consensus** checks spatial position, direction, two-way coverage and
length ratio. It forms complete-link groups in descending pair-score order. Each
group contains at most one candidate per method and every pair in that group passes
the thresholds. The representative maximizes total agreement with group members.
Consensus retains all contributing geometries and IDs. Singleton candidates can
be inspected and exported by method. The consensus table lists the supporting
methods and processing families per object.

The default requirement is support from two processing families. A* and Segment
Tracing share a gradient/tangent family. Canny + Hough and RANSAC share Canny inputs.
The four filter banks share a ridge-linking family. LSD and EDLines each form
a family. Method families can still respond similarly to the same terrain and
noise. Consensus records a support count and leaves confidence unset.

Use identical reference data and tolerances when comparing raw, postprocessed and
consensus rows. Default profiles are starting configurations. Report resolution,
footprint, matching thresholds and reference completeness with each comparison.
