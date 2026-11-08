# Third-party dependencies

Metal builds fetch Apple's versioned `metal-cpp_macOS15_iOS18.zip` into
`third_party/metal-cpp/` (ignored, including the upstream license). CPU-only builds
never download or include metal-cpp. GoogleTest 1.15.2 and Google Benchmark 1.9.4
are fetched into each CMake build tree with archive SHA-256 verification.

Keep upstream licenses with downloaded sources. No third-party implementation
code is committed here. CMake's `FETCHCONTENT_SOURCE_DIR_METAL_CPP`,
`FETCHCONTENT_SOURCE_DIR_GOOGLETEST` and `FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK`
can point to previously downloaded **project-local** source trees for offline use.
