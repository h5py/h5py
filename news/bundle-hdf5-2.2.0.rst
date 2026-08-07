Building h5py
-------------

* The pre-built wheels now bundle HDF5 2.2.0, previously 2.0.0 (:pr:`2947`).
  This picks up the fixes for several CVEs affecting 2.0.0 — CVE-2025-44904,
  CVE-2025-2308, CVE-2025-2309, CVE-2026-26197 and CVE-2026-26199 (fixed
  upstream in HDF5 2.1.0), and CVE-2026-17572, CVE-2026-17573, CVE-2026-17574
  and CVE-2025-9274 (fixed in HDF5 2.2.0) — and relaxes a chunk validation
  check that made 2.0.0 reject spec-conforming files written by some
  third-party libraries (:issue:`2930`).
