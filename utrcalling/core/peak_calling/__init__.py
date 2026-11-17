"""Peak calling: collapsing the molecules' UTR sizes into consensus sites.

Each module holds one part, so a method can be read, replaced or added on its own:

- ``settings``: the names and spellings of the settings (kernels, bandwidths,
  methods, scores), checked as the command line parses them. It imports only
  the standard library.
- ``callers``: a peak caller turns one UTR's molecule sizes into the site of
  each molecule; ``get_caller`` builds one by its method's name.
- ``kde``: the kernel density estimate (KDE), utrcalling's method: a site at
  each peak of the density of the sizes.
- ``bandwidth_rules``: rules that set each UTR's bandwidth from its sizes
  (``scott``, ``silverman``).
- ``scores``: scores of peak-calling settings, by how well their sites fall
  where true sites are (``pas``, ``initiator``, ``known-sites``), with which
  `qc score-bandwidths` compares settings and ``-b optimize`` chooses one.

Peak callers, bandwidth rules and scores are extension points: a new one is a
function or a class, registered by name (see core.registry and
docs/extending.md). This package's __init__ imports nothing, so the command
line can read ``settings`` quickly.
"""
