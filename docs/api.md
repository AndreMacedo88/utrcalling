# Python API

The command line calls these functions, which notebooks and scripts can call
too. Count matrices are AnnData objects; tables are pandas DataFrames.

## The pipeline

```{eval-rst}
.. automodule:: utrcalling.core.core
   :members:
```

## Protocols

```{eval-rst}
.. automodule:: utrcalling.core.protocol
   :members: create, protocol_class, PROTOCOLS, Protocol, ProtocolThreePrime, ProtocolFivePrime
```

## Peak calling of count matrices

```{eval-rst}
.. automodule:: utrcalling.core.peak_calling_anndata
   :members: peak_call_anndata_parallel, peak_call_anndata, peak_table, score_bandwidths, best_setting, remove_peaks_below_count, pas_distance, pas_distance_table, estimate_best_bandwidth
```

## Peak callers, bandwidth rules and scores

```{eval-rst}
.. automodule:: utrcalling.core.peak_calling.settings
   :members:

.. automodule:: utrcalling.core.peak_calling.callers
   :members:

.. automodule:: utrcalling.core.peak_calling.kde
   :members: kde_caller, call_peaks_using_kernel_density_estimator

.. automodule:: utrcalling.core.peak_calling.bandwidth_rules
   :members:

.. automodule:: utrcalling.core.peak_calling.scores
   :members:

.. automodule:: utrcalling.core.registry
   :members:
```

## Count matrices

```{eval-rst}
.. automodule:: utrcalling.core.anndata_processing
   :members:

.. automodule:: utrcalling.core.provenance
   :members:
```

## Differential site usage

```{eval-rst}
.. automodule:: utrcalling.core.stats.site_usage
   :members: site_usage, SiteUsageResult

.. automodule:: utrcalling.core.stats.measures
   :members:

.. automodule:: utrcalling.core.stats.design
   :members: build, Design, Contrast, independent_columns

.. automodule:: utrcalling.core.stats.dirmult
   :members: Counts, Fit, fit, log_likelihood, proportions, moment_precision, variance_ratios

.. automodule:: utrcalling.core.stats.permutation
   :members: scheme, Scheme, permutation_test, PermutationResult

.. automodule:: utrcalling.core.stats.effects
   :members:

.. automodule:: utrcalling.core.stats.drimseq
   :members: check, run, DrimseqResult
```

## The genome and quality control

```{eval-rst}
.. automodule:: utrcalling.core.genome
   :members:

.. automodule:: utrcalling.core.qc
   :members:
```

## The UTR reference

```{eval-rst}
.. automodule:: utrcalling.core.utr_reference
   :members:
```
