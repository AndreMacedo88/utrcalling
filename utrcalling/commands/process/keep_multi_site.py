"""`utrcalling process keep-multi-site`: keep the UTRs with two or more sites."""

from __future__ import annotations


def keep_multi_site(input_path: str, out_path: str) -> None:
    """Keep only the UTRs with two or more sites.

    These are the alternative polyadenylation (APA) events of 3' UTRs, or the
    alternative transcription initiation (ATI) events of 5' UTRs.

    Args:
        input_path: Count matrix (.h5ad).
        out_path: Where to write the selected UTRs' sites (.h5ad).
    """
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_new = ap.select_alternative_transcription_events(adata=adata)
    adata_new.write_h5ad(out_path)
