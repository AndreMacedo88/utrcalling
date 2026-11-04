# import general tools
import argparse


def subset_experiments(input_path, annotations_keep, anno_types, out_paths):
    from utrcalling.core import anndata_processing as ap
    from utrcalling.core import helpers

    # Checked before loading the input, which can take long for large files.
    if len(out_paths) != len(annotations_keep):
        raise ValueError(
            f"Got {len(annotations_keep)} annotation files but {len(out_paths)} "
            "output paths: give one output path per annotation file."
        )
    if len(anno_types) != len(annotations_keep):
        raise ValueError(
            f"Got {len(annotations_keep)} annotation files but {len(anno_types)} "
            "annotation types: give one type per file."
        )

    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_list = ap.subset_adata(
        adata=adata, annotations_keep=annotations_keep, anno_types=anno_types
    )

    for adata_new, out_path in zip(adata_list, out_paths, strict=True):
        adata_new.write_h5ad(out_path)


def create_parser(subparsers):

    help = (
        "Subsets an experiment file (h5ad), given a list of annotations to match one "
        "column of the var or obs."
    )

    description = (
        f"{help} "
        "The annotation files' first column will be used to subset the h5ad experiment "
        "file. Make sure that this column values match the experiment file annotation "
        "index."
    )

    parser = subparsers.add_parser(
        name="subset-experiments",
        help=help,
        description=f"DESCRIPTION: {description}",
        epilog="""
    Example use:
    $ utrcalling subset-experiments adata.h5ad -a anno_1 anno_2 anno_3 -t var obs var
        -o adata_s1.h5ad adata_s2.h5ad adata_s3.h5ad
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_path", type=str, help="Path to input AnnData object file (h5ad format)."
    )
    parser.add_argument(
        "--annotations_keep",
        "-a",
        nargs="+",
        type=str,
        required=True,
        help="Paths to the annotation files that describe what to subset.",
    )
    parser.add_argument(
        "--anno_types",
        "-t",
        nargs="+",
        type=str,
        required=True,
        help="Type of the annotation to use for subsetting.",
    )
    parser.add_argument(
        "--out_paths",
        "-o",
        nargs="+",
        type=str,
        required=True,
        help="Paths to write the subsets.",
    )

    parser.set_defaults(func=subset_experiments)
