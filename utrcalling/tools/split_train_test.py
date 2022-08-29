# import general tools
import argparse
# import internal tools
from utrcalling.core import anndata_processing as ap
from utrcalling.core import helpers


def split_train_test(input_path, seed, train_fraction, out_paths):
    adata = helpers.check_if_path_and_open(input_path, method="h5ad")
    adata_list = ap.split_train_test_data(adata=adata,
                                          train_fraction=train_fraction,
                                          seed=seed)

    for adata_new, out_path in zip(adata_list, out_paths):
        adata_new.write_h5ad(out_path)


def main():
    args_all = parser.parse_args()

    args = vars(args_all).copy()
    del args["func"]

    args_all.func(**args)


parser = argparse.ArgumentParser(
    description="""
Splits an UTR sizes experiment file (h5ad) into train/test samples, 
stratified by UMI number.
`train_fraction` is a float representing the fraction of samples set as training samples.
Setting `seed` allows for reproducible splits.
""",
    epilog="""
Example use:
$ split-train-test ./results/adata.h5ad -s 111 -o ./results/adata
""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input_path", type=str,
                    help="Path to input AnnData object file (h5ad format).")
parser.add_argument("--seed", "-s", type=int,
                    help="Integer used to set a random seed for reproducibility.",
                    default=None)
parser.add_argument("--train_fraction", "-t", type=int,
                    help="Fraction of data to assign as train data.",
                    default=0.75)
parser.add_argument("--out_path", "-o", type=str,
                    help="Path with the filename prefix to write the train and test data.")

# This allows for assigning different parsers to different functions:
parser.set_defaults(func=split_train_test)

if __name__ == "__main__":
    main()
