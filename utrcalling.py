import argparse
from difflib import get_close_matches
import re

from tests import test_case
import scripts

def run_tests(args=None):
    """
    Runs the test suite for the module.
    """
    if args:
        kwargs_new = parse_unknown_args(args, 0)
    else:
        kwargs_new={}
    test_case.run_all_tests(**kwargs_new)

def run(args=None):
    kwargs_new = parse_unknown_args(args, 2, keywords=scripts.run.keyword_args)
    scripts.run.main(args[0], args[1], **kwargs_new)

def call_peaks(args=None):
    kwargs_new = parse_unknown_args(args, 1, keywords=scripts.call_peaks.keyword_args)
    scripts.call_peaks.main(args[0], **kwargs_new)

def parse_unknown_args(args, n_args, keywords=None):
    # get only the keyword arguments
    list_to_parse = args[n_args:]
    # convert short hand arguments to the long form
    list_to_parse = [keywords[item] if re.match('-[a-zA-Z]', item) else item \
        for item in list_to_parse]
    
    list_to_parse_new = []
    possible_list = []
    for item in list_to_parse:
        if re.match('--', item):
            if possible_list:
                if len(possible_list) > 1:
                    list_to_parse_new.append(possible_list)
                else:
                    list_to_parse_new.append(possible_list[0])
                possible_list = []
            list_to_parse_new.append(item)
        else:
            possible_list.append(item)
    if len(possible_list) > 1:
        list_to_parse_new.append(possible_list)
    else:
        list_to_parse_new.append(possible_list[0])

    # create a dict to supply the function kwargs
    iterator = iter(list_to_parse_new)
    dict_parsed = {item.replace("--", ""): next(iterator) for item in iterator}
    return dict_parsed

def main(process, unknown_arguments, possibilities):
    """
    Checks the parameters input by the user and passes them along the pipeline to the correct function.
    Parameters
    ----------
    process : str
    unknown_arguments : list
    possibilities : list
    """
    
    close_matches = get_close_matches(process, possibilities, n=3, cutoff=0.6)
    if process not in close_matches:
        if close_matches:
            matches = '\n '.join(close_matches)
            print(f"Process {process} does not exist. Did you mean one of the following;\n {matches}")
        else:
            print(f"Process {process} does not exist.")
        return
    
    eval(f"{process}(args={unknown_arguments})")


if __name__ == "__main__":
    list_of_processes = ["run_tests", "run", "call_peaks"]
    # standard_run("/mnt/o_drive/andre/projects/3utr/youngNSC/Mus_musculus.GRCm39.104.gtf.gz")
    parser = argparse.ArgumentParser(description="""
    Utility script for easy access to most of the VEnCode scripts and
    tests. For the list of arguments to use, refer to this module's documentation or 
    check the help for the script you are trying to use, which is located in the 
    scripts folder.
    Examples on how to use in the command-line:
    $ python utrcalling.py run_tests --verbosity 1
    $ python utrcalling.py run mus_musculus.gtf mus_musculus.fasta 
        -a bowtie_align_wt.bam bowtie_align_ko.bam -f three_prime_utr -s experiment.csv 
        -p 2 -o results.h5ad
    """)

    parser.add_argument("process", type=str, help="The process/script to run.")

    args_, unknown_args = parser.parse_known_args()
    main(args_.process, unknown_args, list_of_processes)