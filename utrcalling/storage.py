from dataclasses import dataclass, field, asdict
from copy import deepcopy, copy
import pickle
from collections import defaultdict
from statistics import mode

import joblib
from numpy import median

from .module_log import module_logger
from .helpers import get_number_cores


@dataclass
class UtrMappings:
    utr_ids: list[str] = field(default_factory=list)
    _index: int = 0
    read_start_positions: list[list[int]] = field(default_factory=list)
    read_strands: list[list[int]] = field(default_factory=list)
    # read_sequences: list[list[int]] = field(default_factory=list)
    read_barcodes: list[list[int]] = field(default_factory=list)
    read_umis: list[list[int]] = field(default_factory=list)
    read_lengths: list[list[int]] = field(default_factory=list)
    utr_sizes_from_reads: list[list[int]] = field(default_factory=list)

    def __postinit__(self):
        self.logger = module_logger()

    # Set up the necessary magic methods.
    # This object should work like a dictionary with extra methods and structure.

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name not in self.__dataclass_fields__:
            self.utr_ids.append(name)
            self.read_start_positions.append(value.read_start_positions)
            self.read_strands.append(value.read_strands)
            # self.read_sequences.append(value.read_sequences)
            self.read_barcodes.append(value.read_barcodes)
            self.read_umis.append(value.read_umis)
            self.read_lengths.append(value.read_lengths)
            self.utr_sizes_from_reads.append(value.utr_sizes_from_reads)

    def __len__(self):
        return len(self.utr_ids)

    # Next come some methods to behave a bit more like a dictionary

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, (int, slice)):
            return self.utr_ids[item]
        else:
            return [getattr(self, i) for i in item]

    def __delitem__(self, key):
        try:
            idx = self.utr_ids.index(key)
            del self.utr_ids[idx]
            del self.read_start_positions[idx]
            del self.read_strands[idx]
            # del self.read_sequences[idx]
            del self.read_barcodes[idx]
            del self.read_umis[idx]
            del self.read_lengths[idx]
            del self.utr_sizes_from_reads[idx]
            del self.__dict__[key]
        except (ValueError, AttributeError, KeyError):
            self.logger.error(f'Attribute {key} not found')

    # And set up iteration capabilities

    def __iter__(self):
        return deepcopy(self)

    def __next__(self):
        if self._index >= len(self.utr_ids):
            raise StopIteration
        else:
            self._index += 1
            return self[self.utr_ids[self._index - 1]]

    # Set up some supporting functions

    def head(self, n=1):
        return self.utr_ids[:n]

    def tail(self, n=1):
        return self.utr_ids[-n:]

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def todict(self):
        as_dictionary = dict()
        for attr in self.utr_ids:
            as_dictionary[attr] = asdict(self[attr])
        return as_dictionary

    def export(self, path):
        with open(path, 'wb') as file_:
            pickle.dump(self, file_)

    def filter(self, func, *args, parallel=False, **kwargs):
        """
        Filter features by user-defined function.

        Takes a function `func` that is called for each UTR in the
        `UtrMappings` object and returns only those for which the function returns
        True.

        *args and **kwargs are passed to `func`.

        Returns a new UtrMappings object.
        """
        if parallel:
            filtered = self.filter_parallel(func, *args, *kwargs)
        else:
            filtered = self.filter_sequential(func, *args, *kwargs)
        return filtered

    def filter_parallel(self, func, *args, **kwargs):
        filtered = UtrMappings()
        cores = get_number_cores()
        filtered_list = joblib.Parallel(n_jobs=cores)(joblib.delayed(
            self._filter_parallel)(read_info=read_info, func=func, args=args,
                                   kwargs=kwargs) for read_info in iter(self))

        for utr_mappings in filtered_list:
            filtered.append(utr_mappings)
        return filtered

    def filter_sequential(self, func, *args, **kwargs):
        filtered = UtrMappings()
        for read_info in iter(self):
            filtered_info = func(read_info, *args, **kwargs)
            if filtered_info:
                if isinstance(filtered_info, MappedReadInformation):
                    filtered[read_info.utr_name] = filtered_info
                else:
                    filtered[read_info.utr_name] = read_info
        return filtered

    def append(self, other):
        for info in other:
            self[info.utr_name] = info

    def merge(self, other, method='intercept'):
        def merge_read_info(x, other_):
            for key in x.__dict__.keys():
                if key != 'utr_name':
                    x[key].extend(other_[key])
            return x

        merged = UtrMappings()
        for utr in self.utr_ids:
            if utr in other.utr_ids:
                merged[utr] = self[utr].each(merge_read_info, other[utr])
            elif method=='union':
                merged[utr] = self[utr]
        return merged

    @staticmethod
    def _filter_parallel(read_info, func, args=None, kwargs=None):
        filtered = UtrMappings()
        filtered_info = func(read_info, *args, **kwargs)
        if filtered_info:
            if isinstance(filtered_info, MappedReadInformation):
                filtered[read_info.utr_name] = filtered_info
            else:
                filtered[read_info.utr_name] = read_info
        return filtered


@dataclass
class MappedReadInformation:
    """Information on the reads that mapped to a reference UTR region."""
    utr_name: str = ''
    read_start_positions: list[int] = field(default_factory=list)
    read_strands: list[str] = field(default_factory=list)
    # read_sequences: list[str] = field(default_factory=list)
    read_barcodes: list[str] = field(default_factory=list)
    read_umis: list[str] = field(default_factory=list)
    read_lengths: list[int] = field(default_factory=list)
    utr_sizes_from_reads: list[int] = field(default_factory=list)

    def __postinit__(self):
        self.logger = module_logger()

    def __bool__(self):
        return bool(self.read_start_positions or self.read_strands
                    or self.read_barcodes or self.read_umis or self.read_lengths or
                    self.utr_sizes_from_reads)  # or self.read_sequences

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __delitem__(self, key):
        try:
            del self.__dict__[key]
        except KeyError:
            self.logger.error(f'Attribute {key} not found')

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def todict(self):
        return asdict(self)

    def filter(self, func, *args, **kwargs):
        filtered = MappedReadInformation()
        filtered['utr_name'] = self.utr_name
        mask = func(self.__dict__, *args, **kwargs)
        for key, values in self.__dict__.items():
            if key == 'utr_name':
                continue
            filtered[key] = [b for a, b in zip(mask, values) if a]
        return filtered

    def each(self, func, *args, **kwargs):
        """
        Modify each feature with a user-defined function.

        Applies user-defined function *func* to each feature.  *func* must
        accept an MappedReadInformation as its first argument; *args and **kwargs 
        will be passed to *func*.

        *func* must return an MappedReadInformation object.
        """

        modified = func(self, *args, **kwargs)

        return modified

    def get_unique_umis(self):
        def count_duplicates(seq):
            dups = defaultdict(list)
            for i, item in enumerate(seq):
                dups[item].append(i)
            return dups

        filtered = MappedReadInformation()
        filtered['utr_name'] = self.utr_name
        read_start_positions_append = []
        read_strands_append = []
        read_barcodes_append = []
        read_umis_append = []
        read_lengths_append = []
        utr_sizes_from_reads_append = []

        duplicates_barcodes = count_duplicates(self.read_barcodes)
        for barcode, idx_barcode in duplicates_barcodes.items():
            read_umis = [self.read_umis[i] for i in idx_barcode]
            duplicates_umi = count_duplicates(read_umis)
            for umi, idx_umis in duplicates_umi.items():
                real_idx = [idx_barcode[idx] for idx in idx_umis]
                read_start_positions_append.append(
                    mode([self.read_start_positions[i] for i in real_idx]))
                read_strands_append.append(self.read_strands[real_idx[0]])
                read_barcodes_append.append(barcode)
                read_umis_append.append(umi)
                read_lengths_append.append(
                    mode([self.read_lengths[i] for i in real_idx]))
                utr_sizes_from_reads_append.append(
                    mode([self.utr_sizes_from_reads[i] for i in real_idx]))
        
        filtered['read_start_positions'] = read_start_positions_append
        filtered['read_strands'] = read_strands_append
        filtered['read_barcodes'] = read_barcodes_append
        filtered['read_umis'] = read_umis_append
        filtered['read_lengths'] = read_lengths_append
        filtered['utr_sizes_from_reads'] = utr_sizes_from_reads_append

        return filtered