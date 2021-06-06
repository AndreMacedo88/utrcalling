from dataclasses import dataclass, field, asdict
from copy import deepcopy, copy

from .module_log import module_logger

# Set up logging
logger = module_logger()

@dataclass
class UtrMappings:
    utr_ids: list[str] = field(default_factory=list)
    _index: int = 0
    read_start_positions: list[list] = field(default_factory=list)
    read_strands: list[list] = field(default_factory=list)
    read_sequences: list[list] = field(default_factory=list)
    read_barcodes: list[list] = field(default_factory=list)
    read_umis: list[list] = field(default_factory=list)
    read_lengths: list[list] = field(default_factory=list)
    utr_sizes_from_reads: list[list] = field(default_factory=list)

    # Set up the necessary magic methods.
    # This object should work like a dictionary with extra methods and structure.

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name not in self.__dataclass_fields__:
            self.utr_ids.append(name)
            self.read_start_positions.append(value.read_start_positions)
            self.read_strands.append(value.read_strands)
            self.read_sequences.append(value.read_sequences)
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
            del self.a[key]
        except AttributeError:
            logger.error(f'Attribute {key} not found')
    
    # And set up iteration capabilities

    def __iter__(self):
            return self
    
    def __next__(self):
        if self._index >= len(self.utr_ids):
            self._index = 0
            raise StopIteration
        else:
            self._index += 1
            return self.utr_ids[self._index-1]
    
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


@dataclass
class MappedReadInformation:
    """Information on the reads that mapped to a reference UTR region."""
    utr_name: str = ''
    read_start_positions: list[int] = field(default_factory=list)
    read_strands: list[str] = field(default_factory=list)
    read_sequences: list[str] = field(default_factory=list)
    read_barcodes: list[str] = field(default_factory=list)
    read_umis: list[str] = field(default_factory=list)
    read_lengths: list[int] = field(default_factory=list)
    utr_sizes_from_reads: list[int] = field(default_factory=list)

    def __bool__(self):
        return bool(self.read_start_positions or self.read_strands or self.read_sequences \
            or self.read_barcodes or self.read_umis or self.read_lengths or \
                self.utr_sizes_from_reads)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __getitem__(self,key):
        return getattr(self, key)
    
    def __delitem__(self, key):
        try:
            del self.a[key]
        except AttributeError:
            logger.error(f'Attribute {key} not found')

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)