from dataclasses import dataclass, field, asdict
from copy import deepcopy, copy
import operator

from .module_log import module_logger

# Set up logging
logger = module_logger()

@dataclass
class UtrMappings:
    utr_ids: list[str] = field(default_factory=list)
    _index: int = 0

    # Set up the necessary magic methods.
    # This object should work like a dictionary with extra methods and structure.

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name != 'utr_ids' and not name.startswith('_'):
            self.utr_ids.append(name)
    
    def __len__(self):
        return len(self.utr_ids)

    # Next come some methods to behave a bit more like a dictionary

    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __getitem__(self,key):
        return getattr(self, key)
    
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
        for attr in self.__dict__.keys():
            if attr != 'utr_ids' and not attr.startswith('_'):
                as_dictionary[attr] = asdict(self[attr])
        return as_dictionary
    
    # Set up the main operations

    def filter(self, operation, by, amount, inplace=False):
        if operation in ('>','<','=','>=', '<='):
            result = self._operator_functions(operation, by, amount, inplace=inplace)
        return result

    # Private functions

    def _operator_functions(self, operation, by, amount, inplace=False):
        ops = { 
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "=": operator.eq,
            }
        if not inplace:
            new = self.copy(deep=True)
        for utr in self.utr_ids:
            new[utr][by] = [item for item in new[utr][by] if ops[operation](item,amount)]
        if not inplace:
            return new
        else:
            return None
    
@dataclass
class MappedReadInformation:
    """Information on the reads that mapped to a reference UTR region."""
    utr_name: str = ''
    read_start_positions: list[int] = field(default_factory=list)
    read_strands: list[str] = field(default_factory=list)
    read_sequences: list[str] = field(default_factory=list)
    read_barcodes: list[str] = field(default_factory=list)
    read_lengths: list[int] = field(default_factory=list)
    utr_sizes_from_reads: list[int] = field(default_factory=list)

    def __bool__(self):
        return bool(self.read_start_positions or self.read_strands or self.read_sequences \
            or self.read_barcodes or self.read_lengths or self.utr_sizes_from_reads)
    
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