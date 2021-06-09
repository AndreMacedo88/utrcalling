from dataclasses import dataclass, field, asdict
from copy import deepcopy, copy

from .module_log import module_logger

# Set up logging
logger = module_logger()

@dataclass
class UtrMappings:
    utr_ids: list[str] = field(default_factory=list)
    _index: int = 0
    read_start_positions: list[list[int]] = field(default_factory=list)
    read_strands: list[list[int]] = field(default_factory=list)
    read_sequences: list[list[int]] = field(default_factory=list)
    read_barcodes: list[list[int]] = field(default_factory=list)
    read_umis: list[list[int]] = field(default_factory=list)
    read_lengths: list[list[int]] = field(default_factory=list)
    utr_sizes_from_reads: list[list[int]] = field(default_factory=list)

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
            idx = self.utr_ids.index(key)
            del self.utr_ids[idx]
            del self.read_start_positions[idx]
            del self.read_strands[idx]
            del self.read_sequences[idx]
            del self.read_barcodes[idx]
            del self.read_umis[idx]
            del self.read_lengths[idx]
            del self.utr_sizes_from_reads[idx]
            del self.__dict__[key]
        except (ValueError, AttributeError, KeyError):
            logger.error(f'Attribute {key} not found')
    
    # And set up iteration capabilities

    def __iter__(self):
            return deepcopy(self)
    
    def __next__(self):
        if self._index >= len(self.utr_ids):
            raise StopIteration
        else:
            self._index += 1
            return self[self.utr_ids[self._index-1]]
    
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

    def filter(self, func, *args, **kwargs):
        """
        Filter features by user-defined function.

        Takes a function *func* that is called for each UTR in the
        `UtrMappings` object and returns only those for which the function returns
        True.

        *args and **kwargs are passed directly to *func*.

        Returns a new UtrMappings object.
        """

        filtered = UtrMappings()
        for read_info in iter(self):
            filtered_info = func(read_info, *args, **kwargs)
            if filtered_info:
                if isinstance(filtered_info, MappedReadInformation):
                    filtered[read_info.utr_name] = filtered_info
                else:
                    filtered[read_info.utr_name] = read_info
        return filtered

    def merge(self, other):
        def merge_read_info(x, other_):
            for key in x.__dict__.keys():
                if key != 'utr_name':
                    x[key].extend(other_[key])
            return x

        merged = UtrMappings()
        for utr in self.utr_ids:
            if utr in other.utr_ids:
                merged[utr] = self[utr].each(merge_read_info, other[utr])
        return merged

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
            del self.__dict__[key]
        except KeyError:
            logger.error(f'Attribute {key} not found')

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)
        
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
