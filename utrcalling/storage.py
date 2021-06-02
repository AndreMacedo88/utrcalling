from dataclasses import dataclass, field

@dataclass
class UtrMappings:
    utr_ids: list[str] = field(default_factory=list)

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