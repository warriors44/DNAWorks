from pathlib import Path
from typing import Literal, Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import subprocess
import re
import math
from collections import Counter
from itertools import product
from scipy.stats import entropy
import numpy as np

MeltingTemp = Literal[55,60,65,68,70,72,75]
Tolerance = Literal[1,2,3,4,5]
Timelimit = Literal[0,1,30,60,120,180,300]
Seqtype = Literal["protein", "nucleotide"]
Threshold = Literal[0,5,10,15,20,25,30]
Organism = Literal["ecoli2", "ecoli", "c_elegans","d_melanogaster","h_sapiens","m_musculus",
                    "r_norvegicus","s_cerevisiae","x_laevis","p_pastoris"]

Enzyme = Literal["AflII", "BamHI", "EcoRI", "NcoI","NdeI","HindIII"]
re_dict = {"AflII": "CCTAGG",
           "BamHI": "GGATCC",
           "EcoRI": "GAATTC",
           "NcoI": "CCATGG",
           "NdeI": "CATATG",
           "HindIII": "AAGCTT"  
           }

DATA_DIR = Path(__file__).parent / "data"

@dataclass
class CodonEntry:
    codon: str
    amino_acid: str
    frequency: float

@dataclass
class CodonTable:
    organism: str
    entries: List[CodonEntry] = field(default_factory=list)

    _codon_to_aa: Dict[str, str] = field(default_factory=dict, repr=False, init=False)
    _aa_to_codons: Dict[str, List[CodonEntry]] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        self._build_indices()

    def _build_indices(self):
        self._codon_to_aa = {}
        self._aa_to_codons = {}
        for entry in self.entries:
            self._codon_to_aa[entry.codon] = entry.amino_acid
            self._aa_to_codons.setdefault(entry.amino_acid, []).append(entry)

    def codon_to_aa(self, codon: str) -> str:
        return self._codon_to_aa.get(codon, 'X')

    def aa_to_codons(self, amino_acid: str) -> List[str]:
        return [e.codon for e in self._aa_to_codons.get(amino_acid, [])]

    def aa_to_codon_entries(self, amino_acid: str) -> List[CodonEntry]:
        return self._aa_to_codons.get(amino_acid, [])

    def get_frequency(self, codon: str) -> float:
        for entry in self.entries:
            if entry.codon == codon:
                return entry.frequency
        return 0.0

    def codon_to_aa_dict(self) -> Dict[str, str]:
        return dict(self._codon_to_aa)

    def aa_to_codons_dict(self) -> Dict[str, List[str]]:
        return {aa: [e.codon for e in entries] for aa, entries in self._aa_to_codons.items()}

    @staticmethod
    def load(organism: str, data_dir: Optional[Path] = None) -> "CodonTable":
        if data_dir is None:
            data_dir = DATA_DIR
        tsv_path = data_dir / f"{organism}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Codon table not found: {tsv_path}")

        entries = []
        with open(tsv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if parts[0] == "amino_acid":
                    continue
                amino_acid, codon, freq = parts[0], parts[1], float(parts[2])
                entries.append(CodonEntry(codon=codon, amino_acid=amino_acid, frequency=freq))

        return CodonTable(organism=organism, entries=entries)

@dataclass
class Dnawork:
    seqtype: Seqtype
    melting_low: MeltingTemp
    melting_high: MeltingTemp
    tolerance: Tolerance
    timelimit: Timelimit
    frequency_threshold: Threshold
    organism: Organism
    sequence: str
    jobname: str
    logfile: Path
    enzyme: List[Enzyme|None] 
    oligo_length: Optional[int] = None
    strict: bool = True

    def generate_input(self) -> str:
        
        contents = f"""DNAWorks for {self.jobname}

logfile '{self.logfile}'
title '{self.jobname}'

timelimit {self.timelimit}
solutions 1
melting low {self.melting_low}
melting high {self.melting_high}
frequency threshold {self.frequency_threshold} {"strict" if self.strict else ""}
"""
        if self.oligo_length:
            contents += f"length low {int(self.oligo_length)}\n"
        
        patterns = "pattern\n"
        pattern_flag = False
        for enzyme in self.enzyme:
            if enzyme is not None and enzyme in re_dict:
                patterns += f"\t{enzyme}\t{re_dict[enzyme]}\n"
                pattern_flag = True
        
        if pattern_flag:
            patterns += "//\n"
            contents += patterns


        input_seq = ""
        for i, s in enumerate(self.sequence, 1):
            if i%60 == 0:
                input_seq += s + "\n"
            elif i%10 ==0:
                input_seq += s + " "
            else:
                input_seq += s
        input_seq += "\n"

        contents += f"""
codon {self.organism}
{self.seqtype}
{input_seq}
//
"""
        return contents


    def write_input_file(self,path: Path) -> None:
        with open(path, "w") as f:
            f.write(self.generate_input())


    @staticmethod
    def run_dnaworks(input_path: Path,executable:str="./DNAWorks/dnaworks") -> None:
        subprocess.run(f"{executable} {input_path}", shell=True)

    @staticmethod
    def parse_logfile(logfile: Path) -> None:
        with open(logfile, "r") as f:
            lines = f.readlines()


        flag = False
        sequence = ""
        cnt = 0
        for line in lines:
            match = re.search(r"The DNA sequence #\s+(\d+)\s+is:",line)
            if match:
                flag = True
                continue

            if flag:
                if line.startswith(" -"):
                    cnt += 1
                    if cnt == 2:break
                    continue
                match = re.search(r"(\d+)\s+([ATGC]+)", line)
                if match:
                    #print(match.group(1), match.group(2))
                    sequence += match.group(2).strip()
        
        return sequence

    @staticmethod
    def translate_dna_to_protein(dna_sequence: str, codon_table: Optional[CodonTable] = None) -> str:
        if codon_table is None:
            codon_table = CodonTable.load("ecoli2")

        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')
        
        print(f"Length of DNA sequence: {len(dna_sequence)}")

        protein_sequence = ""
        for i in range(0, len(dna_sequence) - len(dna_sequence) % 3, 3):
            codon = dna_sequence[i:i+3]
            amino_acid = codon_table.codon_to_aa(codon)
            if amino_acid == 'X':
                protein_sequence += 'X'
            elif amino_acid != '*':
                protein_sequence += amino_acid
            else:
                break
        
        return protein_sequence

    @staticmethod
    def find_consecutive_bases(dna_sequence: str, min_length: int = 6) -> List[tuple]:
        """
        Detect regions where the same base is repeated min_length or more times.
        
        Args:
            dna_sequence: DNA sequence
            min_length: Minimum consecutive length to detect (default: 6)
        
        Returns:
            List of consecutive regions. Each element is a (start_index, end_index, base) tuple.
        """
        consecutive_regions = []
        i = 0
        while i < len(dna_sequence):
            base = dna_sequence[i]
            start = i
            while i < len(dna_sequence) and dna_sequence[i] == base:
                i += 1
            length = i - start
            if length >= min_length:
                consecutive_regions.append((start, i, base))
        return consecutive_regions

    @staticmethod
    def get_alternative_codons(amino_acid: str, codon_table: Optional[CodonTable] = None) -> List[str]:
        """
        Return a list of all codons encoding the specified amino acid.
        
        Args:
            amino_acid: Amino acid (single letter)
            codon_table: CodonTable instance (default: ecoli2)
        
        Returns:
            List of codons
        """
        if codon_table is None:
            codon_table = CodonTable.load("ecoli2")
        return codon_table.aa_to_codons(amino_acid)

    @staticmethod
    def find_restriction_sites(dna_sequence: str, excluded_enzymes: List[str]) -> List[Tuple[int, int, str]]:
        """
        Return all occurrences of restriction enzyme recognition sites in the DNA sequence.
        
        Args:
            dna_sequence: DNA sequence
            excluded_enzymes: List of restriction enzyme names to check
        
        Returns:
            List of (start, end, enzyme_name) tuples
        """
        sites = []
        for enzyme in excluded_enzymes:
            if enzyme not in re_dict:
                continue
            recognition = re_dict[enzyme]
            start = 0
            while True:
                pos = dna_sequence.find(recognition, start)
                if pos == -1:
                    break
                sites.append((pos, pos + len(recognition), enzyme))
                start = pos + 1
        return sites

    @staticmethod
    def collect_problem_codons(dna_sequence: str, codons: List[str],
                                max_consecutive: int,
                                excluded_enzymes: List[str]) -> Tuple[set, List, List]:
        """
        Collect all codon indices involved in consecutive base runs and restriction enzyme recognition sites.
        
        Args:
            dna_sequence: DNA sequence
            codons: List of codons
            max_consecutive: Maximum allowed consecutive count
            excluded_enzymes: List of restriction enzyme names to check
        
        Returns:
            Tuple of (problem_codon_indices, consecutive_regions, restriction_sites)
        """
        num_codons = len(codons)
        problem_indices = set()
        
        # 1. Collect codons involved in consecutive base runs
        consecutive_regions = Dnawork.find_consecutive_bases(dna_sequence, max_consecutive + 1)
        for start, end, base in consecutive_regions:
            for i in range(num_codons):
                codon_start = i * 3
                codon_end = codon_start + 3
                if codon_start < end and codon_end > start:
                    problem_indices.add(i)
        
        # 2. Collect codons involved in restriction enzyme recognition sites
        restriction_sites = Dnawork.find_restriction_sites(dna_sequence, excluded_enzymes)
        for start, end, enzyme in restriction_sites:
            for i in range(num_codons):
                codon_start = i * 3
                codon_end = codon_start + 3
                if codon_start < end and codon_end > start:
                    problem_indices.add(i)
        
        return problem_indices, consecutive_regions, restriction_sites

    @staticmethod
    def expand_and_fragment(problem_indices: set, num_codons: int,offset: int = 1) -> List[List[int]]:
        """
        Expand problem codons by adding ±offset neighbors, then group consecutive indices into fragments.
        
        Args:
            problem_indices: Set of problematic codon indices
            num_codons: Total number of codons
        
        Returns:
            List of fragments. Each fragment is a sorted list of codon indices.
        """
        # Add ±offset neighbors
        expanded = set()
        for idx in problem_indices:
            expanded.add(idx)
            if idx > offset:
                expanded.add(idx - offset)
            if idx < num_codons - offset:
                expanded.add(idx + offset)
        
        if not expanded:
            return []
        
        # Sort and group consecutive indices
        sorted_indices = sorted(expanded)
        fragments = []
        current_fragment = [sorted_indices[0]]
        
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == current_fragment[-1] + 1:
                current_fragment.append(sorted_indices[i])
            else:
                fragments.append(current_fragment)
                current_fragment = [sorted_indices[i]]
        fragments.append(current_fragment)
        
        return fragments

    @staticmethod
    def encode_dna_to_codons(dna_sequence: str) -> List[str]:
        return [dna_sequence[i:i+3] for i in range(0, len(dna_sequence) - len(dna_sequence) % 3, 3)]

    @staticmethod
    def codon_damage(old_codon: str, new_codon: str, codon_table: "CodonTable",
                     epsilon: float = 1e-6) -> float:
        """
        Compute per-codon replacement damage based on frequency ratio.
        d_i = max(ln(f_old / f_new), 0)  (only penalize worsening)

        Returns 0.0 if the codon is unchanged.
        """
        if old_codon == new_codon:
            return 0.0
        f_old = max(codon_table.get_frequency(old_codon), epsilon)
        f_new = max(codon_table.get_frequency(new_codon), epsilon)
        return max(math.log(f_old / f_new), 0.0)

    @staticmethod
    def modify_unfavorble_codons(dna_sequence: str, max_consecutive: int = 5,
                                excluded_enzymes: Optional[List[str]] = None,
                                offset: int = 1,
                                codon_table: Optional[CodonTable] = None,
                                max_damage: float = 2.0) -> str:
        """
        Remove consecutive base runs (>max_consecutive) and restriction enzyme recognition
        sites simultaneously, selecting replacements that minimize translation damage.

        Candidate scoring (lexicographic order):
          (k, D_max, D_sum) where
            k     = number of changed codons
            D_max = max per-codon damage  d_i+ = max(ln(f_old/f_new), 0)
            D_sum = sum of per-codon damages

        Candidates with D_max > max_damage are filtered out.  If all candidates
        exceed the threshold, the best unfiltered candidate is adopted with a warning
        (fallback C).

        Args:
            dna_sequence: DNA sequence (length must be a multiple of 3)
            max_consecutive: Maximum allowed consecutive count (default: 5, i.e. avoid 6+)
            excluded_enzymes: List of restriction enzyme names to avoid (default: all)
            offset: Flanking codon expansion width (default: 1)
            codon_table: CodonTable instance (default: ecoli2)
            max_damage: D_max threshold for filtering candidates (default: 2.0)

        Returns:
            Corrected DNA sequence
        """
        if excluded_enzymes is None:
            excluded_enzymes = list(re_dict.keys())
        if codon_table is None:
            codon_table = CodonTable.load("ecoli2")

        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')

        codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence) - len(dna_sequence) % 3, 3)]
        num_codons = len(codons)
        amino_acids = [codon_table.codon_to_aa(c) for c in codons]
        aa_to_codons = codon_table.aa_to_codons_dict()

        # Step 1: Collect problem codons
        problem_indices, consecutive_regions, restriction_sites = \
            Dnawork.collect_problem_codons(dna_sequence, codons, max_consecutive, excluded_enzymes)

        print(f"\n{'='*60}")
        print(f"[Sequence optimization started] DNA length: {len(dna_sequence)} bp ({num_codons} codons)")
        print(f"  Organism: {codon_table.organism}")
        print(f"  Excluded restriction enzymes: {excluded_enzymes}")
        print(f"  Max allowed consecutive: {max_consecutive}")
        print(f"  Max damage threshold (D_max): {max_damage}")
        print(f"\n  Detected problems:")
        if consecutive_regions:
            print(f"    Consecutive runs: {len(consecutive_regions)} site(s)")
            for s, e, b in consecutive_regions:
                print(f"      pos {s}-{e}: {b} x {e-s}")
        else:
            print(f"    Consecutive runs: none")
        if restriction_sites:
            print(f"    Restriction sites: {len(restriction_sites)} site(s)")
            for s, e, enz in restriction_sites:
                print(f"      pos {s}-{e}: {enz} ({re_dict[enz]})")
        else:
            print(f"    Restriction sites: none")

        if not problem_indices:
            print(f"\n  No problems found. No correction needed.")
            print(f"{'='*60}")
            return dna_sequence

        print(f"\n  Number of problem codons: {len(problem_indices)}")

        # Step 2: Expand ±offset and fragment
        fragments = Dnawork.expand_and_fragment(problem_indices, num_codons, offset)

        print(f"  Number of fragments: {len(fragments)}")
        for i, frag in enumerate(fragments):
            print(f"    Fragment {i+1}: codon pos {frag[0]}-{frag[-1]} ({len(frag)} codons)")
        print(f"{'='*60}")

        # Step 3: Exhaustive search per fragment with damage scoring
        total_replacements = 0
        failed_fragments = []
        fallback_fragments = []

        for frag_idx, fragment in enumerate(fragments):
            codon_options = []
            for idx in fragment:
                aa = amino_acids[idx]
                if aa in aa_to_codons:
                    codon_options.append(aa_to_codons[aa])
                else:
                    codon_options.append([codons[idx]])

            total_combinations = 1
            for opts in codon_options:
                total_combinations *= len(opts)

            print(f"\n[Fragment {frag_idx+1}] codon pos {fragment[0]}-{fragment[-1]}")
            print(f"  Current codons: {[codons[i] for i in fragment]}")
            print(f"  Amino acids:    {[amino_acids[i] for i in fragment]}")
            print(f"  Combinations to search: {total_combinations}")

            check_start_codon = max(0, fragment[0] - 2)
            check_end_codon = min(num_codons - 1, fragment[-1] + 2)

            # Enumerate all valid candidates with (k, D_max, D_sum, combo)
            candidates = []

            for combo in product(*codon_options):
                temp_codons = codons.copy()
                for i, idx in enumerate(fragment):
                    temp_codons[idx] = combo[i]

                check_subsequence = ''.join(temp_codons[check_start_codon:check_end_codon + 1])

                if Dnawork.find_consecutive_bases(check_subsequence, max_consecutive + 1):
                    continue
                if Dnawork.find_restriction_sites(check_subsequence, excluded_enzymes):
                    continue

                damages = []
                k = 0
                for i, idx in enumerate(fragment):
                    if combo[i] != codons[idx]:
                        k += 1
                        d = Dnawork.codon_damage(codons[idx], combo[i], codon_table)
                        damages.append(d)

                d_max = max(damages) if damages else 0.0
                d_sum = sum(damages) if damages else 0.0
                candidates.append((k, d_max, d_sum, combo))

            if not candidates:
                failed_fragments.append((frag_idx + 1, fragment))
                print(f"  [FAILED] Searched all {total_combinations} combinations but could not resolve")
                continue

            # Filter by D_max threshold
            filtered = [c for c in candidates if c[1] <= max_damage]

            if filtered:
                best = min(filtered, key=lambda x: (x[0], x[1], x[2]))
                used_fallback = False
            else : 
                filtered = [c for c in candidates if c[1] <= max_damage+offset]
                if filtered:
                    best = min(filtered, key=lambda x: (x[0], x[1], x[2]))
                    used_fallback = True
                    fallback_fragments.append((frag_idx + 1, fragment, best))
                else:
                    best = min(candidates, key=lambda x: (x[0], x[1], x[2]))
                    used_fallback = True
                    fallback_fragments.append((frag_idx + 1, fragment, best))

            best_k, best_dmax, best_dsum, best_combo = best

            # Apply the best combination
            changes = []
            for i, idx in enumerate(fragment):
                if best_combo[i] != codons[idx]:
                    d = Dnawork.codon_damage(codons[idx], best_combo[i], codon_table)
                    changes.append((idx, codons[idx], best_combo[i], amino_acids[idx], d))
                    codons[idx] = best_combo[i]

            total_replacements += len(changes)
            if changes:
                status = "[Fix successful]" if not used_fallback else "[Fix applied - EXCEEDS D_max threshold]"
                print(f"  {status}")
                print(f"    Candidates: {len(candidates)} valid, {len(filtered)} passed D_max filter")
                print(f"    Score: k={best_k}, D_max={best_dmax:.4f}, D_sum={best_dsum:.4f}")
                for idx, old, new, aa, d in changes:
                    f_old = codon_table.get_frequency(old)
                    f_new = codon_table.get_frequency(new)
                    print(f"    Codon {idx}: {old}(f={f_old:.3f}) -> {new}(f={f_new:.3f})  AA={aa}  d={d:.4f}")
            else:
                print(f"  [No fix needed] Original codons are fine")

        # Final report
        final_sequence = ''.join(codons)
        final_consecutive = Dnawork.find_consecutive_bases(final_sequence, max_consecutive + 1)
        final_restriction = Dnawork.find_restriction_sites(final_sequence, excluded_enzymes)

        print(f"\n{'='*60}")
        print(f"[Sequence optimization completed]")
        print(f"  Total replaced codons: {total_replacements}")

        if final_consecutive:
            print(f"  Remaining consecutive runs: {len(final_consecutive)} site(s)")
            for s, e, b in final_consecutive:
                print(f"    pos {s}-{e}: {b} x {e-s}")
        else:
            print(f"  Remaining consecutive runs: none (OK)")

        if final_restriction:
            print(f"  Remaining restriction sites: {len(final_restriction)} site(s)")
            for s, e, enz in final_restriction:
                print(f"    pos {s}-{e}: {enz} ({re_dict[enz]})")
        else:
            print(f"  Remaining restriction sites: none (OK)")

        if fallback_fragments:
            print(f"\n  [Warning] {len(fallback_fragments)} fragment(s) exceeded D_max threshold (fallback applied):")
            for frag_num, frag, (bk, bd, bs, _) in fallback_fragments:
                print(f"    Fragment {frag_num}: codon {frag[0]}-{frag[-1]}  k={bk} D_max={bd:.4f} D_sum={bs:.4f}")

        if failed_fragments:
            print(f"\n  [Error] {len(failed_fragments)} fragment(s) could not be resolved:")
            for frag_num, frag in failed_fragments:
                print(f"    Fragment {frag_num}: codon pos {frag[0]}-{frag[-1]}")

        print(f"{'='*60}")

        return final_sequence


    def get_codon_table(self) -> CodonTable:
        return CodonTable.load(self.organism)

    def assert_sequence(self, dna_sequence: str) -> None:
        ct = self.get_codon_table()
        translated_sequence = self.translate_dna_to_protein(dna_sequence, codon_table=ct)
        assert len(translated_sequence) == len(dna_sequence)/3, "Sequence length mismatch"
        mismatch_count = 0
        for i in range(len(translated_sequence)):
            if self.sequence[i] != translated_sequence[i]:
                print(f"Sequence mismatch at position {i+1}: {self.sequence[i]} != {translated_sequence[i]}")

        if mismatch_count > 0:
            raise ValueError(f"Sequence mismatch at {mismatch_count} positions")

        print("Sequence assertion passed")

    @staticmethod
    def calculate_sequence_entropy(sequence: str) -> float:
        sequence = sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')
        counts = Counter(sequence)
        probabilities = np.array(list(counts.values())) / sum(counts.values())
        return entropy(probabilities, base=2)


def generate_input_file(fasta_path: Path, input_path: Path, logfile_path: Path) -> None:
    with open(fasta_path, "r") as f:
        sequence = f.readlines()[1].strip()

    sequence = sequence.replace("X", "TYG")
    sequence = "G"+sequence
    config = Dnawork(
        jobname=fasta_path.stem,
        seqtype="protein",
        melting_low=68,
        melting_high=68,
        tolerance=1,
        frequency_threshold=25,
        strict=True,
        timelimit=0,
        organism="ecoli2",
        sequence=sequence,
        logfile=logfile_path,
        enzyme=["AflII", "BamHI", "EcoRI", "NcoI","NdeI","HindIII"],
    )
    config.write_input_file(input_path)
    return config

def run_dnaworks(input_path: Path) -> None:
    excutable=Path(__file__).parent/"dnaworks"
    subprocess.run(f"{excutable} {input_path}", shell=True)

def parse_logfile(logfile: Path) -> None:
    with open(logfile, "r") as f:
        lines = f.readlines()

    flag = False
    sequence = ""
    cnt = 0
    for line in lines:
        match = re.search(r"The DNA sequence #\s+(\d+)\s+is:",line)
        if match:
            flag = True
            continue

        if flag:
            if line.startswith("-"):
                cnt += 1
                if cnt == 2:break
                continue
            match = re.search(r"(\d+)\s+([ATGC]+)", line)
            if match:
                #print(match.group(1), match.group(2))
                sequence += match.group(2).strip()
    print(sequence)
    return sequence
