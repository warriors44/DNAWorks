"""
OligoEvaluator — Oligonucleotide evaluation module

Calculates Tm (melting temperature) using the appropriate method
based on sequence length, following the published methods of
Sigma-Aldrich OligoEvaluator.

Methods:
  - <=14 bases: Basic method (modified Marmur-Doty)
  - 15-120 bases: Nearest-Neighbor method (Breslauer et al. 1986)

References:
  1. Breslauer KJ et al. (1986) PNAS 83:3746-3750
  2. Freier SM et al. (1986) PNAS 83:9373-9377
  3. Marmur J, Doty P (1962) J Mol Biol 5:109-118
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


@dataclass
class DimerResult:
    """Self-dimer detection result"""
    delta_g: float                    # dG (kcal/mol)
    strength: str                     # "Strong" / "Moderate" / "Weak" / "Very Weak"
    num_pairs: int                    # Number of consecutive base pairs
    offset: int                       # Alignment offset
    paired_region: Tuple[int, int]    # (start, end) on copy 1
    involves_3prime: bool             # Whether 3' end is involved
    alignment: str                    # Text display


@dataclass
class HairpinResult:
    """Hairpin (secondary structure) detection result"""
    delta_g: float                    # Total dG (kcal/mol)
    delta_g_stem: float               # Stem dG
    delta_g_loop: float               # Loop penalty
    strength: str
    stem_length: int
    loop_size: int
    stem_5prime: str
    loop_seq: str
    stem_3prime: str
    position: Tuple[int, int, int, int]  # (stem5_start, loop_start, loop_end, stem3_end)
    alignment: str


class OligoEvaluator:
    """
    Oligonucleotide evaluation class (static methods only).

    Calculates Tm using the appropriate method based on sequence length,
    following the published methods of Sigma-Aldrich OligoEvaluator.

    Methods:
      - <=14 bases: Basic method (modified Marmur-Doty)
      - 15-120 bases: Nearest-Neighbor method (Breslauer et al. 1986)

    References:
      1. Breslauer KJ et al. (1986) PNAS 83:3746-3750
      2. Freier SM et al. (1986) PNAS 83:9373-9377
      3. Marmur J, Doty P (1962) J Mol Biol 5:109-118
    """

    # ===== Nearest-Neighbor thermodynamic parameters (Breslauer et al., 1986) =====
    # Key: 5'->3' dinucleotide
    # Value: (dH [kcal/mol], dS [cal/(mol*K)])
    # Symmetric pairs have identical values (e.g. AA/TT = TT/AA)
    NN_PARAMS: Dict[str, Tuple[float, float]] = {
        'AA': (-9.1, -24.0),   # AA/TT
        'AC': (-6.5, -17.3),   # AC/TG ≡ GT/CA
        'AG': (-7.8, -20.8),   # AG/TC ≡ CT/GA
        'AT': (-8.6, -23.9),   # AT/TA
        'CA': (-5.8, -12.9),   # CA/GT
        'CC': (-11.0, -26.6),  # CC/GG ≡ GG/CC
        'CG': (-11.9, -27.8),  # CG/GC
        'CT': (-7.8, -20.8),   # CT/GA
        'GA': (-5.6, -13.5),   # GA/CT
        'GC': (-11.1, -26.7),  # GC/CG
        'GG': (-11.0, -26.6),  # GG/CC
        'GT': (-6.5, -17.3),   # GT/CA
        'TA': (-6.0, -16.9),   # TA/AT
        'TC': (-5.6, -13.5),   # TC/AG ≡ GA/CT
        'TG': (-5.8, -12.9),   # TG/AC ≡ CA/GT
        'TT': (-9.1, -24.0),   # TT/AA ≡ AA/TT
    }

    # ===== Constants =====
    HELIX_INITIATION_A: float = -0.0108   # kcal/(K*mol) helix initiation correction
    GAS_CONSTANT_R: float     = 0.00199   # kcal/(K*mol) gas constant
    DEFAULT_OLIGO_CONC: float = 0.5e-6    # 0.5 uM oligo concentration
    DEFAULT_NA_CONC: float    = 0.05      # 50 mM Na+ concentration
    KELVIN_OFFSET: float      = 273.15

    COMPLEMENT_MAP: Dict[str, str] = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

    # ===== Internal utilities =====

    @staticmethod
    def _clean_sequence(sequence: str) -> str:
        """Clean sequence (remove whitespace, convert to uppercase)"""
        return sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')

    @staticmethod
    def _validate_dna(sequence: str) -> None:
        """Validate DNA sequence (detect non-A/T/G/C characters)"""
        invalid = set(sequence) - {'A', 'T', 'G', 'C'}
        if invalid:
            raise ValueError(f"Invalid bases detected: {invalid}")

    # ===== Tm calculation =====

    @staticmethod
    def calculate_tm_basic(sequence: str) -> float:
        """
        Tm calculation using the Basic method (modified Marmur-Doty).
        Applied to oligonucleotides of 14 bases or less.

        Formula: Tm = 2(A + T) + 4(C + G) - 7
          -7 is a solution correction factor (assuming non-membrane hybridization)

        Args:
            sequence: DNA sequence

        Returns:
            Tm (C)
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        if len(seq) == 0:
            raise ValueError("Empty sequence")

        a = seq.count('A')
        t = seq.count('T')
        c = seq.count('C')
        g = seq.count('G')

        tm = 2 * (a + t) + 4 * (c + g) - 7
        return float(tm)

    @staticmethod
    def calculate_tm_nn(sequence: str,
                        oligo_conc: Optional[float] = None,
                        na_conc: Optional[float] = None) -> float:
        """
        Tm calculation using the Nearest-Neighbor method.
        Applied to oligonucleotides of 15-120 bases.

        Formula:
          Tm = dH / (A + dS + R * ln(C/4)) - 273.15 + 16.6 * log10([Na+])

          dH  : Total enthalpy change [kcal/mol]
          A   : Helix initiation correction = -0.0108 kcal/(K*mol)
          dS  : Total entropy change [kcal/(K*mol)]
          R   : Gas constant = 0.00199 kcal/(K*mol)
          C   : Oligo concentration [M] (default 0.5 uM)
          [Na+]: Sodium ion concentration [M] (default 50 mM)

        Args:
            sequence:   DNA sequence
            oligo_conc: Oligo concentration (M). None defaults to 0.5 uM
            na_conc:    Na+ concentration (M). None defaults to 50 mM

        Returns:
            Tm (C), rounded to 1 decimal place
        """
        if oligo_conc is None:
            oligo_conc = OligoEvaluator.DEFAULT_OLIGO_CONC
        if na_conc is None:
            na_conc = OligoEvaluator.DEFAULT_NA_CONC

        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)

        if len(seq) < 2:
            raise ValueError("Nearest-Neighbor method requires a sequence of 2 or more bases")

        # Sum dH (kcal/mol) and dS (cal/(mol*K))
        delta_h = 0.0
        delta_s_cal = 0.0

        for i in range(len(seq) - 1):
            dn = seq[i:i + 2]
            if dn not in OligoEvaluator.NN_PARAMS:
                raise ValueError(f"Unknown dinucleotide: {dn}")
            dh, ds = OligoEvaluator.NN_PARAMS[dn]
            delta_h += dh
            delta_s_cal += ds

        # Convert dS to kcal/(mol*K)
        delta_s = delta_s_cal / 1000.0

        A = OligoEvaluator.HELIX_INITIATION_A
        R = OligoEvaluator.GAS_CONSTANT_R

        denominator = A + delta_s + R * math.log(oligo_conc / 4.0)
        if denominator == 0.0:
            raise ValueError("Calculation error: denominator is zero")

        tm = delta_h / denominator - OligoEvaluator.KELVIN_OFFSET \
             + 16.6 * math.log10(na_conc)

        return round(tm, 1)

    @staticmethod
    def calculate_tm(sequence: str,
                     oligo_conc: Optional[float] = None,
                     na_conc: Optional[float] = None) -> float:
        """
        Calculate Tm using the appropriate method based on sequence length.

        - <=14 bases -> Basic method (modified Marmur-Doty)
        - 15-120 bases -> Nearest-Neighbor method (Breslauer 1986)
        - >120 bases -> Nearest-Neighbor method with warning

        Args:
            sequence:   DNA sequence
            oligo_conc: Oligo concentration (M). Used for NN method. Default 0.5 uM
            na_conc:    Na+ concentration (M). Used for NN method. Default 50 mM

        Returns:
            Tm (C)
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)

        if len(seq) == 0:
            raise ValueError("Empty sequence")

        if len(seq) <= 14:
            return OligoEvaluator.calculate_tm_basic(seq)

        if len(seq) > 120:
            import warnings
            warnings.warn(
                f"Sequence length {len(seq)} exceeds the recommended range (15-120 bases) "
                "for the Nearest-Neighbor method. Accuracy is not guaranteed.",
                stacklevel=2,
            )

        return OligoEvaluator.calculate_tm_nn(seq, oligo_conc, na_conc)

    # ===== Utilities =====

    @staticmethod
    def gc_content(sequence: str) -> float:
        """
        Return GC content (%).

        Args:
            sequence: DNA sequence

        Returns:
            GC content (0.0-100.0 %)
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        if len(seq) == 0:
            return 0.0
        return round((seq.count('G') + seq.count('C')) / len(seq) * 100, 1)

    @staticmethod
    def complement(sequence: str) -> str:
        """Return the complement sequence (each base replaced with its complement, 5'->3' preserved)."""
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        return ''.join(OligoEvaluator.COMPLEMENT_MAP[b] for b in seq)

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Return the reverse complement sequence."""
        return OligoEvaluator.complement(sequence)[::-1]

    @staticmethod
    def molecular_weight(sequence: str) -> float:
        """
        Return the approximate molecular weight (g/mol) of a single-stranded DNA oligo.

        Molecular weight of each deoxyribonucleoside monophosphate:
          dAMP=331.2, dTMP=322.2, dCMP=307.2, dGMP=347.2

        MW = sum(NTP) - (N-1)*18.02 + 17.01

        Args:
            sequence: DNA sequence

        Returns:
            Molecular weight (g/mol)
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)

        ntp_mw = {'A': 331.2, 'T': 322.2, 'C': 307.2, 'G': 347.2}

        mw = sum(ntp_mw[b] for b in seq)
        if len(seq) > 1:
            mw -= (len(seq) - 1) * 18.02
        mw += 17.01  # 5' terminal OH

        return round(mw, 1)

    @staticmethod
    def nn_thermodynamics(sequence: str) -> Tuple[float, float]:
        """
        Return the sum of Nearest-Neighbor parameters.

        Args:
            sequence: DNA sequence

        Returns:
            Tuple of (dH [kcal/mol], dS [cal/(mol*K)])
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)

        delta_h = 0.0
        delta_s = 0.0
        for i in range(len(seq) - 1):
            dn = seq[i:i + 2]
            dh, ds = OligoEvaluator.NN_PARAMS[dn]
            delta_h += dh
            delta_s += ds

        return (round(delta_h, 1), round(delta_s, 1))

    # ===== Summary =====

    @staticmethod
    def summary(sequence: str,
                oligo_conc: Optional[float] = None,
                na_conc: Optional[float] = None) -> Dict:
        """
        Return a comprehensive evaluation of the oligonucleotide as a dictionary.

        Args:
            sequence:   DNA sequence
            oligo_conc: Oligo concentration (M)
            na_conc:    Na+ concentration (M)

        Returns:
            dict with keys: sequence, length, gc_content, tm, tm_method,
                            molecular_weight, complement, reverse_complement,
                            delta_h, delta_s
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)

        length = len(seq)
        gc = OligoEvaluator.gc_content(seq)
        tm = OligoEvaluator.calculate_tm(seq, oligo_conc, na_conc)
        mw = OligoEvaluator.molecular_weight(seq)
        method = "Basic (Marmur-Doty)" if length <= 14 else "Nearest-Neighbor (Breslauer 1986)"

        result = {
            'sequence': seq,
            'length': length,
            'gc_content': gc,
            'tm': tm,
            'tm_method': method,
            'molecular_weight': mw,
            'complement': OligoEvaluator.complement(seq),
            'reverse_complement': OligoEvaluator.reverse_complement(seq),
        }

        if length >= 2:
            dh, ds = OligoEvaluator.nn_thermodynamics(seq)
            result['delta_h'] = dh
            result['delta_s'] = ds

        return result

    @staticmethod
    def print_summary(sequence: str,
                      oligo_conc: Optional[float] = None,
                      na_conc: Optional[float] = None) -> None:
        """Print the comprehensive oligonucleotide evaluation to console."""
        info = OligoEvaluator.summary(sequence, oligo_conc, na_conc)

        print(f"\n{'='*60}")
        print(f"  OligoEvaluator Summary")
        print(f"{'='*60}")
        print(f"  Sequence:      5'-{info['sequence']}-3'")
        print(f"  Length:        {info['length']} bases")
        print(f"  GC content:   {info['gc_content']}%")
        print(f"  Tm:           {info['tm']} C  ({info['tm_method']})")
        print(f"  MW:           {info['molecular_weight']} g/mol")
        if 'delta_h' in info:
            print(f"  dH:           {info['delta_h']} kcal/mol")
            print(f"  dS:           {info['delta_s']} cal/(mol*K)")
        print(f"  Complement:   3'-{info['complement']}-5'")
        print(f"  Rev. comp.:   5'-{info['reverse_complement']}-3'")
        print(f"{'='*60}")

    # =================================================================
    #  Self-Dimer / Hairpin Analysis
    #  dG(T) = dH - T*dS  (calculated from NN parameters)
    #  Strength classification follows thresholds commonly used by IDT / Primer3
    # =================================================================

    # Hairpin loop free energy penalty (kcal/mol, 37 C)
    # Approximated from SantaLucia (1998) / Zuker (2003)
    _LOOP_PENALTY_37: Dict[int, float] = {
        3: 5.4, 4: 4.5, 5: 5.0, 6: 5.0,
        7: 5.2, 8: 5.2, 9: 5.3, 10: 5.3,
    }

    # ----- Internal helpers -----

    @staticmethod
    def _calc_nn_delta_g(paired_seq: str, temperature: float = 37.0) -> float:
        """
        Calculate dG of the base-paired region from NN parameters.

        Args:
            paired_seq:  Sense strand sequence in 5'->3' direction
            temperature: Temperature (C)

        Returns:
            dG (kcal/mol)
        """
        T_K = temperature + 273.15
        delta_g = 0.0
        for i in range(len(paired_seq) - 1):
            dn = paired_seq[i:i + 2]
            dh, ds_cal = OligoEvaluator.NN_PARAMS.get(dn, (0.0, 0.0))
            delta_g += dh - T_K * ds_cal / 1000.0
        return round(delta_g, 2)

    @staticmethod
    def _hairpin_loop_penalty(loop_size: int) -> float:
        """
        Hairpin loop dG penalty (kcal/mol, 37 C).
        Positive value (destabilizing factor).

        Uses Jacobson-Stockmayer extrapolation for loop_size > 10.
        """
        if loop_size <= 10:
            return OligoEvaluator._LOOP_PENALTY_37.get(loop_size, 5.3)
        R_cal = 1.987   # cal/(mol·K)
        T_K = 310.15     # 37 C
        return round(5.3 + 1.75 * R_cal * T_K * math.log(loop_size / 10.0) / 1000.0, 2)

    @staticmethod
    def _classify_dimer_strength(delta_g: float) -> str:
        """Self-dimer strength classification (dG thresholds follow IDT/Primer3)"""
        if delta_g <= -6.0:
            return "Strong"
        if delta_g <= -4.0:
            return "Moderate"
        if delta_g <= -2.0:
            return "Weak"
        return "Very Weak"

    @staticmethod
    def _classify_hairpin_strength(delta_g: float) -> str:
        """Hairpin strength classification"""
        if delta_g <= -3.0:
            return "Strong"
        if delta_g <= -2.0:
            return "Moderate"
        if delta_g <= -1.0:
            return "Weak"
        return "Very Weak"

    # ----- Alignment display -----

    @staticmethod
    def _format_dimer_alignment(seq: str, offset: int,
                                pair_start: int, pair_end: int) -> str:
        """Generate text alignment for self-dimer."""
        n = len(seq)
        comp = OligoEvaluator.COMPLEMENT_MAP
        rev = seq[::-1]

        line1 = "5' " + seq + " 3'"

        indicators = [' '] * (3 + n)
        for i in range(pair_start, pair_end):
            j = n - 1 - i + offset
            if 0 <= j < n and comp.get(seq[i]) == seq[j]:
                indicators[3 + i] = '|'
        line2 = ''.join(indicators)

        padding = ' ' * offset
        line3 = padding + "3' " + rev + " 5'"

        return line1 + '\n' + line2 + '\n' + line3

    @staticmethod
    def _format_hairpin_alignment(seq: str, s5s: int, ls: int,
                                  le: int, s3e: int) -> str:
        """Text representation of hairpin structure (dot-bracket + pair display)."""
        n = len(seq)
        stem_len = ls - s5s
        stem5 = seq[s5s:ls]
        loop = seq[ls:le]
        stem3 = seq[le:s3e]
        stem3_rc = OligoEvaluator.reverse_complement(stem3)

        db = list('.' * n)
        for k in range(stem_len):
            db[s5s + k] = '('
            db[s3e - 1 - k] = ')'

        lines = [
            f"5'-{seq}-3'",
            f"   {''.join(db)}",
            f"   Stem 5': {stem5} (pos {s5s}-{ls - 1})",
            f"   Loop:    {loop} ({le - ls} nt, pos {ls}-{le - 1})",
            f"   Stem 3': {stem3} (pos {le}-{s3e - 1})",
            f"   Pairing: 5'-{stem5}-3'",
            f"            {'|' * stem_len}",
            f"            3'-{stem3_rc}-5'",
        ]
        return '\n'.join(lines)

    # =================================================================
    #  find_self_dimers
    # =================================================================

    @staticmethod
    def find_self_dimers(sequence: str,
                         min_pairs: int = 3,
                         temperature: float = 37.0,
                         max_results: int = 10) -> List[DimerResult]:
        """
        Detect self-dimer (self-dimerization).

        When two copies of the same oligo align antiparallel, exhaustively
        search for continuous Watson-Crick base-pairing regions and compute dG.

        Algorithm:
          1. Slide two copies with offset k (0 <= k < n)
          2. Position i of copy 1 faces position (n-1-i+k) of copy 2
          3. Detect continuous complementary regions >= min_pairs
          4. Compute dG from NN parameters and classify strength

        Args:
            sequence:    DNA sequence
            min_pairs:   Minimum consecutive base pairs to detect (default 3)
            temperature: Temperature (C, default 37.0)
            max_results: Maximum number of results to return

        Returns:
            List of DimerResult (sorted by dG ascending = most stable first)
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        n = len(seq)
        comp = OligoEvaluator.COMPLEMENT_MAP

        results: List[DimerResult] = []

        for offset in range(0, n - min_pairs + 1):
            paired_positions: List[int] = []
            for i in range(offset, n):
                j = n - 1 - i + offset
                if 0 <= j < n and comp.get(seq[i]) == seq[j]:
                    paired_positions.append(i)

            if not paired_positions:
                continue

            # Detect runs of consecutive positions
            runs: List[Tuple[int, int]] = []
            run_s = paired_positions[0]
            prev = run_s
            for pos in paired_positions[1:]:
                if pos == prev + 1:
                    prev = pos
                else:
                    if prev - run_s + 1 >= min_pairs:
                        runs.append((run_s, prev + 1))
                    run_s = pos
                    prev = pos
            if prev - run_s + 1 >= min_pairs:
                runs.append((run_s, prev + 1))

            for start, end in runs:
                num_pairs = end - start
                paired_seq = seq[start:end]
                delta_g = OligoEvaluator._calc_nn_delta_g(paired_seq, temperature)
                involves_3p = end >= n - 2
                strength = OligoEvaluator._classify_dimer_strength(delta_g)
                alignment = OligoEvaluator._format_dimer_alignment(
                    seq, offset, start, end)

                results.append(DimerResult(
                    delta_g=delta_g,
                    strength=strength,
                    num_pairs=num_pairs,
                    offset=offset,
                    paired_region=(start, end),
                    involves_3prime=involves_3p,
                    alignment=alignment,
                ))

        results.sort(key=lambda x: x.delta_g)
        return results[:max_results]

    # =================================================================
    #  find_hairpins
    # =================================================================

    @staticmethod
    def find_hairpins(sequence: str,
                      min_stem: int = 3,
                      min_loop: int = 3,
                      max_loop: int = 30,
                      temperature: float = 37.0,
                      max_results: int = 10) -> List[HairpinResult]:
        """
        Detect hairpin (stem-loop) secondary structures.

        Algorithm:
          1. For all (i, j) pairs in sequence, check if seq[i] and seq[j] are complementary
          2. Extend stem outward with (i, j) as innermost base pair
          3. Record when stem length >= min_stem and min_loop <= loop length <= max_loop
          4. Evaluate stability by dG = dG_stem + dG_loop (NN + loop penalty)

        dG_loop is approximated from SantaLucia (1998) / Zuker (2003) at 37 C.

        Args:
            sequence:    DNA sequence
            min_stem:    Minimum stem length (bp, default 3)
            min_loop:    Minimum loop length (nt, default 3)
            max_loop:    Maximum loop length (nt, default 30)
            temperature: Temperature (C, default 37.0)
            max_results: Maximum number of results to return

        Returns:
            List of HairpinResult (sorted by dG ascending = most stable first)
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        n = len(seq)
        comp = OligoEvaluator.COMPLEMENT_MAP

        results: List[HairpinResult] = []
        seen: set = set()

        for i in range(n - min_loop - 2 * min_stem + 1):
            for j in range(i + min_loop + 1,
                           min(i + max_loop + 2, n)):
                if comp.get(seq[i]) != seq[j]:
                    continue

                # Extend stem outward with (i, j) as innermost pair
                stem_len = 1
                while True:
                    p5 = i - stem_len
                    p3 = j + stem_len
                    if p5 < 0 or p3 >= n:
                        break
                    if comp.get(seq[p5]) != seq[p3]:
                        break
                    stem_len += 1

                if stem_len < min_stem:
                    continue

                s5s = i - stem_len + 1      # stem 5' start
                ls = i + 1                   # loop start
                le = j                       # loop end
                s3e = j + stem_len           # stem 3' end
                loop_size = le - ls

                key = (s5s, ls, le, s3e)
                if key in seen:
                    continue
                seen.add(key)

                stem5 = seq[s5s:ls]
                loop_seq = seq[ls:le]
                stem3 = seq[le:s3e]

                dg_stem = OligoEvaluator._calc_nn_delta_g(stem5, temperature)
                dg_loop = OligoEvaluator._hairpin_loop_penalty(loop_size)
                dg_total = round(dg_stem + dg_loop, 2)

                strength = OligoEvaluator._classify_hairpin_strength(dg_total)
                alignment = OligoEvaluator._format_hairpin_alignment(
                    seq, s5s, ls, le, s3e)

                results.append(HairpinResult(
                    delta_g=dg_total,
                    delta_g_stem=round(dg_stem, 2),
                    delta_g_loop=round(dg_loop, 2),
                    strength=strength,
                    stem_length=stem_len,
                    loop_size=loop_size,
                    stem_5prime=stem5,
                    loop_seq=loop_seq,
                    stem_3prime=stem3,
                    position=key,
                    alignment=alignment,
                ))

        results.sort(key=lambda x: x.delta_g)
        return results[:max_results]

    # =================================================================
    #  Display Methods
    # =================================================================

    @staticmethod
    def print_self_dimers(sequence: str,
                          min_pairs: int = 3,
                          temperature: float = 37.0,
                          max_results: int = 10) -> None:
        """Print self-dimer analysis results to console."""
        seq = OligoEvaluator._clean_sequence(sequence)
        dimers = OligoEvaluator.find_self_dimers(
            seq, min_pairs, temperature, max_results)

        print(f"\n{'='*60}")
        print(f"  Self-Dimer Analysis")
        print(f"  Sequence: 5'-{seq}-3' ({len(seq)} nt)")
        print(f"  Temperature: {temperature} C")
        print(f"{'='*60}")

        if not dimers:
            print("  No self-dimers detected.")
            print(f"{'='*60}")
            return

        for idx, d in enumerate(dimers, 1):
            tag_3p = " *3' end*" if d.involves_3prime else ""
            print(f"\n  #{idx}  dG = {d.delta_g} kcal/mol  "
                  f"[{d.strength}]  {d.num_pairs} bp{tag_3p}")
            for line in d.alignment.split('\n'):
                print(f"    {line}")

        print(f"\n{'='*60}")

    @staticmethod
    def print_hairpins(sequence: str,
                       min_stem: int = 3,
                       min_loop: int = 3,
                       max_loop: int = 30,
                       temperature: float = 37.0,
                       max_results: int = 10) -> None:
        """Print hairpin (secondary structure) analysis results to console."""
        seq = OligoEvaluator._clean_sequence(sequence)
        hairpins = OligoEvaluator.find_hairpins(
            seq, min_stem, min_loop, max_loop, temperature, max_results)

        print(f"\n{'='*60}")
        print(f"  Secondary Structure (Hairpin) Analysis")
        print(f"  Sequence: 5'-{seq}-3' ({len(seq)} nt)")
        print(f"  Temperature: {temperature} C")
        print(f"{'='*60}")

        if not hairpins:
            print("  No hairpin structures detected.")
            print(f"{'='*60}")
            return

        for idx, h in enumerate(hairpins, 1):
            print(f"\n  #{idx}  dG = {h.delta_g} kcal/mol "
                  f"(stem: {h.delta_g_stem}, loop: +{h.delta_g_loop})  "
                  f"[{h.strength}]")
            print(f"       Stem: {h.stem_length} bp, "
                  f"Loop: {h.loop_size} nt")
            for line in h.alignment.split('\n'):
                print(f"       {line}")

        print(f"\n{'='*60}")

    # =================================================================
    #  DP-based Secondary Structure Prediction
    #  [1] Nussinov - base pair maximization  O(n^3)
    #  [2] Zuker (simplified) - free energy minimization  O(n^4)
    # =================================================================

    _BULGE_PENALTY_37: Dict[int, float] = {
        1: 3.8, 2: 2.8, 3: 3.2, 4: 3.6, 5: 4.0,
    }
    _INTERNAL_PENALTY_37: Dict[int, float] = {
        2: 1.0, 3: 1.5, 4: 1.7, 5: 2.0, 6: 2.2,
    }
    _MULTI_INIT: float = 3.4
    _MAX_INTERNAL_LOOP: int = 30

    @staticmethod
    def _bulge_penalty(size: int) -> float:
        """Bulge loop dG penalty (kcal/mol, 37 C)"""
        if size <= 0:
            return float('inf')
        if size <= 5:
            return OligoEvaluator._BULGE_PENALTY_37.get(size, 4.0)
        return 4.0 + 1.75 * 1.987 * 310.15 * math.log(size / 5.0) / 1000.0

    @staticmethod
    def _internal_penalty(size1: int, size2: int) -> float:
        """Internal loop dG penalty (kcal/mol, 37 C)"""
        total = size1 + size2
        if total <= 0:
            return float('inf')
        if total <= 6:
            base = OligoEvaluator._INTERNAL_PENALTY_37.get(total, 2.2)
        else:
            base = (2.2 + 1.75 * 1.987 * 310.15
                    * math.log(total / 6.0) / 1000.0)
        return base + 0.3 * abs(size1 - size2)

    # -------------------- Nussinov --------------------

    @staticmethod
    def nussinov_fold(sequence: str, min_loop: int = 3) -> dict:
        """
        Nussinov algorithm: DP that maximizes Watson-Crick base pairs.

        Recurrence:
          dp[i][j] = max(
            dp[i][j-1],
            max_{i<=k<j-ml} dp[i][k-1] + 1 + dp[k+1][j-1]  (if k,j pair)
          )

        Time: O(n^3),  Space: O(n^2)

        Returns:
            dict: method, max_pairs, structure (dot-bracket), pairs
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        n = len(seq)
        comp = OligoEvaluator.COMPLEMENT_MAP

        dp = [[0] * n for _ in range(n)]

        for length in range(min_loop + 2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = dp[i][j - 1] if j > i else 0
                for k in range(i, j - min_loop):
                    if comp.get(seq[k]) == seq[j]:
                        left = dp[i][k - 1] if k > i else 0
                        inner = dp[k + 1][j - 1] if k + 1 <= j - 1 else 0
                        dp[i][j] = max(dp[i][j], left + 1 + inner)

        structure = list('.' * n)
        pairs: List[Tuple[int, int]] = []
        OligoEvaluator._nussinov_bt(
            dp, seq, 0, n - 1, structure, pairs, min_loop, comp)

        return {
            'method': 'Nussinov',
            'max_pairs': dp[0][n - 1],
            'structure': ''.join(structure),
            'pairs': sorted(pairs),
        }

    @staticmethod
    def _nussinov_bt(dp, seq, i, j, st, pairs, ml, comp):
        """Nussinov traceback (recursive)"""
        if i >= j or j - i < ml + 1:
            return
        if dp[i][j] == (dp[i][j - 1] if j > i else 0):
            OligoEvaluator._nussinov_bt(
                dp, seq, i, j - 1, st, pairs, ml, comp)
            return
        for k in range(i, j - ml):
            if comp.get(seq[k]) == seq[j]:
                left = dp[i][k - 1] if k > i else 0
                inner = dp[k + 1][j - 1] if k + 1 <= j - 1 else 0
                if left + 1 + inner == dp[i][j]:
                    st[k] = '('
                    st[j] = ')'
                    pairs.append((k, j))
                    if k > i:
                        OligoEvaluator._nussinov_bt(
                            dp, seq, i, k - 1, st, pairs, ml, comp)
                    if k + 1 <= j - 1:
                        OligoEvaluator._nussinov_bt(
                            dp, seq, k + 1, j - 1, st, pairs, ml, comp)
                    return

    # -------------------- Zuker (simplified) --------------------

    @staticmethod
    def zuker_fold(sequence: str,
                   temperature: float = 37.0,
                   min_loop: int = 3) -> dict:
        """
        Simplified Zuker algorithm: find MFE structure by DP.

        DP tables:
          V[i][j] - MFE when i and j form a base pair
          W[i][j] - MFE of seq[i..j] (any structure, 0 = no folding)

        Energy model:
          NN stacking (Breslauer 1986), hairpin loop,
          bulge, internal loop, multibranch (simplified)

        Time: O(n^4),  Space: O(n^2)

        Returns:
            dict: method, mfe, structure (dot-bracket), pairs
        """
        seq = OligoEvaluator._clean_sequence(sequence)
        OligoEvaluator._validate_dna(seq)
        n = len(seq)
        comp = OligoEvaluator.COMPLEMENT_MAP
        T_K = temperature + 273.15
        INF = float('inf')
        MAX_IL = OligoEvaluator._MAX_INTERNAL_LOOP
        MULTI = OligoEvaluator._MULTI_INIT

        def nn_dg(a: int, b: int) -> float:
            dn = seq[a] + seq[b]
            dh, ds = OligoEvaluator.NN_PARAMS.get(dn, (0.0, 0.0))
            return dh - T_K * ds / 1000.0

        V = [[INF] * n for _ in range(n)]
        W = [[0.0] * n for _ in range(n)]
        tb_V: List[List] = [[None] * n for _ in range(n)]
        tb_W: List[List] = [[None] * n for _ in range(n)]

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                # ===== V[i][j]: i pairs with j =====
                if (comp.get(seq[i]) == seq[j]
                        and j - i - 1 >= min_loop):
                    best_v = INF
                    best_tbv: tuple = ('H',)

                    # (a) Hairpin
                    e = OligoEvaluator._hairpin_loop_penalty(j - i - 1)
                    if e < best_v:
                        best_v = e
                        best_tbv = ('H',)

                    # (b) Stacking
                    if (j - i > min_loop + 2
                            and comp.get(seq[i + 1]) == seq[j - 1]
                            and V[i + 1][j - 1] < INF):
                        e = nn_dg(i, i + 1) + V[i + 1][j - 1]
                        if e < best_v:
                            best_v = e
                            best_tbv = ('S',)

                    # (c) Bulge / internal loop
                    for i2 in range(i + 1,
                                    min(i + MAX_IL + 2, j - min_loop)):
                        s1 = i2 - i - 1
                        j2_lo = j + s1 - MAX_IL - 1
                        j2_start = max(i2 + min_loop + 1, j2_lo)
                        for j2 in range(j2_start, j):
                            s2 = j - j2 - 1
                            if s1 + s2 == 0 or s1 + s2 > MAX_IL:
                                continue
                            if comp.get(seq[i2]) != seq[j2]:
                                continue
                            if V[i2][j2] >= INF:
                                continue
                            if s1 == 0 or s2 == 0:
                                e_il = OligoEvaluator._bulge_penalty(
                                    max(s1, s2))
                            else:
                                e_il = OligoEvaluator._internal_penalty(
                                    s1, s2)
                            e_il += V[i2][j2]
                            if e_il < best_v:
                                best_v = e_il
                                best_tbv = ('I', i2, j2)

                    # (d) Multibranch
                    for k in range(i + min_loop + 2, j - min_loop - 1):
                        e = W[i + 1][k] + W[k + 1][j - 1] + MULTI
                        if e < best_v:
                            best_v = e
                            best_tbv = ('M', k)

                    V[i][j] = best_v
                    tb_V[i][j] = best_tbv

                # ===== W[i][j]: any structure =====
                best_w = 0.0
                best_tbw: tuple = ('N',)

                if j > i:
                    if W[i][j - 1] < best_w:
                        best_w = W[i][j - 1]
                        best_tbw = ('L',)
                    if W[i + 1][j] < best_w:
                        best_w = W[i + 1][j]
                        best_tbw = ('R',)

                for k in range(i, j - min_loop):
                    if V[k][j] < INF:
                        e = V[k][j] + (W[i][k - 1] if k > i else 0.0)
                        if e < best_w:
                            best_w = e
                            best_tbw = ('P', k)

                W[i][j] = best_w
                tb_W[i][j] = best_tbw

        # ----- Backtrack -----
        mfe = W[0][n - 1]
        structure = list('.' * n)
        pairs: List[Tuple[int, int]] = []

        def bt_W(i: int, j: int) -> None:
            if i > j:
                return
            tb = tb_W[i][j]
            if tb is None or tb[0] == 'N':
                return
            if tb[0] == 'L':
                bt_W(i, j - 1)
            elif tb[0] == 'R':
                bt_W(i + 1, j)
            elif tb[0] == 'P':
                k = tb[1]
                if k > i:
                    bt_W(i, k - 1)
                bt_V(k, j)

        def bt_V(i: int, j: int) -> None:
            structure[i] = '('
            structure[j] = ')'
            pairs.append((i, j))
            tb = tb_V[i][j]
            if tb is None or tb[0] == 'H':
                return
            if tb[0] == 'S':
                bt_V(i + 1, j - 1)
            elif tb[0] == 'I':
                bt_V(tb[1], tb[2])
            elif tb[0] == 'M':
                k = tb[1]
                bt_W(i + 1, k)
                bt_W(k + 1, j - 1)

        bt_W(0, n - 1)

        return {
            'method': 'Zuker (simplified)',
            'mfe': round(mfe, 2),
            'structure': ''.join(structure),
            'pairs': sorted(pairs),
        }

    # -------------------- 3-method comparison --------------------

    @staticmethod
    def print_fold_comparison(sequence: str,
                              temperature: float = 37.0) -> None:
        """Compare Brute-Force / Nussinov / Zuker results side by side."""
        seq = OligoEvaluator._clean_sequence(sequence)

        print(f"\n{'='*70}")
        print(f"  Secondary Structure Prediction — Method Comparison")
        print(f"  Sequence: 5'-{seq}-3' ({len(seq)} nt)")
        print(f"  Temperature: {temperature} C")
        print(f"{'='*70}")

        # [1] Brute-Force
        hairpins = OligoEvaluator.find_hairpins(
            seq, temperature=temperature, max_results=3)
        print(f"\n  [1] Brute-Force Enumeration")
        print(f"      Complexity: O(n^2 * L_max)")
        if hairpins:
            h = hairpins[0]
            db = list('.' * len(seq))
            s5s, ls, le, s3e = h.position
            for k in range(ls - s5s):
                db[s5s + k] = '('
                db[s3e - 1 - k] = ')'
            print(f"      Most stable dG = {h.delta_g} kcal/mol  [{h.strength}]")
            print(f"      Stem: {h.stem_length} bp, Loop: {h.loop_size} nt")
            print(f"      5'-{seq}-3'")
            print(f"         {''.join(db)}")
        else:
            print(f"      No stable structure")

        # [2] Nussinov
        nuss = OligoEvaluator.nussinov_fold(seq)
        print(f"\n  [2] Nussinov (DP - base pair maximization)")
        print(f"      Complexity: O(n^3)")
        print(f"      Max base pairs: {nuss['max_pairs']} bp")
        print(f"      5'-{seq}-3'")
        print(f"         {nuss['structure']}")
        if nuss['pairs']:
            print(f"      Base pairs: {nuss['pairs']}")

        # [3] Zuker
        zuk = OligoEvaluator.zuker_fold(seq, temperature=temperature)
        print(f"\n  [3] Zuker (DP - free energy minimization)")
        print(f"      Complexity: O(n^4)")
        print(f"      MFE = {zuk['mfe']} kcal/mol")
        print(f"      5'-{seq}-3'")
        print(f"         {zuk['structure']}")
        if zuk['pairs']:
            print(f"      Base pairs: {zuk['pairs']}")
        else:
            print(f"      No stable secondary structure (dG >= 0)")

        print(f"\n{'='*70}")
