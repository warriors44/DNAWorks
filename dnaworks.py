from pathlib import Path
from typing import Literal, Optional, List, Tuple
from dataclasses import dataclass, field
import subprocess
import re
from collections import Counter
from itertools import product
from scipy.stats import entropy
import numpy as np

MeltingTemp = Literal[55,60,65,68,70,72,75]
Tolerance = Literal[1,2,3,4,5]
Timelimit = Literal[0,1,30,60,120,180,300]
Seqtype = Literal["protein", "nucleotide"]
Threshold = Literal[0,5,10,15,20,25,30]
Organism = Literal["ecoli2", "E.coli", "C.elegans","D.melanogaster","H.sapiens","M.musculus"]

Enzyme = Literal["AflII", "BamHI", "EcoRI", "NcoI","NdeI","HindIII"]
re_dict = {"AflII": "CCTAGG",
           "BamHI": "GGATCC",
           "EcoRI": "GAATTC",
           "NcoI": "CCATGG",
           "NdeI": "CATATG",
           "HindIII": "AAGCTT"  
           }

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

@dataclass
class DNAWorksConfig:
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
    def translate_dna_to_protein(dna_sequence: str) -> str:
        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')
        
       #print(f"Translating DNA sequence to protein sequence: {dna_sequence}")
        print(f"Length of DNA sequence: {len(dna_sequence)}")

        protein_sequence = ""
        for i in range(0, len(dna_sequence) - len(dna_sequence) % 3, 3):
            codon = dna_sequence[i:i+3]
            if codon in CODON_TABLE:
                amino_acid = CODON_TABLE[codon]
                if amino_acid != '*':
                    protein_sequence += amino_acid
                else:
                    break
            else:
                protein_sequence += 'X'
        
        return protein_sequence

    @staticmethod
    def find_consecutive_bases(dna_sequence: str, min_length: int = 6) -> List[tuple]:
        """
        同じ塩基がmin_length以上連続している箇所を検出する
        
        Args:
            dna_sequence: DNA配列
            min_length: 検出する最小連続長（デフォルト: 6）
        
        Returns:
            連続箇所のリスト。各要素は(start_index, end_index, base)のタプル
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
    def get_alternative_codons(amino_acid: str) -> List[str]:
        """
        指定されたアミノ酸をコードする全てのコドンのリストを返す
        
        Args:
            amino_acid: アミノ酸（1文字）
        
        Returns:
            コドンのリスト
        """
        alternative_codons = []
        for codon, aa in CODON_TABLE.items():
            if aa == amino_acid:
                alternative_codons.append(codon)
        return alternative_codons

    @staticmethod
    def find_restriction_sites(dna_sequence: str, excluded_enzymes: List[str]) -> List[Tuple[int, int, str]]:
        """
        DNA配列中の制限酵素認識配列の全出現位置を返す
        
        Args:
            dna_sequence: DNA配列
            excluded_enzymes: チェック対象の制限酵素名リスト
        
        Returns:
            (start, end, enzyme_name) のリスト
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
        連続配列および制限酵素認識配列に関わるコドンインデックスを全て収集する
        
        Args:
            dna_sequence: DNA配列
            codons: コドンのリスト
            max_consecutive: 許容する最大連続数
            excluded_enzymes: チェック対象の制限酵素名リスト
        
        Returns:
            (problem_codon_indices, consecutive_regions, restriction_sites) のタプル
        """
        num_codons = len(codons)
        problem_indices = set()
        
        # 1. 連続配列に関わるコドンを収集
        consecutive_regions = DNAWorksConfig.find_consecutive_bases(dna_sequence, max_consecutive + 1)
        for start, end, base in consecutive_regions:
            for i in range(num_codons):
                codon_start = i * 3
                codon_end = codon_start + 3
                if codon_start < end and codon_end > start:
                    problem_indices.add(i)
        
        # 2. 制限酵素認識配列に関わるコドンを収集
        restriction_sites = DNAWorksConfig.find_restriction_sites(dna_sequence, excluded_enzymes)
        for start, end, enzyme in restriction_sites:
            for i in range(num_codons):
                codon_start = i * 3
                codon_end = codon_start + 3
                if codon_start < end and codon_end > start:
                    problem_indices.add(i)
        
        return problem_indices, consecutive_regions, restriction_sites

    @staticmethod
    def expand_and_fragment(problem_indices: set, num_codons: int) -> List[List[int]]:
        """
        問題コドンに前後±1を追加し、連続するインデックスをフラグメントにグループ化する
        
        Args:
            problem_indices: 問題のあるコドンインデックスのset
            num_codons: 全コドン数
        
        Returns:
            フラグメントのリスト。各フラグメントはソート済みのコドンインデックスリスト
        """
        # 前後±1を追加
        expanded = set()
        for idx in problem_indices:
            expanded.add(idx)
            if idx > 0:
                expanded.add(idx - 1)
            if idx < num_codons - 1:
                expanded.add(idx + 1)
        
        if not expanded:
            return []
        
        # ソートして連続するインデックスをグループ化
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
    def avoid_consecutive_bases(dna_sequence: str, max_consecutive: int = 5,
                                excluded_enzymes: Optional[List[str]] = None) -> str:
        """
        連続配列(max_consecutive+1以上)および制限酵素認識配列を同時に除去する。
        
        アルゴリズム:
          1. 全体をスキャンし、連続配列・制限酵素認識配列に関わるコドンを収集
          2. 前後±1コドンを追加し、連続するインデックスをフラグメントに分割
          3. 各フラグメントについて全コドン組み合わせを網羅的に探索
          4. フラグメント内部±1コドンの範囲で連続配列・認識配列がないことを確認
        
        Args:
            dna_sequence: DNA配列（3の倍数である必要がある）
            max_consecutive: 許容する最大連続数（デフォルト: 5、つまり6以上を避ける）
            excluded_enzymes: 避けるべき制限酵素名のリスト（デフォルト: None → re_dict全体）
        
        Returns:
            修正されたDNA配列
        """
        if excluded_enzymes is None:
            excluded_enzymes = list(re_dict.keys())
        
        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')
        
        # DNA配列をコドンに分割
        codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence) - len(dna_sequence) % 3, 3)]
        num_codons = len(codons)
        
        # 各コドンをアミノ酸に翻訳
        amino_acids = [CODON_TABLE.get(c, 'X') for c in codons]
        
        # 各アミノ酸に対応する全コドンリストを事前計算
        aa_to_codons = {}
        for codon, aa in CODON_TABLE.items():
            aa_to_codons.setdefault(aa, []).append(codon)
        
        # ステップ1: 問題コドンを収集
        problem_indices, consecutive_regions, restriction_sites = \
            DNAWorksConfig.collect_problem_codons(dna_sequence, codons, max_consecutive, excluded_enzymes)
        
        print(f"\n{'='*60}")
        print(f"[配列最適化開始] DNA配列長: {len(dna_sequence)}塩基 ({num_codons}コドン)")
        print(f"  除外対象の制限酵素: {excluded_enzymes}")
        print(f"  最大許容連続数: {max_consecutive}")
        print(f"\n  検出された問題:")
        if consecutive_regions:
            print(f"    連続配列: {len(consecutive_regions)}箇所")
            for s, e, b in consecutive_regions:
                print(f"      位置 {s}-{e}: {b} x {e-s}")
        else:
            print(f"    連続配列: なし")
        if restriction_sites:
            print(f"    制限酵素認識配列: {len(restriction_sites)}箇所")
            for s, e, enz in restriction_sites:
                print(f"      位置 {s}-{e}: {enz} ({re_dict[enz]})")
        else:
            print(f"    制限酵素認識配列: なし")
        
        if not problem_indices:
            print(f"\n  問題箇所なし。修正不要です。")
            print(f"{'='*60}")
            return dna_sequence
        
        print(f"\n  問題に関わるコドン数: {len(problem_indices)}")
        
        # ステップ2: 前後±1を追加してフラグメント化
        fragments = DNAWorksConfig.expand_and_fragment(problem_indices, num_codons)
        
        print(f"  フラグメント数: {len(fragments)}")
        for i, frag in enumerate(fragments):
            print(f"    フラグメント{i+1}: コドン位置 {frag[0]}-{frag[-1]} ({len(frag)}コドン)")
        print(f"{'='*60}")
        
        # ステップ3: フラグメント単位で網羅的探索
        total_replacements = 0
        failed_fragments = []
        
        for frag_idx, fragment in enumerate(fragments):
            # フラグメント内の各コドンの代替コドン候補を列挙
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
            
            print(f"\n[フラグメント{frag_idx+1}] コドン位置 {fragment[0]}-{fragment[-1]}")
            print(f"  現在のコドン: {[codons[i] for i in fragment]}")
            print(f"  アミノ酸:     {[amino_acids[i] for i in fragment]}")
            print(f"  探索する組み合わせ数: {total_combinations}")
            
            # チェック範囲: フラグメントの前後1コドンを含むDNA部分文字列
            check_start_codon = max(0, fragment[0] - 1)
            check_end_codon = min(num_codons - 1, fragment[-1] + 1)
            check_start_bp = check_start_codon * 3
            check_end_bp = (check_end_codon + 1) * 3
            
            # 全組み合わせを網羅的に探索
            best_combo = None
            best_num_mutations = float('inf')
            
            for combo in product(*codon_options):
                # フラグメント内のコドンを一時的に置き換え
                temp_codons = codons.copy()
                for i, idx in enumerate(fragment):
                    temp_codons[idx] = combo[i]
                
                # チェック範囲の部分配列を取得
                check_subsequence = ''.join(temp_codons[check_start_codon:check_end_codon + 1])
                
                # 連続配列チェック（チェック範囲内）
                if DNAWorksConfig.find_consecutive_bases(check_subsequence, max_consecutive + 1):
                    continue
                
                # 制限酵素認識配列チェック（チェック範囲内）
                if DNAWorksConfig.find_restriction_sites(check_subsequence, excluded_enzymes):
                    continue
                
                # 変異数を計算
                num_mutations = sum(1 for i, idx in enumerate(fragment)
                                    if combo[i] != codons[idx])
                
                if num_mutations < best_num_mutations:
                    best_num_mutations = num_mutations
                    best_combo = combo
                    # 変異数0は元のまま（問題あるはずだが念のため）
                    # 変異数1で十分なら即採用
                    if best_num_mutations <= 1:
                        break
            
            if best_combo is not None:
                # 最適な組み合わせを適用
                changes = []
                for i, idx in enumerate(fragment):
                    if best_combo[i] != codons[idx]:
                        changes.append((idx, codons[idx], best_combo[i], amino_acids[idx]))
                        codons[idx] = best_combo[i]
                
                total_replacements += len(changes)
                if changes:
                    print(f"  [修正成功] 変異数: {len(changes)}")
                    for idx, old, new, aa in changes:
                        print(f"    コドン位置 {idx}: {old} -> {new} (アミノ酸: {aa})")
                else:
                    print(f"  [修正不要] 元のコドンで問題なし")
            else:
                failed_fragments.append((frag_idx + 1, fragment))
                print(f"  [警告] 全{total_combinations}通りを探索しましたが解決できませんでした")
        
        # 最終レポート
        final_sequence = ''.join(codons)
        final_consecutive = DNAWorksConfig.find_consecutive_bases(final_sequence, max_consecutive + 1)
        final_restriction = DNAWorksConfig.find_restriction_sites(final_sequence, excluded_enzymes)
        
        print(f"\n{'='*60}")
        print(f"[配列最適化完了]")
        print(f"  総置き換えコドン数: {total_replacements}")
        
        if final_consecutive:
            print(f"  残存連続配列: {len(final_consecutive)}箇所")
            for s, e, b in final_consecutive:
                print(f"    位置 {s}-{e}: {b} x {e-s}")
        else:
            print(f"  残存連続配列: なし (OK)")
        
        if final_restriction:
            print(f"  残存制限酵素認識配列: {len(final_restriction)}箇所")
            for s, e, enz in final_restriction:
                print(f"    位置 {s}-{e}: {enz} ({re_dict[enz]})")
        else:
            print(f"  残存制限酵素認識配列: なし (OK)")
        
        if failed_fragments:
            print(f"\n  [警告] 解決できなかったフラグメント: {len(failed_fragments)}個")
            for frag_num, frag in failed_fragments:
                print(f"    フラグメント{frag_num}: コドン位置 {frag[0]}-{frag[-1]}")
        
        print(f"{'='*60}")
        
        return final_sequence


    def assert_sequence(self,dna_sequence: str) -> None:
        translated_sequence = self.translate_dna_to_protein(dna_sequence)
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
        counts = Counter(sequence)    # 各文字の出現頻度をカウント
        probabilities = np.array(list(counts.values())) / sum(counts.values())  # 確率分布
        return entropy(probabilities, base=2)  # エントロピーを計算（単位: bit）
  

def generate_input_file(fasta_path: Path, input_path: Path, logfile_path: Path) -> None:
    with open(fasta_path, "r") as f:
        sequence = f.readlines()[1].strip()

    sequence = sequence.replace("X", "TYG")
    sequence = "G"+sequence
    config = DNAWorksConfig(
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



def main():
    Path("in").mkdir(parents=True, exist_ok=True)
    Path("fna_tmp").mkdir(parents=True, exist_ok=True)
    Path("log").mkdir(parents=True, exist_ok=True)

    for fasta_path in Path.cwd().glob("fa/*.fa"):
        identifier = fasta_path.stem
        
        input_path = Path("in")/(identifier+".inp")
        logfile_path = Path("log")/(identifier+".txt")
        fna_path = Path("fna_tmp")/(identifier+".fna")

        config = generate_input_file(fasta_path,input_path,logfile_path)
        # entropy = DNAWorksConfig.calculate_sequence_entropy(config.sequence)
        # print(f"{identifier}: {entropy}")
        DNAWorksConfig.run_dnaworks(input_path)
        dna_sequence = DNAWorksConfig.parse_logfile(logfile_path)
        config.assert_sequence(dna_sequence)

        with open(fna_path, "w") as f:
            f.write(">"+identifier+"\n"+dna_sequence)

    # Path("fna_modified").mkdir(parents=True, exist_ok=True)
    # for fasta_path in Path.cwd().glob("fna/*.fna"):
    #     nucleotide_sequence = open(fasta_path, "r").readlines()[1].strip()
    #     amino_acid_sequence = DNAWorksConfig.translate_dna_to_protein(nucleotide_sequence)
    #     nucleotide_sequence = DNAWorksConfig.avoid_consecutive_bases(nucleotide_sequence)
    #     assert DNAWorksConfig.translate_dna_to_protein(nucleotide_sequence) == amino_acid_sequence, "Translation mismatch"
    #     with open(f"./fna_modified/{fasta_path.stem}.fna", "w") as f:
    #         f.write(">"+fasta_path.stem+"\n"+nucleotide_sequence)
if __name__ == "__main__":
    main()