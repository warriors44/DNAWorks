from pathlib import Path
from typing import Literal, Optional, List
from dataclasses import dataclass, field
import subprocess
import re


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
        codon_table = {
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

        dna_sequence = dna_sequence.upper().replace(' ', '').replace('\n', '').replace('\t', '')
        
        print(f"Translating DNA sequence to protein sequence: {dna_sequence}")
        print(f"Length of DNA sequence: {len(dna_sequence)}")

        protein_sequence = ""
        for i in range(0, len(dna_sequence) - len(dna_sequence) % 3, 3):
            codon = dna_sequence[i:i+3]
            if codon in codon_table:
                amino_acid = codon_table[codon]
                if amino_acid != '*':
                    protein_sequence += amino_acid
                else:
                    break
            else:
                protein_sequence += 'X'
        
        return protein_sequence


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
        

def generate_input_file(fasta_path: Path, input_path: Path, logfile_path: Path) -> None:
    with open(fasta_path, "r") as f:
        sequence = f.readlines()[1].strip()

    sequence = sequence.replace("X", "TYR")
    config = DNAWorksConfig(
        jobname=fasta_path.parent.name,
        seqtype="protein",
        melting_low=68,
        melting_high=68,
        tolerance=1,
        frequency_threshold=25,
        strict=True,
        timelimit=300,
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
    Path("fna").mkdir(parents=True, exist_ok=True)
    Path("log").mkdir(parents=True, exist_ok=True)

    for fasta_path in Path.cwd().glob("**/*.fa"):
        identifier = fasta_path.parent.name
        
        input_path = Path("in")/(identifier+".inp")
        logfile_path = Path("log")/(identifier+".txt")
        fna_path = Path("fna")/(identifier+".fna")

        config = generate_input_file(fasta_path,input_path,logfile_path)
        DNAWorksConfig.run_dnaworks(input_path)
        dna_sequence = DNAWorksConfig.parse_logfile(logfile_path)
        config.assert_sequence(dna_sequence)

        with open(fna_path, "w") as f:
            f.write(">"+identifier+"\n"+dna_sequence)

if __name__ == "__main__":
    main()