from pathlib import Path
from dnaworks import Dnawork, CodonTable, generate_input_file

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
        config.run_dnaworks(input_path)
        dna_sequence = config.parse_logfile(logfile_path)
        config.assert_sequence(dna_sequence)

        with open(fna_path, "w") as f:
            f.write(">"+identifier+"\n"+dna_sequence)

    ct = CodonTable.load("ecoli2")
    Path("fna_modified").mkdir(parents=True, exist_ok=True)
    for fasta_path in Path.cwd().glob("fna/*.fna"):
        nucleotide_sequence = open(fasta_path, "r").readlines()[1].strip()
        amino_acid_sequence = Dnawork.translate_dna_to_protein(nucleotide_sequence, codon_table=ct)
        nucleotide_sequence = Dnawork.modify_unfavorble_codons(nucleotide_sequence, offset=1, codon_table=ct)
        assert Dnawork.translate_dna_to_protein(nucleotide_sequence, codon_table=ct) == amino_acid_sequence, "Translation mismatch"
        with open(f"./fna_modified/{fasta_path.stem}.fna", "w") as f:
            f.write(">"+fasta_path.stem+"\n"+nucleotide_sequence)
if __name__ == "__main__":
    main()