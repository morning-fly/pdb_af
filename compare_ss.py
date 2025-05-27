#!/usr/bin/env python3
"""
compare_ss.py

Compare secondary structure between experimental PDB chains and AlphaFold predictions.

This script filters PDB entries by experimental method, selects one representative per UniProt ID,
performs trimming by sequence alignment, analyzes secondary structure via DSSP,
and aggregates the results for plotting comparison statistics.
"""

import argparse
import tempfile
import shutil
from Bio.PDB import PDBParser, DSSP, PDBIO, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB import PPBuilder
from Bio.Align import PairwiseAligner
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List, Any
from loguru import logger
from multiprocessing import Pool
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from functools import partial

def main():
    parser = argparse.ArgumentParser(
        description="Compare the structures deposited in PDB with the predicted structure in AF"
        )
    parser.add_argument(
        "-d",
        "--database",
        help="directory containing PDB deposited structures",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    database_path = Path(args.database)
    
    log = Path.cwd() / f"{Path(__file__).stem}.log"
    logger.add(log, format="{time:YYYY-MM-DD HH:mm:ss} [{process.name}/{process.id}] {level}: {message}")    
    
    pdb_info_csv = next(database_path.glob("all_chain_pdb_info.csv"), None)
    all_chain_info_df = pd.read_csv(pdb_info_csv)
    
    filtered_df = all_chain_info_df[all_chain_info_df['EXPERIMENT TYPE (IF NOT X-RAY)'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    filtered_df = filtered_df.copy()
    filtered_df["RESOLUTION"] = filtered_df["RESOLUTION"].astype("float")
    remove_dup_df = filtered_df.sort_values("RESOLUTION").drop_duplicates(subset="UNIPROT_ID", keep="first")
    
    remove_dup_rows = remove_dup_df.to_dict(orient="records")
    with Pool(processes=16) as pool:
        results = pool.map(partial(main_flow, database_path), remove_dup_rows)
    
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)
    results_df.to_csv("ss_comparison_allchains.csv", index=False)
    
    plot_from_df(df=results_df)

def main_flow(db_path: Union[str, Path], row_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single chain: load structures, align, trim, and analyze SS.

    Args:
        db_path (Union[str, Path]): Base directory of downloaded structures.
        row (Dict[str, Any]): Row with keys 'IDCODE', 'CHAIN', 'UNIPROT_ID', 'HEADER'.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with comparison results or None if skipped.
    """
    # Initialize the PDB parser (QUIET mode suppresses warnings)
    parser = PDBParser(QUIET=True)
    pdb_id = str(row_dict["IDCODE"])
    chain_id = str(row_dict["CHAIN"])
    uniprot_id = str(row_dict["UNIPROT_ID"])
    header = str(row_dict["HEADER"])
    db_path = Path(db_path)
    
    # get exp structure
    pdb_dir = db_path / "PDB" / pdb_id
    exp_pdb = get_structure_in_dir(directory=pdb_dir, name=f"{pdb_id}_{chain_id}")
    if exp_pdb is None or not exp_pdb.exists():
        logger.warning(f"Skipping {pdb_id}_{chain_id}: experimental PDB not found ({exp_pdb})")
        return None
    try:
        exp_structure = parser.get_structure('exp', str(exp_pdb))
    except Exception as e:
        logger.warning(f"Skipping {pdb_id}_{chain_id}: failed to parse experimental PDB ({e})")
        return None
    try:
        exp_chain = exp_structure[0]["A"]
    except KeyError:
        logger.warning(f"Skipping {pdb_id}_{chain_id}: chain {chain_id} not found in experimental structure")
        return None

    # get af structure
    af_dir = db_path / "AlphaFold"
    af_pdb = get_structure_in_dir(directory=af_dir, name=uniprot_id)
    if af_pdb is None or not af_pdb.exists():
        logger.warning(f"Skipping {pdb_id}_{chain_id}: AlphaFold PDB not found ({af_pdb})")
        return None
    try:
        af_structure = parser.get_structure('af', str(af_pdb))
    except Exception as e:
        logger.warning(f"Skipping {pdb_id}_{chain_id}: failed to parse AlphaFold PDB ({e})")
        return None
    try:
        af_chain = af_structure[0]["A"]
    except KeyError:
        logger.warning(f"Skipping {pdb_id}_{chain_id}: chain A not found in AlphaFold structure")
        return None
    

    trimmed_residues = trim_af_structure_by_alignment(exp_chain, af_chain)
    trimmed_af_chain = create_chain_from_residues(trimmed_residues, "A")
    
    exp_ss = analyze_ss_from_chain(exp_chain, pdb_id, "exp")
    af_ss = analyze_ss_from_chain(trimmed_af_chain, pdb_id, "af")
    
    result = {
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'uniprot_id': uniprot_id,
        'header': header,
        'exp_secondary': exp_ss,
        'af_secondary': af_ss
    }

    return result

def get_structure_in_dir(directory: Union[str, Path], name: str) -> Optional[Path]:
    """
    Retrieve the path of a PDB structure file with the given basename from a directory.

    Args:
        directory (Union[str, Path]): Directory in which to look for the PDB file.
        name (str): Basename of the file to find (without “.pdb”).

    Returns:
        Optional[Path]: Path to the first matching file if found, otherwise None.
    """
    directory = Path(directory)
    return next(directory.glob(f"{name}.pdb"), None)

def add_dummy_cryst1(pdb_path: Union[str, Path]) -> None:
    """
    Ensure the PDB file begins with a CRYST1 record by inserting a dummy if missing.

    Some DSSP implementations require CRYST1; this adds a minimal one.

    Args:
        pdb_path (Union[str, Path]): Path to the PDB file to modify.

    Returns:
        None
    """
    pdb_path = Path(pdb_path)
    # Read file content while preserving line endings
    lines = pdb_path.read_text().splitlines(keepends=True)
    
    # Check if a CRYST1 record is present
    if not any(line.startswith("CRYST1") for line in lines):
        dummy_cryst1 = "CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {sg:<11}{z:>4}\n".format(
            a=1.0, b=1.0, c=1.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            sg="P 1", z=1
        )
        # Insert dummy CRYST1 record at the beginning
        lines.insert(0, dummy_cryst1)
        pdb_path.write_text("".join(lines))

class ChainSelect(Select):
    """
    PDBIO Select subclass that filters for a single chain.

    Args:
        chain_id (str): Identifier of the chain to accept.
    """
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
def write_chain_to_temp(chain: Chain, pdb_id: str, source_tag: str) -> Tuple[Path, Path]:
    """
    Write a single chain to a temporary PDB file (with CRYST1 dummy).

    Args:
        chain (Chain): The chain to write.
        pdb_id (str): PDB identifier for naming.
        source_tag (str): Tag describing origin (“exp”/“af” etc).

    Returns:
        Tuple[Path, Path]: (path to temp PDB file, path to temp directory).
    """
    # Create a temporary directory as a Path object
    tmp_dir = Path(tempfile.mkdtemp())
    temp_file = tmp_dir / f"{pdb_id}_{chain.id}_{source_tag}.pdb"
    
    io = PDBIO()
    io.set_structure(chain)
    # Save the chain to the temporary file using ChainSelect to filter the chain by id
    io.save(str(temp_file), select=ChainSelect(chain.id))
    
    # Add a dummy CRYST1 record if missing (to prevent DSSP errors)
    add_dummy_cryst1(temp_file)
    
    return temp_file, tmp_dir

def analyze_ss_from_chain(chain: Chain, pdb_id: str, source_tag: str) -> Dict[str, float]:
    """
    Compute secondary-structure percentages for a chain via DSSP.

    This writes the chain out, runs DSSP, counts assignments, and
    returns percentage composition.

    Args:
        chain (Chain): Chain to analyze.
        pdb_id (str): PDB identifier (for logging).
        source_tag (str): Tag indicating source (“exp”/“af”).

    Returns:
        Dict[str, float]: Mapping DSSP symbols (e.g. 'H','E','B') → percentage.
    """
    # 1) Create a temporary PDB file for this chain
    temp_file, tmp_dir = write_chain_to_temp(
        chain=chain, pdb_id=pdb_id, source_tag=source_tag
    )
    try:
        # 2) Parse the temporary PDB file into a Structure object
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('temp', str(temp_file))
        except Exception as e:
            logger.warning(f"DSSP prep failed for {pdb_id}[{source_tag}]: cannot parse PDB ({e})")
            return {}

        # 3) Retrieve the first model (model id 0)
        try:
            model = structure[0]
        except (KeyError, IndexError):
            logger.warning(f"DSSP prep failed for {pdb_id}[{source_tag}]: no model 0")
            return {}

        # 4) Run DSSP on the parsed model
        try:
            dssp = DSSP(model, str(temp_file), dssp="mkdssp")
        except Exception as e:
            logger.warning(f"DSSP failed for {pdb_id} chain {chain.id} ({source_tag}): {e}")
            return {}

        # 5) Count secondary structure assignments for this chain
        counts: Dict[str, int] = {}
        total = 0
        for (chain_id_key, res_id), d in dssp.property_dict.items():
            if chain_id_key != chain.id:
                continue
            ss_type = d[2]  # Secondary structure symbol
            counts[ss_type] = counts.get(ss_type, 0) + 1
            total += 1

        # 6) Compute percentage composition of each secondary structure type
        if total == 0:
            return {}
        percentages = {
            ss_type: (count / total) * 100.0
            for ss_type, count in counts.items()
        }
        return percentages

    finally:
        # 7) Clean up the temporary directory and files
        shutil.rmtree(tmp_dir, ignore_errors=True)

def get_chain_sequence_and_residues(chain: Chain) -> Tuple[str, List[Residue]]:
    """
    Extract full sequence and residue list from a chain.

    Args:
        chain (Chain): Target chain.

    Returns:
        Tuple[str, List[Residue]]: (amino‐acid sequence,  corresponding residues).
    """
    ppb = PPBuilder()
    seq = ""
    residues = []
    for peptide in ppb.build_peptides(chain):
        # Concatenate peptide fragments if multiple fragments exist
        seq += str(peptide.get_sequence())
        residues.extend(peptide)
    return seq, residues

def trim_af_structure_by_alignment(exp_chain: Chain, af_chain: Chain) -> List[Residue]:
    """
    Align experimental vs. AF sequences and return AF residues matching exp regions.

    Also filters by sequence identity ≥90%.

    Args:
        exp_chain (Chain): Experimental chain.
        af_chain (Chain): AlphaFold chain.

    Returns:
        List[Residue]: Selected AF residues aligned to exp non-gap positions.
    """

    # Suppress PDB construction warnings
    warnings.simplefilter("ignore", PDBConstructionWarning)

    # Extract sequences and residue lists from both chains.
    # It is assumed that get_chain_sequence_and_residues() returns a tuple (sequence, residue_list).
    exp_seq, _ = get_chain_sequence_and_residues(exp_chain)
    af_seq, af_residues = get_chain_sequence_and_residues(af_chain)
    
    if not exp_seq:
        logger.warning("Skipping trim: experimental sequence is empty")
        return []
    if not af_seq:
        logger.warning("Skipping trim: AlphaFold sequence is empty")
        return []

    # Perform global alignment using PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1

    alignments = aligner.align(exp_seq, af_seq)
    best_alignment = alignments[0]

    # Convert the alignment to a string and split into lines
    alignment_str = str(best_alignment)
    logger.info(alignment_str)
    lines = alignment_str.splitlines()

    aligned_exp_full = ""
    aligned_af_full = ""
    block = []  # Temporary container for lines within each alignment block

    # Process alignment blocks separated by empty lines
    for line in lines:
        if line.strip() == "":
            # Process the current block if it contains at least 3 lines
            if len(block) >= 3:
                target_line = block[0]
                query_line = block[2]
                # Check if the line contains header info (if so, split; otherwise, use the line directly)
                target_parts = target_line.split(maxsplit=2)
                query_parts = query_line.split(maxsplit=2)
                if len(target_parts) == 1:
                    target_seq = target_line.strip()
                elif len(target_parts) >= 3:
                    target_seq = target_parts[2].strip()
                else:
                    target_seq = ""
                if len(query_parts) == 1:
                    query_seq = query_line.strip()
                elif len(query_parts) >= 3:
                    query_seq = query_parts[2].strip()
                else:
                    query_seq = ""
                # Skip the block if the last token of target_seq is numeric
                if target_seq.split() and target_seq.split()[-1].isdigit():
                    logger.debug("Skipping block with trailing numeric tokens.")
                else:
                    aligned_exp_full += target_seq
                    aligned_af_full += query_seq
            block = []  # Reset the block for the next alignment segment
        else:
            block.append(line)

    # Process the final block if not terminated by an empty line
    if block and len(block) >= 3:
        target_line = block[0]
        query_line = block[2]
        target_parts = target_line.split(maxsplit=2)
        query_parts = query_line.split(maxsplit=2)
        if len(target_parts) == 1:
            target_seq = target_line.strip()
        elif len(target_parts) >= 3:
            target_seq = target_parts[2].strip()
        else:
            target_seq = ""
        if len(query_parts) == 1:
            query_seq = query_line.strip()
        elif len(query_parts) >= 3:
            query_seq = query_parts[2].strip()
        else:
            query_seq = ""
        if not (target_seq.split() and target_seq.split()[-1].isdigit()):
            aligned_exp_full += target_seq
            aligned_af_full += query_seq

    # Ensure the aligned sequences have the same length.
    if len(aligned_exp_full) != len(aligned_af_full):
        raise ValueError("Aligned sequences have different lengths.")
    
    identity = calculate_identity(aligned_exp_full, aligned_af_full)
    if identity < 90:
        logger.warning(f"sequence identity is not sufficient: {identity}%")
        return []

    # Determine which positions in the aligned AlphaFold sequence correspond to actual residues.
    af_counter = 0
    indices_to_keep = []  # List of indices in af_residues to keep
    for pos in range(len(aligned_af_full)):
        if aligned_af_full[pos] != "-":
            # Consider this position only if the experimental aligned sequence is not a gap
            if aligned_exp_full[pos] != "-":
                indices_to_keep.append(af_counter)
            af_counter += 1  # Increment counter only for non-gap characters
        # Do not increment af_counter if a gap is encountered.

    continuous_helix_seg_list = continuous_helix(chain=af_chain, pdb_id="test", source_tag="temp")
    new_indices_to_keep = indices_to_keep.copy()
    for helix_seg in continuous_helix_seg_list:
        for i in indices_to_keep:
            resi_num = i + 1
            if resi_num in helix_seg:
                new_indices_to_keep += [j - 1 for j in helix_seg]
                break
    new_indices_to_keep = sorted(list(set(new_indices_to_keep)))

    # Extract the Residue objects from af_residues corresponding to the selected indices (not residue number).
    trimmed_residues = [af_residues[i] for i in new_indices_to_keep]
    return trimmed_residues

def calculate_identity(aligned_seq1: str, aligned_seq2: str) -> float:
    """
    Compute percent identity between two aligned sequences (ignore gaps).

    Args:
        aligned_seq1 (str): First aligned sequence with '-' for gaps.
        aligned_seq2 (str): Second aligned sequence.

    Returns:
        float: Percent identity over non-gap columns.
    """
    # Initialize counters for match count and valid (non-gap) position count
    match_count = 0
    valid_count = 0
    
    # Loop over each pair of characters from both sequences
    for a, b in zip(aligned_seq1, aligned_seq2):
        # Skip the positions where either sequence has a gap
        if a == '-' or b == '-':
            continue
        
        valid_count += 1
        if a == b:
            match_count += 1

    # Return 0.0 if there are no valid positions to avoid division by zero
    if valid_count == 0:
        return 0.0

    # Calculate the percent identity
    identity = (match_count / valid_count) * 100
    return identity

def create_chain_from_residues(residues: List[Residue], chain_id: str) -> Chain:
    """
    Build a new Chain object containing the given residues.

    Args:
        residues (List[Residue]): Residue objects to include.
        chain_id (str): New chain’s identifier.

    Returns:
        Chain: Chain populated with `residues`.
    """
    new_chain = Chain(chain_id)
    for res in residues:
        new_chain.add(res)
    return new_chain

def continuous_helix(chain: Chain, pdb_id: str, source_tag: str) -> List[List[int]]:
    """
    Identify continuous alpha-helix segments (DSSP 'H') in a chain.

    Args:
        chain (Chain): Chain to inspect.
        pdb_id (str): PDB identifier.
        source_tag (str): Tag for logging (“exp”/“af”).

    Returns:
        List[List[int]]: List of DSSP index-lists, each a continuous helix segment.
    """
    temp_file, tmp_dir = write_chain_to_temp(chain=chain, pdb_id=pdb_id, source_tag=source_tag)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('temp', temp_file)
    model = structure[0]
    try:
        dssp = DSSP(model, temp_file, dssp="mkdssp")
    except Exception as e:
        print(f"DSSP failed for {pdb_id} chain {chain.id} ({source_tag}): {e}")
        shutil.rmtree(tmp_dir)
        return []
    
    helix_segments = []
    current_segment = []

    chain_keys = sorted([key for key in dssp.keys() if key[0] == chain.id], key=lambda k: dssp[k][0])
    
    for key in chain_keys:
        ss = dssp[key][2]
        if ss == 'H':
            current_segment.append(dssp[key][0])
        else:
            if current_segment:
                helix_segments.append(current_segment)
                current_segment = []

    if current_segment:
        helix_segments.append(current_segment)
    
    shutil.rmtree(tmp_dir)
    return helix_segments

def plot_from_df(df: pd.DataFrame) -> None:
    """
    Generate scatter/bar plots comparing experimental vs. AF secondary-structure stats.

    Args:
        df (pd.DataFrame): DataFrame with columns 'exp_secondary' and 'af_secondary'
                           holding dicts of %SS assignments.

    Returns:
        None
    """
    # 1: Expand secondary-structure dicts into numeric columns
    exp = pd.DataFrame(df['exp_secondary'].tolist()).fillna(0).astype(float)
    af  = pd.DataFrame(df['af_secondary'].tolist()).fillna(0).astype(float)
    df = df.assign(
        exp_H=exp['H'], exp_BE=exp['B'] + exp['E'],
        af_H=af['H'],   af_BE=af['B'] + af['E']
    )

    # 2: Separate by membrane/soluble and also include total
    mask = df['header'].str.contains('MEMBRANE PROTEIN', na=False)
    groups = {'Membrane': df[mask], 'Soluble': df[~mask], 'Total': df}

    # 3: Compute arrays and counts for each category
    stats = {}
    for name, g in groups.items():
        stats[name] = {
            'exp_H':  g['exp_H'].to_numpy(),
            'af_H':   g['af_H'].to_numpy(),
            'exp_BE': g['exp_BE'].to_numpy(),
            'af_BE':  g['af_BE'].to_numpy()
        }

    # 4: Create 3x4 subplot grid for H and BE comparisons with counts
    fig, axs = plt.subplots(3, 4, figsize=(14, 10))
    for i, (name, s) in enumerate(stats.items()):
        # H: scatter + diagonal + bar
        ax_sc, ax_bar = axs[i, 0], axs[i, 1]
        ax_sc.scatter(s['exp_H'], s['af_H'], s=20, alpha=0.6)
        m0, m1 = min(s['exp_H'].min(), s['af_H'].min()), max(s['exp_H'].max(), s['af_H'].max())
        ax_sc.plot([m0, m1], [m0, m1], 'r--')
        ax_sc.set(title=f"{name} Helix (H)", xlabel="Experimental", ylabel="AF")
        counts = [(s['af_H'] > s['exp_H']).sum(), (s['af_H'] < s['exp_H']).sum(), (s['af_H'] == s['exp_H']).sum()]
        ax_bar.bar(['>', '<', '='], counts)
        ax_bar.set_ylabel("Count")

        # BE: scatter + diagonal + bar
        ax_sc, ax_bar = axs[i, 2], axs[i, 3]
        ax_sc.scatter(s['exp_BE'], s['af_BE'], s=20, alpha=0.6)
        m0, m1 = min(s['exp_BE'].min(), s['af_BE'].min()), max(s['exp_BE'].max(), s['af_BE'].max())
        ax_sc.plot([m0, m1], [m0, m1], 'r--')
        ax_sc.set(title=f"{name} Beta+Extended (BE)", xlabel="Experimental", ylabel="AF")
        counts = [(s['af_BE'] > s['exp_BE']).sum(), (s['af_BE'] < s['exp_BE']).sum(), (s['af_BE'] == s['exp_BE']).sum()]
        ax_bar.bar(['>', '<', '='], counts)
        ax_bar.set_ylabel("Count")

    # 5: Final adjustments and save
    fig.suptitle("Experimental vs Predicted Structure Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig("exp_vs_af_ss.pdf")

if __name__ == "__main__":
    main()