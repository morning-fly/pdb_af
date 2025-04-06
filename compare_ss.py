#!/usr/bin/env python3

import requests
import argparse
import tempfile
import shutil
from functools import partial
from Bio.PDB import PDBParser, DSSP, PDBIO, Select, MMCIFParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
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

def main():
    parser = argparse.ArgumentParser(
        description="Compare the structures deposited in PDB with the predicted structure in AF"
        )
    parser.add_argument(
        "-p",
        "--pdb",
        help="directory containing PDB deposited structures",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--af",
        help="directory containing AF predicted structures",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--csv",
        help="csv file containing pdb information created in advance",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    pdb_dir_path = Path(args.pdb)
    af_dir_path = Path(args.af)
    
    log = Path.cwd() / f"{Path(__file__).stem}.log"
    logger.add(log)    
    
    logger.info(f"Started getting info from PDB")
    pdb_info_df = get_all_pdb_info()
    
    if args.csv:
        csv_path = Path(args.csv)
        all_chain_info_df = pd.read_csv(csv_path)
    else:
        all_chain_info_df = split_chain_in_dataframe(pdb_info_df=pdb_info_df)
    
    filtered_df = all_chain_info_df[all_chain_info_df['EXPERIMENT TYPE (IF NOT X-RAY)'].isin(['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'])]
    filtered_df = filtered_df.copy()
    filtered_df["RESOLUTION"] = filtered_df["RESOLUTION"].astype("float")
    remove_dup_df = filtered_df.sort_values("RESOLUTION").drop_duplicates(subset="UNIPROT ID", keep="first")
    remove_dup_df.to_csv("remove_duplicated_info.csv")
    logger.info(f"Finished getting info from PDB")
    
    remove_dup_rows = remove_dup_df.to_dict(orient="records")
    partial_main = partial(main_flow, pdb_dir_path, af_dir_path)
    with Pool(processes=16) as pool:
        results = pool.map(partial_main,  remove_dup_rows)
    
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)
    results_df.to_csv("ss_comparison_allchains.csv", index=False)

    exp_H = []
    af_H = []
    labels = []
    for res in results:
        exp_h = res['exp_secondary'].get('H', 0)
        af_h = res['af_secondary'].get('H', 0)
        exp_H.append(exp_h)
        af_H.append(af_h)
        labels.append(f"{res['pdb_id']}_{res['chain_id']}")
    
    x = range(len(labels))
    plt.figure(figsize=(12,6))
    plt.bar(x, exp_H, width=0.4, label='Exp', align='center')
    plt.bar([i+0.4 for i in x], af_H, width=0.4, label='AlphaFold', align='center')
    plt.ylabel("alpha-helix (%)")
    plt.legend()
    plt.title("exp vs af")
    plt.tight_layout()
    plt.savefig("exp_vs_af_helix.pdf")

def main_flow(pdb_dir_path: Union[str, Path], af_dir_path: Union[str, Path], row_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Main processing flow for comparing experimental and AlphaFold structures.
    
    This function performs the following steps:
      1. Download the experimental mmCIF file from RCSB using the provided PDB ID.
      2. Convert the mmCIF file to PDB format by chain if it has not been converted already.
      3. Load the experimental structure and extract the specified chain.
      4. Download the corresponding AlphaFold structure using the UniProt ID.
      5. Load the AlphaFold structure and extract the chain (forced as chain "A").
      6. Trim the AlphaFold structure to align with the experimental chain.
      7. Analyze the secondary structure for both experimental and trimmed AlphaFold chains.
      8. Return a dictionary containing the PDB ID, chain ID, UniProt ID, and secondary structure analyses.
      
    Args:
        pdb_dir_path (Union[str, Path]): Directory path where experimental files are stored.
        af_dir_path (Union[str, Path]): Directory path where AlphaFold files are stored.
        row_dict (Dict[str, Any]): A dictionary containing keys "IDCODE", "CHAIN", and "UNIPROT ID".
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary with analysis results if successful; otherwise, None.
    """
    # Initialize the PDB parser (QUIET mode suppresses warnings)
    parser = PDBParser(QUIET=True)
    pdb_id = str(row_dict["IDCODE"])
    chain_id = str(row_dict["CHAIN"])
    uniprot_id = str(row_dict["UNIPROT ID"])
    logger.info(f"Processing {pdb_id}_{chain_id}")

    try:
        # 1. Download the experimental mmCIF file
        exp_cif = download_cif(local_db_path=pdb_dir_path, pdb_id=pdb_id)
        if exp_cif is None:
            logger.warning(f"{pdb_id}_{chain_id} is skipped (mmCIF file could not be downloaded)")
            return None

        # 2. Convert mmCIF to PDB by chain if not already done
        cif_dir = exp_cif.parent
        if not any(cif_dir.glob(f"{pdb_id}_*.pdb")):
            cif2pdb_bychain(cif_file=exp_cif)

        # 3. Retrieve the experimental PDB file by name
        pdb_name = f"{pdb_id}_{chain_id}"
        exp_pdb = get_structure_in_dir(directory=cif_dir, name=pdb_name)
        if exp_pdb is None:
            logger.warning(f"{pdb_id}_{chain_id} is skipped (PDB file does not exist)")
            return None

        exp_structure = parser.get_structure('exp', exp_pdb)
        try:
            # Here, we assume the chain in the experimental structure is forced to "A"
            exp_chain = exp_structure[0]["A"]
        except KeyError:
            logger.warning(f"{pdb_id}_{chain_id} is skipped (chain {chain_id} not found in experimental structure)")
            return None

        # 4. Download the corresponding AlphaFold structure file
        af_pdb = download_af_structure(out_dir_path=af_dir_path, uniprot_id=uniprot_id)
        if af_pdb is None:
            logger.warning(f"{pdb_id}_{chain_id} is skipped (AlphaFold file does not exist)")
            return None

        # 5. Load the AlphaFold structure and extract chain "A"
        af_structure = parser.get_structure('af', af_pdb)
        try:
            af_chain = af_structure[0]["A"]
        except KeyError:
            logger.warning(f"{pdb_id}_{chain_id} is skipped (chain A not found in AlphaFold structure)")
            return None

        # 6. Trim the AlphaFold structure to align with the experimental chain
        trimmed_residues = trim_af_structure_by_alignment(exp_chain, af_chain)
        trimmed_af_chain = create_chain_from_residues(trimmed_residues, "A")

        # 7. Analyze secondary structure for both experimental and AlphaFold (trimmed) chains
        exp_ss = analyze_ss_from_chain(exp_chain, pdb_id, "exp")
        af_ss = analyze_ss_from_chain(trimmed_af_chain, pdb_id, "af")

        # 8. Prepare and return the result dictionary
        result = {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'uniprot_id': uniprot_id,
            'exp_secondary': exp_ss,
            'af_secondary': af_ss
        }
        logger.info(f"Completed analysis for {pdb_id}_{chain_id}")
    
    except Exception as e:
        logger.error(f"Error processing {pdb_id}_{chain_id}: {e}")
        return None

    return result

def get_all_pdb_info() -> pd.DataFrame:
    """Retrieve all deposited structure information from the PDB.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved data.
    """
    url = "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
    response = requests.get(url)
    
    # Raise an exception if the HTTP request was not successful
    if response.status_code != 200:
        raise Exception(f"Failed to acquire PDB information, status code: {response.status_code}")
    
    lines = response.text.splitlines()
    
    columns = lines[0].split(", ")
    records = [line.split("\t") for line in lines[2:] if line.strip()]
    
    df = pd.DataFrame(records, columns=columns)
    return df
        
def check_local_pdb_db(local_db_path: Union[str, Path], pdb_info_df: pd.DataFrame) -> list:
    """
    Compare the local PDB files with deposited PDB IDs from the DataFrame
    and return a list of IDs that are deposited remotely but not downloaded locally.

    Args:
        local_db_path (Union[str, Path]): The directory path containing local PDB files.
        pdb_info_df (pd.DataFrame): DataFrame with deposited PDB information. It must contain an 'IDCODE' column.

    Returns:
        list: A list of PDB IDs that are present in the remote deposit but missing in the local database.
    """
    local_db_path = Path(local_db_path)

    downloaded_pdbs = [p.stem for p in local_db_path.glob("*.pdb")]
    deposited_pdbs = list(pdb_info_df["IDCODE"])

    missing_pdbs = [pdb_id for pdb_id in deposited_pdbs if pdb_id not in downloaded_pdbs]
    
    return missing_pdbs

def download_pdb(local_db_path: Union[str, Path], pdb_id: str) -> None:
    """
    Download a PDB file from the RCSB website and save it to the specified local directory.

    Args:
        local_db_path (Union[str, Path]): The directory path where the PDB file will be stored.
        pdb_id (str): The PDB ID of the structure to download.

    Returns:
        None. In case of errors, messages are logged instead of raising exceptions.
    """
    local_db_path = Path(local_db_path)
    local_db_path.mkdir(parents=True, exist_ok=True)
    download_path = local_db_path / f"{pdb_id}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download PDB file for {pdb_id}. Error: {e}")
        return
    
    try:
        encoding = response.encoding if response.encoding else "utf-8"
        with open(download_path, "w", encoding=encoding) as f:
            f.write(response.text)
        logger.info(f"Successfully downloaded PDB file for {pdb_id} to {download_path}")
    except Exception as e:
        logger.error(f"Failed to write PDB file for {pdb_id}. Error: {e}")

def cif2pdb_bychain(cif_file: Union[str, Path]) -> None:
    """
    Convert each chain from an mmCIF file into a separate PDB file.
    In each resulting PDB file, the chain ID is forced to "A".
    The output filename includes the original chain ID for reference.
    
    Args:
        cif_file (Union[str, Path]): Path to the input mmCIF file.
    """
    warnings.simplefilter("ignore", PDBConstructionWarning)
    cif_file_path = Path(cif_file)
    if not cif_file_path.exists():
        raise FileNotFoundError(f"The file {cif_file_path} does not exist.")
    
    pdb_id = cif_file_path.stem
    parser = MMCIFParser()
    structure_id = pdb_id
    
    # Parse the mmCIF file to get the structure
    structure = parser.get_structure(structure_id, cif_file_path)
    io = PDBIO()
    
    # Iterate over each model and chain in the structure
    for model in structure:
        for chain in model:
            # Create a new structure and model for each chain so it can be saved as an individual PDB file
            new_structure = Structure(structure_id)
            new_model = Model(0)
            
            # Copy the chain and set its ID to "A"
            new_chain = chain.copy()
            new_chain.id = "A"[0]
            
            new_model.add(new_chain)
            new_structure.add(new_model)
            
            # Create an output filename that includes the original chain ID for reference
            pdb_filename = cif_file_path.parent / f"{structure_id}_{chain.id}.pdb"
            io.set_structure(new_structure)
            io.save(str(pdb_filename))  # Convert Path to string to avoid 'write' attribute error
            print(f"{pdb_id} Chain {chain.id} saved as {pdb_filename}")
   
def download_cif(local_db_path: Union[str, Path], pdb_id: str) -> Optional[Path]:
    """
    Download an mmCIF file from RCSB and save it to the specified local directory.

    Args:
        local_db_path (Union[str, Path]): The directory where the mmCIF file will be saved.
        pdb_id (str): The PDB ID of the structure to download.

    Returns:
        Optional[Path]: The path to the saved mmCIF file if successful; otherwise, None.
    """
    # Convert local_db_path to a Path object and create the directory if it doesn't exist
    local_db_path = Path(local_db_path)
    local_db_path.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for the specific pdb_id
    pdb_dir_path = local_db_path / pdb_id
    pdb_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define the download path for the mmCIF file
    download_path = pdb_dir_path / f"{pdb_id}.cif"
    
    # If the file already exists, return its path
    if download_path.is_file():
        return download_path

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download mmCIF file for {pdb_id}. Error: {e}")
        return None
    
    try:
        encoding = response.encoding if response.encoding else "utf-8"
        with open(download_path, "w", encoding=encoding) as f:
            f.write(response.text)
        logger.info(f"Successfully downloaded mmCIF file for {pdb_id} to {download_path}")
        return download_path
    except Exception as e:
        logger.error(f"Failed to write mmCIF file for {pdb_id}. Error: {e}")
        return None

def download_af_structure(out_dir_path: Union[str, Path], uniprot_id: str) -> Optional[Path]:
    """
    Download an AlphaFold structure file from the EBI AlphaFold website and save it in the specified directory.
    
    Args:
        out_dir_path (Union[str, Path]): The directory where the AlphaFold structure file will be saved.
        uniprot_id (str): The UniProt ID of the protein structure.
        
    Returns:
        Optional[Path]: The path to the saved structure file if successful; otherwise, None.
    """
    # Convert out_dir_path to a Path object and ensure the directory exists
    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define the download path for the PDB file
    download_path = out_dir_path / f"{uniprot_id}.pdb"
    
    # If the file does not already exist, attempt to download it
    if not download_path.exists():
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            # Determine the encoding, default to "utf-8" if not provided
            encoding = response.encoding if response.encoding else "utf-8"
            with open(download_path, "w", encoding=encoding) as f:
                f.write(response.text)
            return download_path
        else:
            logger.error(f"Failed to download AF structure file for {uniprot_id}. HTTP Status Code: {response.status_code}")
            return None
    # Return the download path if the file already exists
    return download_path
    
def map_pdb_to_uniprot_allchains(pdb_id: str) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Map PDB chain IDs to corresponding UniProt IDs and their amino acid sequence lengths.

    Args:
        pdb_id (str): The PDB ID for which to retrieve mappings.

    Returns:
        Optional[Dict[str, Dict[str, int]]]: A dictionary where each key is a chain ID and the value is a dictionary 
        mapping UniProt IDs to the amino acid sequence length. Returns None if the mapping retrieval fails.
    """
    # Construct the URL using the lowercased PDB ID
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error retrieving mapping for {pdb_id}. HTTP Status Code: {response.status_code}")
        return None
    
    data = response.json()
    mapping = data.get(pdb_id.lower(), {})
    if not mapping:
        print(f"No mapping data found for {pdb_id}")
        return None
    
    uniprot_mappings = mapping.get("UniProt", {})
    if not uniprot_mappings:
        print(f"No UniProt mapping found for {pdb_id}")
        return None

    chain_to_uniprot: Dict[str, Dict[str, int]] = {}
    
    for uniprot_id, mapping_info in uniprot_mappings.items():
        mappings_list = mapping_info.get("mappings", [])
        for m in mappings_list:
            chain_id = m.get("chain_id")
            # Calculate the amino acid length from unp_start and unp_end
            try:
                unp_start = int(m.get("unp_start"))
                unp_end = int(m.get("unp_end"))
                aa_length = unp_end - unp_start + 1
            except (TypeError, ValueError):
                # Skip mapping if start/end positions are invalid
                continue
            
            if chain_id:
                if chain_id not in chain_to_uniprot:
                    chain_to_uniprot[chain_id] = {}
                # Add mapping if it doesn't already exist
                if uniprot_id not in chain_to_uniprot[chain_id]:
                    chain_to_uniprot[chain_id][uniprot_id] = aa_length
                    
    return chain_to_uniprot

def convert_row(row_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single row (as a dictionary) into multiple rows,
    one per chain mapping.
    """
    pdb_id = str(row_dict["IDCODE"])
    mapping_dict = map_pdb_to_uniprot_allchains(pdb_id=pdb_id)
    results = []
    if mapping_dict:
        for chain, uni_dict in mapping_dict.items():
            new_row = row_dict.copy()
            # Select the UniProt ID with the highest mapping value
            main_uni_id = max(uni_dict, key=uni_dict.get)
            new_row["CHAIN"] = chain
            new_row["UNIPROT ID"] = main_uni_id
            logger.info(f"PDB {pdb_id} chain {chain} converted to UniProt ID {main_uni_id}")
            results.append(new_row)
    return results

def split_chain_in_dataframe(pdb_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the input DataFrame by splitting each PDB entry into separate rows
    for each chain, and add the corresponding UniProt ID with the longest mapping.
    
    Args:
        pdb_info_df (pd.DataFrame): Original DataFrame with at least an "IDCODE" column.
    
    Returns:
        pd.DataFrame: Expanded DataFrame with additional "CHAIN" and "UNIPROT ID" columns.
    """

    # Convert DataFrame rows to a list of dictionaries using itertuples for speed
    pdb_info_rows = pdb_info_df.to_dict(orient="records")
    
    with Pool(processes=16) as pool:
        results = pool.map(convert_row, pdb_info_rows)
    
    # Flatten the list of lists into a single list of dictionaries
    flat_results = [item for sublist in results for item in sublist]
    out_df = pd.DataFrame(flat_results)
    
    out_df.to_csv("all_chain_pdb_info.csv", index=False)
    
    return out_df

def get_structure_in_dir(directory: Union[str, Path], name: str) -> Optional[Path]:
    """
    Retrieve the path of a PDB structure file with the given name from the specified directory.

    Args:
        directory (Union[str, Path]): The directory where the PDB file is located.
        name (str): The base name of the PDB file (without the .pdb extension).

    Returns:
        Optional[Path]: A Path object for the file if found, otherwise None.
    """
    directory = Path(directory)
    return next(directory.glob(f"{name}.pdb"), None)

def add_dummy_cryst1(pdb_path: Union[str, Path]) -> None:
    """
    Add a dummy CRYST1 record at the beginning of the PDB file if it is missing.
    This is useful to avoid errors during DSSP analysis when a CRYST1 record is absent.

    Args:
        pdb_path (Union[str, Path]): The path to the PDB file.
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
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
def write_chain_to_temp(chain: Chain, pdb_id: str, source_tag: str) -> Tuple[Path, Path]:
    """
    Write the given chain to a temporary PDB file and add a dummy CRYST1 record to avoid DSSP errors.
    
    This function creates a temporary directory, writes the provided chain as a PDB file with a filename
    based on the pdb_id, chain id, and source_tag, then adds a dummy CRYST1 record at the top of the file.
    
    Args:
        chain (Chain): A Bio.PDB Chain object representing the chain to be written.
        pdb_id (str): The PDB identifier.
        source_tag (str): A tag indicating the source (e.g., "exp" for experimental, "af" for AlphaFold).

    Returns:
        Tuple[Path, Path]: A tuple containing the path to the temporary PDB file and the temporary directory.
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
    Analyze the secondary structure of a given chain using DSSP.
    
    This function writes the specified chain to a temporary PDB file,
    adds a dummy CRYST1 record if needed, and runs DSSP to determine the secondary structure.
    It returns a dictionary mapping each secondary structure type (e.g., 'H', 'E', etc.) 
    to its percentage in the chain.
    
    Args:
        chain (Chain): A Bio.PDB Chain object representing the chain to analyze.
        pdb_id (str): The PDB identifier.
        source_tag (str): A tag indicating the source (e.g., "exp" for experimental, "af" for AlphaFold).
    
    Returns:
        Dict[str, float]: A dictionary mapping secondary structure symbols to their percentages.
                          Returns an empty dictionary if DSSP processing fails.
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
        return {}
    
    counts: Dict[str, int] = {}
    total = 0
    for key in dssp.keys():
        if key[0] == chain.id:
            total += 1
            ss = dssp[key][2]
            counts[ss] = counts.get(ss, 0) + 1
            
    percentages: Dict[str, float] = {ss: (cnt / total) * 100 for ss, cnt in counts.items()} if total > 0 else {}
    shutil.rmtree(tmp_dir)
    return percentages

def get_chain_sequence_and_residues(chain: Chain) -> Tuple[str, List[Residue]]:
    """
    Extracts the concatenated polypeptide sequence and the corresponding list of Residue objects
    from the given chain.

    Args:
        chain (Chain): The target chain object.

    Returns:
        Tuple[str, List[Residue]]: The concatenated amino acid sequence and the corresponding list of Residue objects.
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
    Performs a global alignment between the experimental chain (exp_chain) and the AlphaFold chain (af_chain)
    and extracts only the AlphaFold residues corresponding to regions present in the experimental structure.

    Note: This function assumes the alignment output is in blocks separated by blank lines,
    with each block containing 3 lines:
      - Line 1: target (experimental) with a header (e.g., "target   0 ...")
      - Line 2: match line (ignored)
      - Line 3: query (AlphaFold) with a header (e.g., "query    0 ...")
    Blocks that contain trailing numeric tokens (as in the final block) are skipped.

    Args:
        exp_chain (Chain): The experimental chain.
        af_chain (Chain): The AlphaFold predicted chain.

    Returns:
        List[Residue]: A list of AlphaFold Residue objects corresponding to the regions present in the experimental structure.
    """
    # Suppress PDB construction warnings
    warnings.simplefilter("ignore", PDBConstructionWarning)

    # Extract sequences and residue lists from both chains.
    # Assumes get_chain_sequence_and_residues() returns a tuple (sequence, residue_list).
    exp_seq, _ = get_chain_sequence_and_residues(exp_chain)
    af_seq, af_residues = get_chain_sequence_and_residues(af_chain)

    # Perform global alignment using PairwiseAligner.
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1

    alignments = aligner.align(exp_seq, af_seq)
    best_alignment = alignments[0]

    # Convert the alignment to a string and split it into lines.
    alignment_str = str(best_alignment)
    lines = alignment_str.splitlines()

    aligned_exp_full = ""
    aligned_af_full = ""
    block = []  # Temporary container for lines within each alignment block

    # Process alignment blocks separated by empty lines.
    for line in lines:
        if line.strip() == "":
            # Process the current block if it contains at least 3 lines.
            if len(block) >= 3:
                target_line = block[0]
                query_line = block[2]
                # Split each line into up to 3 parts: header, position, and aligned sequence.
                target_parts = target_line.split(maxsplit=2)
                query_parts = query_line.split(maxsplit=2)
                if len(target_parts) < 3 or len(query_parts) < 3:
                    logger.warning("Incomplete alignment block encountered and skipped.")
                else:
                    target_seq = target_parts[2].strip()
                    query_seq = query_parts[2].strip()
                    # Skip the block if the last token in the target aligned sequence is numeric.
                    if target_seq.split() and target_seq.split()[-1].isdigit():
                        logger.debug("Skipping block with trailing numeric tokens.")
                    else:
                        aligned_exp_full += target_seq
                        aligned_af_full += query_seq
            block = []  # Reset the block for the next alignment segment.
        else:
            block.append(line)

    # Process the final block if it is not terminated by an empty line.
    if block and len(block) >= 3:
        target_line = block[0]
        query_line = block[2]
        target_parts = target_line.split(maxsplit=2)
        query_parts = query_line.split(maxsplit=2)
        if len(target_parts) < 3 or len(query_parts) < 3:
            logger.warning("Incomplete final alignment block encountered and skipped.")
        else:
            target_seq = target_parts[2].strip()
            query_seq = query_parts[2].strip()
            if target_seq.split() and target_seq.split()[-1].isdigit():
                logger.debug("Skipping final block with trailing numeric tokens.")
            else:
                aligned_exp_full += target_seq
                aligned_af_full += query_seq

    # Ensure the aligned sequences have the same length.
    if len(aligned_exp_full) != len(aligned_af_full):
        raise ValueError("Aligned sequences have different lengths.")

    # Determine which positions in the aligned AlphaFold sequence correspond to actual residues.
    af_counter = 0
    indices_to_keep = []  # List of indices in af_residues to keep.
    for pos in range(len(aligned_af_full)):
        if aligned_af_full[pos] != "-":
            # Consider the position only if the experimental aligned sequence also has a residue.
            if aligned_exp_full[pos] != "-":
                indices_to_keep.append(af_counter)
            af_counter += 1  # Increment the counter only when a residue (non-gap) is encountered.
        # Do not increment af_counter for gaps in the AlphaFold alignment.

    # Extract Residue objects from af_residues corresponding to the selected indices.
    trimmed_residues = [af_residues[i] for i in indices_to_keep]
    return trimmed_residues

def create_chain_from_residues(residues: List[Residue], chain_id: str) -> Chain:
    """
    Create a new Chain object by adding the provided residues.

    Args:
        residues (List[Residue]): A list of Residue objects to be included in the new chain.
        chain_id (str): The identifier for the new chain.

    Returns:
        Chain: A new Chain object containing all the given residues.
    """
    new_chain = Chain(chain_id)
    for res in residues:
        new_chain.add(res)
    return new_chain

if __name__ == "__main__":
    main()