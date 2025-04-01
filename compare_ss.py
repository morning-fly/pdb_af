#!/usr/bin/env python3

import requests
import argparse
import tempfile
import shutil
from Bio.PDB import PDBParser, DSSP, PDBIO, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List, Any
from loguru import logger
from multiprocessing import Pool

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
    logger.add(log, rotation="1 MB", compression="zip")    
    
    if not args.csv:
        csv_path = Path(args.csv)
        all_chain_info_df = pd.read_csv(csv_path)
    else:
        logger.info(f"Started getting info from PDB")
        pdb_info_df = get_all_pdb_info()
        all_chain_info_df = split_chain_in_dataframe(pdb_info_df=pdb_info_df)
        
    remove_dup_df = all_chain_info_df.sort_values("RESOLUTION").drop_duplicates(subset="UNIPROT ID", keep="first")
    logger.info(f"Finished getting info from PDB")
    
    logger.info(f"Started checking local database")
    missing_pdbs = check_local_pdb_db(local_db_path=pdb_dir_path, pdb_info_df=pdb_info_df)
    if missing_pdbs:
        for pdb_id in missing_pdbs:
            logger.info(f"Download: {pdb_id}")
            download_pdb(local_db_path=pdb_dir_path, pdb_id=pdb_id)
    logger.info(f"Finished checking local database")
    
    results = []
    parser = PDBParser(QUIET=True)
    for _, row in remove_dup_df.iterrows():
        pdb_id = str(row["IDCODE"])
        chain_id = str(row["CHAIN"])
        uniprot_id = str(row["UNIPROT ID"])
        
        exp_pdb = get_structure_in_dir(directory=pdb_dir_path, name=pdb_id)
        exp_structure = parser.get_structure('exp', exp_pdb)
        exp_chain = exp_structure[0][chain_id]
        
        download_af_structure(out_dir_path=af_dir_path, uniprot_id=uniprot_id)
        af_pdb = get_structure_in_dir(directory=af_dir_path, name=uniprot_id)
        af_structure = parser.get_structure('af', af_pdb)
        af_chain = af_structure[0]["A"]
        
        trimmed_residues = trim_af_structure(exp_chain, af_chain)
        trimmed_af_chain = create_chain_from_residues(trimmed_residues, chain_id)
        
        exp_ss = analyze_ss_from_chain(exp_chain, pdb_id, "exp")
        af_ss = analyze_ss_from_chain(trimmed_af_chain, pdb_id, "af")

        result = {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'uniprot_id': uniprot_id,
            'exp_secondary': exp_ss,
            'af_secondary': af_ss
        }
        results.append(result)
        logger.info(f"Analysis of {pdb_id}_{chain_id} is completed")
        
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

    Raises:
        Exception: If the HTTP request fails with a status code other than 200.
    """
    local_db_path = Path(local_db_path)
    local_db_path.mkdir(parents=True, exist_ok=True)
    download_path = local_db_path / f"{pdb_id}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    

    response = requests.get(url)    
    if response.status_code == 200:
        with open(download_path, "w", encoding=response.encoding if response.encoding else "utf-8") as f:
            f.write(response.text)
    else:
        raise Exception(f"Failed to download PDB file for {pdb_id}. HTTP Status Code: {response.status_code}")
    
def download_af_structure(out_dir_path: Union[str, Path], uniprot_id: str) -> None:
    """
    Download an AlphaFold structure file from the EBI AlphaFold website and save it in the specified directory.
    
    Args:
        out_dir_path (Union[str, Path]): The directory where the AlphaFold structure file will be saved.
        uniprot_id (str): The UniProt ID of the protein structure.
        
    Raises:
        Exception: If the HTTP request fails (status code is not 200).
    """
    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    download_path = out_dir_path / f"{uniprot_id}.pdb"
    
    if not download_path.exists():
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            with open(download_path, "w", encoding=response.encoding if response.encoding else "utf-8") as f:
                f.write(response.text)
        else:
            raise Exception(f"Failed to download AF structure file for {uniprot_id}. HTTP Status Code: {response.status_code}")
    
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
        dssp = DSSP(model, temp_file, dssp="dssp")
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

def trim_af_structure(exp_chain: Chain, af_chain: Chain) -> List[Residue]:
    """
    Trim the AlphaFold chain by retaining only those residues that are present in the experimental chain.
    
    Args:
        exp_chain (Chain): The experimental chain containing the reference residues.
        af_chain (Chain): The AlphaFold chain to be trimmed.
    
    Returns:
        List[Residue]: A list of residues from the AlphaFold chain that have matching residue IDs in the experimental chain.
    """
    exp_res_ids = [res.id for res in exp_chain]
    trimmed_residues = [res for res in af_chain if res.id in exp_res_ids]
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