#!/usr/bin/env python3
"""
build_database.py

Build local PDB and AlphaFold database by downloading structures and mapping to UniProt IDs.
"""

import os
import argparse
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException
from loguru import logger
from tqdm import tqdm


def get_request_session() -> requests.Session:
    """Create an HTTP session with retry logic.

    Returns:
        requests.Session: Configured HTTP session with retry adapter.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_all_pdb_info(session: requests.Session) -> pd.DataFrame:
    """Retrieve all PDB entries index.

    Args:
        session (requests.Session): HTTP session to use for the request.

    Returns:
        pd.DataFrame: DataFrame containing PDB index entries.
    """
    url = "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    columns = lines[0].split(", ")
    records = [line.split("\t") for line in lines[2:] if line]
    return pd.DataFrame(records, columns=columns)


def map_pdb_to_uniprot_allchains(session: requests.Session, pdb_id: str) -> Optional[Dict[str, Dict[str, int]]]:
    """Map a PDB entry to UniProt IDs for all chains.

    Args:
        session (requests.Session): HTTP session for API requests.
        pdb_id (str): PDB ID code (case-insensitive).

    Returns:
        Optional[Dict[str, Dict[str, int]]]: Mapping of chain IDs to UniProt IDs and sequence lengths,
            or None if mapping fails or is unavailable.
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Cannot map PDB {pdb_id}: {e}")
        return None

    mapping = data.get(pdb_id.lower(), {}).get("UniProt", {})
    if not mapping:
        logger.warning(f"No UniProt mapping for {pdb_id}")
        return None

    chain_to_uniprot: Dict[str, Dict[str, int]] = {}
    for uni_id, info in mapping.items():
        for m in info.get("mappings", []):
            chain = m.get("chain_id")
            try:
                length = int(m["unp_end"]) - int(m["unp_start"]) + 1
            except (TypeError, KeyError, ValueError):
                continue
            chain_to_uniprot.setdefault(chain, {})[uni_id] = length
    return chain_to_uniprot


def convert_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map a single PDB index row to its UniProt chain(s).

    Args:
        row (Dict[str, Any]): Dictionary representing a row from the PDB index DataFrame.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the original row data,
            chain ID, UniProt ID, and amino acid length for the mapping.
    """
    pdb_id = str(row["IDCODE"])
    session = get_request_session()
    mapping = map_pdb_to_uniprot_allchains(session, pdb_id)
    results: List[Dict[str, Any]] = []
    if mapping:
        for chain, uni in mapping.items():
            uni_id = max(uni, key=uni.get)
            new_row = {
                **row,
                "CHAIN": chain,
                "UNIPROT_ID": uni_id,
                "AA_LENGTH": uni[uni_id]
            }
            logger.info(f"{pdb_id} chain {chain} -> {uni_id}")
            results.append(new_row)
    return results


def split_chain_dataframe(pdb_df: pd.DataFrame, processes: int) -> pd.DataFrame:
    """Expand each PDB entry into individual chain-level entries with UniProt mappings.

    Args:
        pdb_df (pd.DataFrame): DataFrame of PDB index entries.
        processes (int): Number of parallel processes to use.

    Returns:
        pd.DataFrame: DataFrame with one row per chain, including UniProt mapping data.
    """
    rows = pdb_df.to_dict(orient="records")
    mapped: List[List[Dict[str, Any]]] = []
    with Pool(processes=processes) as pool:
        for result in tqdm(
            pool.imap_unordered(convert_row, rows, chunksize=50),
            total=len(rows),
            desc="Mapping PDB → UniProt chains"
        ):
            mapped.append(result)
    # flatten
    flat = [item for sublist in mapped for item in sublist]
    df = pd.DataFrame(flat)
    
    return df


def download_cif(pdb_id: str, out_dir: Path) -> Optional[Path]:
    """Download mmCIF file for a given PDB ID.

    Args:
        pdb_id (str): PDB ID code (uppercase).
        out_dir (Path): Base output directory for downloading.

    Returns:
        Optional[Path]: Path to the downloaded mmCIF file, or None if download fails.
    """
    session = get_request_session()
    dest = out_dir / "PDB" / pdb_id
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / f"{pdb_id}.cif"
    if filepath.exists():
        return filepath

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        filepath.write_text(resp.text, encoding=resp.encoding or "utf-8")
        logger.info(f"Downloaded CIF for {pdb_id}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to download CIF for {pdb_id}: {e}")
        return None


def cif2pdb_by_chain(cif_path: Optional[Path]) -> None:
    """Split a multi-chain mmCIF file into individual PDB files for each chain.

    Args:
        cif_path (Optional[Path]): Path to the mmCIF file to process.

    Returns:
        None
    """
    if not cif_path:
        return

    warnings.simplefilter("ignore", PDBConstructionWarning)
    pdb_id = cif_path.stem
    parser = MMCIFParser()
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except PDBConstructionException as e:
        logger.warning(f"Skipping entire mmCIF {pdb_id}: parse error ({e})")
        return

    io = PDBIO()
    for model in structure:
        for chain in model:
            try:
                new_struct = Structure(structure.id)
                model0 = Model(0)
                chain_copy = chain.copy()
                chain_copy.id = "A"
                model0.add(chain_copy)
                new_struct.add(model0)

                out_path = cif_path.parent / f"{pdb_id}_{chain.id}.pdb"
                io.set_structure(new_struct)
                io.save(str(out_path))
                logger.info(f"Successfully saved {out_path}")
            except PDBConstructionException as e:
                logger.warning(f"Skipping chain {chain.id} of {pdb_id}: {e}")
                continue


def download_af_structure(uni_id: str, out_dir: Path) -> Optional[Path]:
    """Download AlphaFold predicted PDB model for a UniProt ID.

    Args:
        uni_id (str): UniProt accession ID.
        out_dir (Path): Base output directory for downloading.

    Returns:
        Optional[Path]: Path to the downloaded PDB file, or None if download fails.
    """
    session = get_request_session()
    dest = out_dir / "AlphaFold" / f"{uni_id}.pdb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uni_id}-F1-model_v4.pdb"
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding=resp.encoding or "utf-8")
        logger.info(f"Downloaded AlphaFold model for {uni_id}")
        return dest
    except Exception as e:
        logger.warning(f"Failed to download AlphaFold model for {uni_id}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDB/AF database")
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory"
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=cpu_count(),
        help="Number of processes to use"
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    logger.add(
        out / "build_database.log",
        format="{time:YYYY-MM-DD HH:mm:ss} [{process.name}/{process.id}] {level}: {message}",
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False
    )

    session = get_request_session()
    pdb_df = get_all_pdb_info(session)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", args.processes))
    chain_df = split_chain_dataframe(pdb_df, n_cpus)
    chain_df.to_csv(out / "all_chain_pdb_info.csv", index=False)

    # --- Download mmCIF with progress bar ---
    pdb_ids = chain_df["IDCODE"].unique().tolist()
    cif_paths: List[Path] = []
    with Pool(processes=n_cpus) as pool:
        for path in tqdm(pool.imap_unordered(partial(download_cif, out_dir=out), pdb_ids, chunksize=10),
                         total=len(pdb_ids), desc="Downloading mmCIF"):
            if path:
                cif_paths.append(path)

    # --- Split chains ---
    with Pool(processes=n_cpus) as pool:
        _ = list(tqdm(pool.imap_unordered(cif2pdb_by_chain, cif_paths),
                      total=len(cif_paths), desc="Splitting chains"))

    # --- Download AlphaFold models with progress bar ---
    uniprot_ids = chain_df["UNIPROT_ID"].unique().tolist()
    with Pool(processes=n_cpus) as pool:
        _ = list(tqdm(pool.imap_unordered(partial(download_af_structure, out_dir=out), uniprot_ids, chunksize=10),
                      total=len(uniprot_ids), desc="Downloading AF"))


if __name__ == "__main__":
    main()