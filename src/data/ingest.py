"""Data ingestion module for MIT-BIH Arrhythmia Database.

Downloads the 44 patient records defined by the de Chazal et al. (2004) protocol,
explicitly excluding pacemaker records (102, 104, 107, 217).
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import wfdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")

# de Chazal et al. (2004) inter-patient split (44 records total)
DE_CHAZAL_DS1: list[str] = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]

DE_CHAZAL_DS2: list[str] = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]

EXCLUDED_PACEMAKER_RECORDS: list[str] = ["102", "104", "107", "217"]

ALL_CHAZAL_RECORDS: list[str] = sorted(DE_CHAZAL_DS1 + DE_CHAZAL_DS2)


def is_record_downloaded(raw_dir: Path, record_id: str) -> bool:
    """Checks if a record's .dat, .hea, and .atr files exist in raw_dir."""
    dat_file = raw_dir / f"{record_id}.dat"
    hea_file = raw_dir / f"{record_id}.hea"
    atr_file = raw_dir / f"{record_id}.atr"
    return dat_file.is_file() and hea_file.is_file() and atr_file.is_file()


def download_mitbih_records(
    records: Sequence[str],
    raw_dir: Path,
    db_name: str = "mitdb",
) -> list[str]:
    """Downloads missing MIT-BIH records from PhysioNet.

    Args:
        records: List of record IDs to download.
        raw_dir: Destination directory for raw data.
        db_name: Database name on PhysioNet ('mitdb').

    Returns:
        List of newly downloaded records.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    valid_records = [r for r in records if r not in EXCLUDED_PACEMAKER_RECORDS]
    missing_records = [r for r in valid_records if not is_record_downloaded(raw_dir, r)]

    if not missing_records:
        logger.info(
            "All %d records are already present in '%s'. Skipping download.",
            len(valid_records),
            raw_dir,
        )
        return []

    logger.info(
        "Found %d existing records. Downloading %d missing records from PhysioNet '%s' into '%s'...",
        len(valid_records) - len(missing_records),
        len(missing_records),
        db_name,
        raw_dir,
    )

    wfdb.dl_database(
        db_name=db_name,
        dl_dir=str(raw_dir),
        records=missing_records,
        keep_subdirs=False,
        overwrite=False,
    )

    downloaded = [r for r in missing_records if is_record_downloaded(raw_dir, r)]
    logger.info("Successfully verified %d downloaded records.", len(downloaded))
    return downloaded


def main() -> None:
    """Entrypoint for data ingestion stage."""
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"

    logger.info("Starting MIT-BIH ingestion pipeline.")
    logger.info("Target raw directory: %s", raw_dir)
    logger.info(
        "Total de Chazal records to process: %d (DS1: %d, DS2: %d)",
        len(ALL_CHAZAL_RECORDS),
        len(DE_CHAZAL_DS1),
        len(DE_CHAZAL_DS2),
    )
    logger.info("Explicitly excluded pacemaker records: %s", EXCLUDED_PACEMAKER_RECORDS)

    download_mitbih_records(
        records=ALL_CHAZAL_RECORDS,
        raw_dir=raw_dir,
        db_name="mitdb",
    )
    logger.info("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()
