import hashlib
import shutil
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from tabulate import tabulate
from tqdm import tqdm
from urllib3 import util
from loguru import logger

from atg.utils import thread_map_parallel

# ic.configureOutput(prefix=" -> ")


def download_url(input_file: Tuple[str, str, Path]) -> None:
    """Downloads a file from a URL with retry logic and progress bar.
    
    Parameters
    ----------
    input_file : Tuple[str, str, Path]
        Tuple containing (filename, url, output_directory)
    """
    fname, url, output_dir = input_file[0], input_file[1], input_file[2]
    
    session = requests.Session()
    retry = util.Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=["GET"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    
    try:
        r = session.get(url, stream=True, allow_redirects=True, timeout=30)
        r.raise_for_status()
        
        file_size = int(r.headers.get("Content-Length", 0))
        
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=fname) as r_raw:
            with open(output_dir / fname, "wb") as file:
                shutil.copyfileobj(r_raw, file)
                
    except requests.exceptions.Timeout:
        logger.error(f"Timeout downloading {fname}. Please retry later.")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error downloading {fname}: {e}")
        raise
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error downloading {fname}. Please check your internet connection.")
        raise
    except Exception as e:
        logger.error(f"Error downloading {fname}: {e}")
        raise


def fix_urls(urls: list) -> Dict[str, str]:
    urls_dict = {}
    urls = [x for x in urls if isinstance(x, str)]

    for url in urls:
        filename = url.split("/")[-1]
        if "://" in url:
            pass
        else:
            urls_dict[filename] = f"https://{url}"

    return urls_dict


def request_3_times():
    session = requests.Session()
    retry = util.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    return session


def request_get(api: str, pdict: str, fields: str, safe: str = ",") -> pd.DataFrame:
    url_api = f"https://www.ebi.ac.uk/ena/{api}"

    pdict["fields"] = fields

    params = [f"{k}={requests.utils.quote(v, safe=safe)}&" for k, v in pdict.items()]
    params_str = "".join(params)[:-1]

    try:
        r = request_3_times().get(url_api, params=params_str, timeout=20)

        r.raise_for_status()

        df = pd.read_csv(StringIO(r.content.decode("utf-8")), sep="\t")

        return df
    except requests.exceptions.Timeout:
        logger.error("Connection to the server has timed out. Please retry.")
        return None
    except requests.exceptions.HTTPError:
        logger.error("HTTPError: This is likely caused by an invalid search query")
        return None


def ena_fields(id_err: str, save: bool = True, fields: str = "") -> Dict[str, str]:
    if not fields:
        fields = (
            "study_accession,sample_accession,"
            "experiment_accession,run_accession,"
            "tax_id,scientific_name,fastq_ftp,fastq_md5"
        )

    params = {
        "accession": id_err,
        "result": "read_run",
        "fields": None,
        "format": "tsv",
        "download": "true",
    }

    df = request_get("portal/api/filereport", pdict=params, fields=fields)

    if save:
        df.to_csv(f"{id_err}.tsv", sep="\t", index=False)
        logger.info(f"ENA metadata saved as {id_err}.tsv")
    else:
        pass

    return df


def ena_urls(df: pd.DataFrame) -> Dict[str, str]:
    urls = df["fastq_ftp"].str.split(";").explode().tolist()

    return fix_urls(urls)


def md5_hash(filename, block_size=2**20):
    md5 = hashlib.md5()
    with open(filename, "rb") as file:
        while True:
            data = file.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def thread_map_urls(url_dict: Dict[str, str], outdir: Path, cpu: int) -> None:
    iter_url = [(k, v, outdir) for k, v in url_dict.items()]
    thread_map_parallel(download_url, iter_url, cpu)


def checksums(id_err: str, output_dir: Path, file_lst: List[str], threads: int) -> None:
    df = pd.read_csv(f"{id_err}.tsv", sep="\t")[["fastq_ftp", "fastq_md5"]]

    dfmd5 = pd.concat([df[x].str.split(";").explode() for x in df.columns], axis=1)
    dfmd5["file"] = dfmd5["fastq_ftp"].str.split("/").str[-1]

    chk = set(dfmd5["file"].to_list()) - set(file_lst)

    def compare_lists(df_md5: pd.DataFrame, test_list: List[str], outdir: Path, n_cpus: int) -> None:
        compare = df_md5["file"].isin(test_list)
        missing_files = fix_urls(df_md5[compare]["fastq_ftp"].to_list())
        thread_map_urls(missing_files, outdir, n_cpus)

    if len(chk) > 0:
        compare_lists(dfmd5, list(chk), output_dir, threads)

    urls_dict = dict(zip(dfmd5["file"], zip(dfmd5["fastq_md5"], dfmd5["fastq_ftp"])))
    md5_failed = [k for k, v in urls_dict.items() if v[0] != md5_hash(output_dir / k)]

    if compare_lists(dfmd5, md5_failed, output_dir, threads) is None:
        logger.info("All files are already downloaded")


def ena_search(
    query: str,
    domain: str = "project",
    safe: str = ",",
) -> pd.DataFrame:
    params = {"domain": domain, "query": query, "format": "tsv"}

    df = request_get("browser/api/tsv/textsearch", params, fields="all", safe=safe)

    if df.empty:
        logger.error("Check your query")

    return df


def ena_retrieve(keywords: str, save: bool, only_ids: bool):
    """
    Retrieve the ENA data for a given accession number.
    """
    df = ena_search(keywords, safe="")

    keywords = keywords.replace(" ", "_")

    if save and not only_ids:
        df.to_csv(f"{keywords}.tsv", sep="\t", index=False)
        logger.info(f"ENA metadata saved as {keywords}.tsv")
    elif only_ids and not save:
        logger.info(tabulate(df[["accession"]], headers="keys", showindex=False))
    elif only_ids and save:
        df["accession"].to_csv(f"{keywords}_ids.tsv", index=False, header=False)
        logger.info(f"ENA metadata saved as {keywords}_ids.tsv")
    else:
        logger.info(tabulate(df, headers="keys", showindex=False, tablefmt="plain"))


def ena_download(bioproject: str, cpus: int, fields: str = None, output_base_dir: Path = None) -> None:
    """
    Download FASTQ files from ENA given accession number.
    """
    err_id = bioproject
    threads = cpus
    base_path = output_base_dir if output_base_dir else Path.cwd()
    out_dir = base_path / err_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [x.name for x in Path.glob(out_dir, "*.fastq.gz")]

    if len(files) > 0:
        logger.info("Verifying MD5 File Checksums...")
        checksums(err_id, out_dir, files, cpus)
    else:
        err_urls = ena_urls(ena_fields(id_err=err_id, fields=fields))
        thread_map_urls(err_urls, out_dir, threads)
