import dataclasses
import hashlib
import io
import logging
import multiprocessing
import os
import warnings
import signal  # Add this import
from collections import defaultdict
from os import rename, makedirs
from os.path import join, exists
from typing import Union, Dict

import PIL.Image
import datasets
import numpy as np
import requests
import urllib3
from PIL import ImageFile
from urllib3.exceptions import MaxRetryError
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

from tqdm import tqdm

from olmo.data.dataset import DATA_HOME
from olmo.data.model_preprocessor import setup_pil

if "PIXMO_IMAGE_DIR" in os.environ:
    PIXMO_IMAGES = os.environ["PIXMO_IMAGE_DIR"]
elif DATA_HOME is not None:
    PIXMO_IMAGES = join(DATA_HOME, "pixmo_images")
else:
    PIXMO_IMAGES = None
"""Where to save downloaded images"""


PIL.Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclasses.dataclass
class DownloadError:
    url: str
    exception: Exception


@dataclasses.dataclass
class ImageError:
    url: str
    exception: Exception=None


def compute_hash(string: Union[str, bytes]) -> str:
    if isinstance(string, str):
        return hashlib.sha256(string.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(string).hexdigest()


class TimeoutError(Exception):  # Add this class
    pass


def timeout_handler(signum, frame):  # Add this function
    raise TimeoutError("Operation timed out")


def _download_images(args):
    url, image_sha, check_sha, cache_only, kwargs = args
    image_id = compute_hash(url)
    cache_file = join(PIXMO_IMAGES, image_id)

    # Set up timeout for the entire function (30 seconds max per image)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        # Create and configure session
        session = requests.Session()
        retries = Retry(
            total=2,  # Reduce retries to avoid hanging
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            connect=2,  # Add connection retries
            read=2,     # Add read retries
            redirect=2  # Add redirect retries
        )
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        if exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    image_bytes = f.read()
                # Check if this is actually an error file (text instead of binary)
                if len(image_bytes) < 100 and (image_bytes.startswith(b'<') or b'Error' in image_bytes[:100]):
                    # This is likely an error file, treat as download error
                    return DownloadError(url, Exception("Cached error file"))
            except Exception as e:
                return DownloadError(url, Exception(f"Error reading cached file: {str(e)}"))
        elif cache_only:
            return DownloadError(url, ValueError('Not in cache'))
        else:
            try:
                # Use a shorter timeout to avoid hanging
                response = session.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                # Read content with size limit to avoid memory issues
                image_bytes = b''
                max_size = 50 * 1024 * 1024  # 50MB limit
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        image_bytes += chunk
                        if len(image_bytes) > max_size:
                            raise ValueError(f"Image too large: {len(image_bytes)} bytes")
                
            except Exception as e:
                # Write response to file so we know the URL failed and won't try it again
                try:
                    with open(cache_file, 'w') as f:
                        f.write(f"Error: {str(e)}")
                except:
                    pass  # If we can't write the error file, just continue
                # Convert exception to string to avoid pickling issues with SSLContext
                return DownloadError(url, Exception(str(e)))

            # Write the file bytes
            try:
                with open(cache_file + ".tmp", 'wb') as f:
                    f.write(image_bytes)
                rename(cache_file + ".tmp", cache_file)
            except Exception as e:
                return DownloadError(url, Exception(f"Error writing file: {str(e)}"))

        if check_sha:
            downloaded_hash = compute_hash(image_bytes)
            assert image_sha is not None
            if downloaded_hash != image_sha:
                return ImageError(url, ValueError("Mismatched image hash"))
        else:
            # Else make sure we actually got an image, and it can be parsed by PIL
            try:
                # Avoid annoying palette transparency warnings filling up the logs
                with warnings.catch_warnings(record=True) as w:
                    img = PIL.Image.open(io.BytesIO(image_bytes))
                    if min(img.size) == 0:
                        raise ValueError("Zero dimensional image")
            except Exception as e:
                # Convert exception to string to avoid pickling issues
                return ImageError(url, Exception(str(e)))

        return url, cache_file
    
    except TimeoutError:
        return DownloadError(url, Exception("Download timed out after 30 seconds"))
    except Exception as e:
        return DownloadError(url, Exception(f"Unexpected error: {str(e)}"))
    finally:
        # Always cancel the alarm
        signal.alarm(0)


def download_pixmo_urls(
    data: datasets.Dataset,
    n_processes,
    check_sha,
    request_kwargs=None,
    cache_only=False,
    verify=True
) -> Dict[str, str]:
    """Download urls from a PixMo dataset, return a map of urls->filename"""
    if check_sha:
        urls_and_shas = list(dict(zip(data["image_url"], data["image_sha256"])).items())
    else:
        urls_and_shas = [(url, None) for url in list(set(data["image_url"]))]

    # Randomize order so resuming is more convenient, speed is more predictable,
    # and to distribute requests across different domains
    urls_and_shas.sort(key=lambda x: x[0])
    np.random.RandomState(58713).shuffle(urls_and_shas)

    logging.info(f"Getting files for {len(urls_and_shas)} image URLs")
    makedirs(PIXMO_IMAGES, exist_ok=True)
    if request_kwargs is None:
        request_kwargs = dict(timeout=10)  # Reduce timeout from 60 to 10
    if not verify:
        request_kwargs["verify"] = False
        urllib3.disable_warnings()

    images = []
    to_save = [(url, image_sha, check_sha, cache_only, request_kwargs) for url, image_sha in urls_and_shas]
    pbar = tqdm(total=len(to_save), desc=f"{0}/{len(to_save)}")
    image_error, download_err, success = 0, 0, 0

    if n_processes != 1:
        def _iter():
            with multiprocessing.Pool(processes=n_processes, initializer=setup_pil) as pool:
                result = pool.imap_unordered(_download_images, to_save)
                for val in result:
                    yield val
    else:
        setup_pil()
        def _iter():
            for val in to_save:
                result = _download_images(val)
                if result is None:
                    # This should never happen now, but just in case
                    result = DownloadError(val[0], Exception("Function returned None"))
                yield result

    found_urls = {}
    processed = 0
    for val in _iter():
        processed += 1
        if isinstance(val, ImageError):
            image_error += 1
        elif isinstance(val, DownloadError):
            download_err += 1
        else:
            url, filename = val
            found_urls[url] = filename
            success += 1
        pbar.update(1)
        pbar.set_description(
            f"dl_er={download_err} file_err={image_error}",
            refresh=False)
    pbar.close()
    logging.info(f"Got images for {len(found_urls)}/{len(urls_and_shas)} ({len(found_urls)/len(urls_and_shas)*100:0.2f}%) image URLs")
    return found_urls


def filter_and_group_data(data: datasets.Dataset, url_to_path: Dict, check_sha: bool) -> datasets.Dataset:
    """
    Groups a pixmo datasets so each row contains all annotation for one image, and add
    images path using `url_to_path`, removing rows that do not exist in `url_to_path`
    """
    grouped_by_url = defaultdict(list)
    for example in data:
        if example["image_url"] not in url_to_path:
            continue
        grouped_by_url[example["image_url"]].append(example)

    grouped_examples = []
    for image_url, examples in grouped_by_url.items():
        grouped = dict(
            image_url=image_url,
            image=url_to_path[image_url],
        )
        if "image_sha256" in examples[0] and not check_sha:
            assert all(examples[0]["image_sha256"] == ex["image_sha256"] for ex in examples)
            grouped["original_sha256"] = examples[0]["image_sha256"]
        annotations = defaultdict(list)
        for ex in examples:
            for k, v in ex.items():
                if k not in ["image_url", "image_sha256"]:
                    annotations[k].append(v)
        grouped.update(annotations)
        grouped_examples.append(grouped)
    return datasets.Dataset.from_list(grouped_examples)
