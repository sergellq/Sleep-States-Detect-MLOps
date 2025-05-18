import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm.auto import tqdm


def download_data(link: str, data_dir: str | Path):
    """
    link: ссылка на датасет
    force: проверять ли наличие файлов перед скачкой
    data_dir: папка куда качать

    качает датасет и сохраняет в папку data/ (если нет хоть одного файла)

    по идее надо бы с кагла напрямую качать, но там авторизация...
    """

    print(link)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Скачивание {link} ...")
    with requests.get(link, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        content = BytesIO()
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Скачивание"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                content.write(chunk)
                pbar.update(len(chunk))

    content.seek(0)

    print("Распаковка...")
    with zipfile.ZipFile(content) as zip_file:
        for file in tqdm(zip_file.namelist(), desc="Распаковка"):
            zip_file.extract(member=file, path=data_dir)

    print(f"Архив успешно загружен и распакован в {data_dir.resolve()}")
