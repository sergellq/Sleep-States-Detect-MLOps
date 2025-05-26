import zipfile
from pathlib import Path

import gdown


def download_data(file_id: str, target_dir: str | Path):
    """
    link: ссылка на датасет
    force: проверять ли наличие файлов перед скачкой
    data_dir: папка куда качать

    качает датасет и сохраняет в папку data/ (если нет хоть одного файла)

    по идее надо бы с кагла напрямую качать, но там авторизация...
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "downloaded.zip"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Скачивание файла с Google Drive: {url}")
    gdown.download(url, str(output_path), quiet=False)

    print(f"Распаковка архива {output_path} в {target_dir.resolve()} ...")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    print("✅ Успешно скачано и распаковано!")
