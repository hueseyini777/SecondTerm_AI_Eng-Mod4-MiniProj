"""Download the monthly NYC TLC trip file configured in params.yaml."""

from urllib.request import urlretrieve

from src.config import load_params, raw_data_path

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{name}"


def download() -> None:
    target = raw_data_path(load_params())
    if target.exists():
        print(f"{target.name} already present, skipping download")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target.name} ...")
    urlretrieve(TLC_URL.format(name=target.name), target)


if __name__ == "__main__":
    download()
