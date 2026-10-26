
from pathlib import Path
import pandas as pd

SOURCE_FILE = Path(__file__).resolve().parent / "SMSSpamCollection"
OUTPUT_FILE = Path(__file__).resolve().parent / "spam.csv"


def main():
    if not SOURCE_FILE.exists():
        print("Error: SMSSpamCollection file not found.")
        print("Place the extracted dataset in this project folder.")
        return

    df = pd.read_csv(
        SOURCE_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Dataset conversion completed.")
    print("Messages converted:", len(df))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
