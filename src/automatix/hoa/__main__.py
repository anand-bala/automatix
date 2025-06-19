import argparse
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import automatix.hoa as hoa


@dataclass
class Args:
    infile: Path

    @staticmethod
    def parse() -> "Args":
        parser = argparse.ArgumentParser(
            prog="automatix.hoa",
            description="Test parsing HOA files",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _ = parser.add_argument("infile", type=lambda p: Path(p))

        args = parser.parse_args()
        assert isinstance(args.infile, Path) and args.infile.is_file(), "Input file doesn't exist"

        return Args(args.infile)


def main(args: Args) -> None:
    with open(args.infile, "r") as infile:
        spec = infile.read()
        print(spec)
        tree = hoa.parse(spec)

    pprint(tree)


if __name__ == "__main__":
    main(Args.parse())
