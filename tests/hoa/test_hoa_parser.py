from pathlib import Path

import pytest

import automatix.hoa as hoa

EXAMPLES_DIR = Path(__file__).parent / "examples"


@pytest.fixture(scope="function", params=list(EXAMPLES_DIR.glob("*.hoa")))
def hoa_example(request: pytest.FixtureRequest) -> str:
    infile = Path(request.param)
    assert infile.is_file(), f"HOA file not found: {request.param}"

    with open(infile, "r") as example:
        return example.read()


@pytest.mark.parametrize("hoa_file", list(EXAMPLES_DIR.glob("*.hoa")))
def test_hoa_parser(hoa_file: Path) -> None:
    assert hoa_file.is_file(), f"HOA file not found: {hoa_file}"
    hoa_example = hoa_file.read_text()
    parse_aut = hoa.parse(hoa_example)
    assert isinstance(parse_aut, hoa.ParsedAutomaton)
