from src.credit_risk.config import load_config


def test_load_config() -> None:
    config = load_config("params.yaml")

    assert config.dataset.raw_path.as_posix() == "data/raw/german_credit.csv"
    assert config.dataset.target_column == "target_bad"
    assert config.validation.expected_columns == 21
