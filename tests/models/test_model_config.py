import builtins
import os
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.models import ModelConfig

dir_path = os.path.dirname(os.path.abspath(__file__))
VALID_CONFIG_PATH = os.path.join(dir_path, "configs/config.yaml")
VALID_CONFIG_PATH_2 = os.path.join(dir_path, "configs/config_2.yaml")


def test_config_not_set():
    with pytest.raises(
        FileNotFoundError, match="Config file is not provided which is needed to load the model."
    ):
        ModelConfig()


def test_config_not_found():
    with pytest.raises(FileNotFoundError, match="Config file 'nonexistent.yaml' not found."):
        ModelConfig(development_config="nonexistent.yaml")


def test_config_invalid_yaml(tmp_path):
    tmp_file = tmp_path / "invalid_config.yaml"
    tmp_file.write_text("invalid_yaml: \n  - this is not valid \n-yaml")
    config = ModelConfig(development_config=str(tmp_file))
    with pytest.raises(MlflowException, match="Error parsing YAML file: "):
        config.get("key")


def test_config_key_not_found():
    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    with pytest.raises(KeyError, match="Key 'key' not found in configuration: "):
        config.get("key")


def test_config_setup_correctly():
    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    assert config.get("llm_parameters").get("temperature") == 0.01


def test_config_setup_correctly_with_mlflow_langchain():
    with mock.patch("mlflow.models.model_config.__mlflow_model_config__", new=VALID_CONFIG_PATH):
        config = ModelConfig(development_config="nonexistent.yaml")
        assert config.get("llm_parameters").get("temperature") == 0.01


def test_config_setup_with_mlflow_langchain_path():
    with mock.patch("mlflow.models.model_config.__mlflow_model_config__", new=VALID_CONFIG_PATH_2):
        # here the config.yaml has the max_tokens set to 500
        # where as the config_2.yaml has it set to 200.
        # Here we give preference to the __mlflow_model_config__.
        config = ModelConfig(development_config=VALID_CONFIG_PATH)
        assert config.get("llm_parameters").get("max_tokens") == 200


def test_config_development_config_must_be_specified_with_keyword():
    with pytest.raises(TypeError, match="1 positional argument but 2 were given"):
        ModelConfig(VALID_CONFIG_PATH_2)


def test_config_development_config_is_a_dict():
    config = ModelConfig(development_config={"llm_parameters": {"temperature": 0.01}})
    assert config.get("llm_parameters").get("temperature") == 0.01


def test_config_setup_correctly_errors_with_no_config_path():
    with mock.patch("mlflow.models.model_config.__mlflow_model_config__", new=""):
        with pytest.raises(
            FileNotFoundError,
            match="Config file is not provided which is needed to load the model.",
        ):
            ModelConfig(development_config=VALID_CONFIG_PATH)


def test_config_development_config_to_dict():
    config = ModelConfig(development_config={"llm_parameters": {"temperature": 0.01}})
    assert config.to_dict() == {"llm_parameters": {"temperature": 0.01}}

    config = ModelConfig(development_config=VALID_CONFIG_PATH)
    assert config.to_dict() == {
        "embedding_model_query_instructions": "Represent this sentence for searching "
        "relevant passages:",
        "llm_model": "databricks-dbrx-instruct",
        "llm_prompt_template": "You are a trustful assistant.",
        "retriever_config": {"k": 5, "use_mmr": False},
        "llm_parameters": {"temperature": 0.01, "max_tokens": 500},
        "llm_prompt_template_variables": ["chat_history", "context", "question"],
    }


def test_config_reads_utf8_regardless_of_default_encoding(tmp_path, monkeypatch):
    prompt = "You’re a helpful assistant — reply in 日本語"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'llm_prompt_template: "{prompt}"\n', encoding="utf-8")

    # Make text-mode `open` in mlflow.models.model_config default to cp1252, as it does on
    # Windows with a legacy code page, so the test does not depend on the runner's locale.
    def open_with_cp1252_default(file, mode="r", *args, encoding=None, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return builtins.open(file, mode, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr("mlflow.models.model_config.open", open_with_cp1252_default, raising=False)
    config = ModelConfig(development_config=str(config_path))
    assert config.get("llm_prompt_template") == prompt
