import builtins
import json
import os

import pytest
import yaml

import mlflow
from mlflow.models import Model


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def model_config():
    return {
        "use_gpu": True,
        "temperature": 0.9,
        "timeout": 300,
    }


def _load_pyfunc(path):
    return TestModel()


class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


class InferenceContextModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        # This mock class returns the internal inference configuration keys and values available
        return context.model_config.items()


def test_save_with_model_config(model_path, model_config):
    model = InferenceContextModel()
    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path)

    assert loaded_model.model_config
    assert set(model_config.keys()) == set(loaded_model.model_config)
    assert all(loaded_model.model_config[k] == v for k, v in model_config.items())
    assert all(loaded_model.model_config[k] == v for k, v in loaded_model.predict([[0]]))


@pytest.mark.parametrize(
    "model_config_path",
    [
        os.path.abspath("tests/pyfunc/sample_code/config.yml"),
        "tests/pyfunc/../pyfunc/sample_code/config.yml",
    ],
)
def test_save_with_model_config_path(model_path, model_config, model_config_path):
    model = InferenceContextModel()
    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config_path)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path)

    assert loaded_model.model_config
    assert set(model_config.keys()) == set(loaded_model.model_config)
    assert all(loaded_model.model_config[k] == v for k, v in model_config.items())
    assert all(loaded_model.model_config[k] == v for k, v in loaded_model.predict([[0]]))


def test_override_model_config(model_path, model_config):
    model = TestModel()
    inference_override = {"timeout": 400}

    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=inference_override)

    assert loaded_model.model_config["timeout"] == 400
    assert all(loaded_model.model_config[k] == v for k, v in inference_override.items())


@pytest.mark.parametrize(
    "model_config_path",
    [
        os.path.abspath("tests/pyfunc/sample_code/config.yml"),
        "tests/pyfunc/../pyfunc/sample_code/config.yml",
    ],
)
def test_override_model_config_path(tmp_path, model_path, model_config_path):
    model = TestModel()
    inference_override = {"timeout": 400}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(inference_override))

    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config_path)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=str(config_path))

    assert loaded_model.model_config["timeout"] == 400
    assert all(loaded_model.model_config[k] == v for k, v in inference_override.items())


def test_override_model_config_ignore_invalid(model_path, model_config):
    model = TestModel()
    inference_override = {"invalid_key": 400}

    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=inference_override)

    assert loaded_model.predict([[5]])
    assert all(k not in loaded_model.model_config for k in inference_override.keys())


@pytest.mark.parametrize(
    "model_config_path",
    [
        os.path.abspath("tests/pyfunc/sample_code/config.yml"),
        "tests/pyfunc/../pyfunc/sample_code/config.yml",
    ],
)
def test_override_model_config_path_ignore_invalid(tmp_path, model_path, model_config_path):
    model = TestModel()
    inference_override = {"invalid_key": 400}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(inference_override))

    mlflow.pyfunc.save_model(model_path, python_model=model, model_config=model_config_path)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=str(config_path))

    assert loaded_model.predict([[5]])
    assert all(k not in loaded_model.model_config for k in inference_override.keys())


def test_pyfunc_without_model_config(model_path, model_config):
    model = TestModel()
    mlflow.pyfunc.save_model(model_path, python_model=model)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_path, model_config=model_config)

    assert loaded_model.predict([[5]])
    assert not loaded_model.model_config


def test_pyfunc_loader_without_model_config(model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        data_path=".",
        loader_module=__name__,
        code_paths=[__file__],
        mlflow_model=Model(run_id="test", artifact_path="testtest"),
    )

    inference_override = {"invalid_key": 400}
    pyfunc_model = mlflow.pyfunc.load_model(model_path, model_config=inference_override)

    assert not pyfunc_model.model_config


@pytest.mark.parametrize("file_name", ["config.yaml", "config.json"])
def test_log_model_reads_model_config_file_as_utf8(tmp_path, monkeypatch, file_name):
    model_config = {"llm_prompt_template": "You’re a helpful assistant — reply in 日本語"}
    config_path = tmp_path / file_name
    # JSON is also valid YAML, so the same content works for both extensions.
    config_path.write_text(json.dumps(model_config, ensure_ascii=False), encoding="utf-8")

    # Make text-mode `open` default to cp1252, as it does on Windows with a legacy code page,
    # so the test does not depend on the runner's locale.
    def open_with_cp1252_default(file, mode="r", *args, encoding=None, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return builtins.open(file, mode, *args, encoding=encoding, **kwargs)

    for module in ["mlflow.models.model", "mlflow.utils.model_utils"]:
        monkeypatch.setattr(f"{module}.open", open_with_cp1252_default, raising=False)

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=InferenceContextModel(), model_config=str(config_path)
        )

    assert mlflow.get_run(run.info.run_id).data.params == model_config
    assert mlflow.pyfunc.load_model(model_info.model_uri).model_config == model_config
