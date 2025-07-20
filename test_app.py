import importlib
import sys
from types import SimpleNamespace


def _import_app(fake_secrets):
    fake_module = SimpleNamespace()
    fake_streamlit = SimpleNamespace(secrets=fake_secrets)
    stub_modules = {
        "openai": fake_module,
        "streamlit": fake_streamlit,
        "pandas": fake_module,
        "seaborn": fake_module,
        "matplotlib": SimpleNamespace(pyplot=fake_module),
        "matplotlib.pyplot": fake_module,
        "numpy": fake_module,
    }
    sklearn_tree = SimpleNamespace(DecisionTreeClassifier=object, DecisionTreeRegressor=object)
    sklearn_metrics = SimpleNamespace(classification_report=lambda *a, **k: "", mean_squared_error=lambda *a, **k: 0, mean_absolute_error=lambda *a, **k: 0)
    stub_modules["sklearn"] = SimpleNamespace(tree=sklearn_tree, metrics=sklearn_metrics)
    stub_modules["sklearn.tree"] = sklearn_tree
    stub_modules["sklearn.metrics"] = sklearn_metrics

    for name, mod in stub_modules.items():
        sys.modules[name] = mod

    if "app" in sys.modules:
        del sys.modules["app"]
    return importlib.import_module("app")


def test_load_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    app = _import_app({})
    assert app.load_openai_api_key() == "env-key"


def test_load_key_from_secrets(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _import_app({"OPENAI_API_KEY": "secret-key"})
    assert app.load_openai_api_key() == "secret-key"
