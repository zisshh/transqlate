import importlib

def test_cpu_opt_out(monkeypatch):
    monkeypatch.setenv("TRANSQLATE_NO_QUANT", "1")
    from transqlate.utils.hardware import detect_device_and_quant
    dm, dt, q = detect_device_and_quant(True)
    assert q is False


def test_retry_on_bnb_failure(monkeypatch, mocker):
    import torch
    mocker.patch(
        "transqlate.utils.hardware.detect_device_and_quant",
        return_value=("auto", torch.float16, True),
    )
    import transqlate.inference as inf

    class DummyModel:
        def eval(self):
            pass

    mock_model = mocker.patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[RuntimeError("bitsandbytes"), DummyModel()],
    )

    class DummyTok:
        eos_token = ""
        pad_token = ""

    mocker.patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTok())
    inf.NL2SQLInference(model_dir="dummy")
    assert mock_model.call_count == 2
