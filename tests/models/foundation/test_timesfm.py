import os

import pytest

from timecopilot.models.foundation.timesfm import _TimesFMV1, _TimesFMV2_p5

MODEL_PARAMS = [
    (
        _TimesFMV1,
        [
            "timecopilot.models.foundation.timesfm.timesfm_v1.TimesFmCheckpoint",
            "timecopilot.models.foundation.timesfm.timesfm_v1.TimesFm",
        ],
    ),
    (
        _TimesFMV2_p5,
        [
            "timecopilot.models.foundation.timesfm.TimesFM_2p5_200M_torch",
        ],
    ),
]


@pytest.mark.parametrize("model_class, mock_paths", MODEL_PARAMS)
def test_load_model_from_local_path(mocker, model_class, mock_paths):
    """Tests loading from a local path."""
    module_path = "timecopilot.models.foundation.timesfm"
    mock_os_exists = mocker.patch(f"{module_path}.os.path.exists", return_value=True)
    mock_loader = [mocker.patch(i) for i in mock_paths]

    local_path = "/fake/local/path"

    model_instance = model_class(
        repo_id=local_path,
        context_length=64,
        batch_size=32,
        alias="test",
    )

    with model_instance._get_predictor(prediction_length=12) as p:
        predictor = p

    mock_os_exists.assert_called_once_with(local_path)

    if model_class is _TimesFMV1:
        assert predictor is mock_loader[1].return_value
        expected_path = os.path.join(local_path, "torch_model.ckpt")
        mock_loader[0].assert_called_once_with(path=expected_path)
    elif model_class is _TimesFMV2_p5:
        # `from_pretrained` handles both local directories and HF repos.
        assert predictor is mock_loader[0].from_pretrained.return_value
        mock_loader[0].from_pretrained.assert_called_once_with(local_path)


@pytest.mark.parametrize("model_class, mock_paths", MODEL_PARAMS)
def test_load_model_from_hf_repo(mocker, model_class, mock_paths):
    """Tests loading from a Hugging Face repo."""
    module_path = "timecopilot.models.foundation.timesfm"
    mock_os_exists = mocker.patch(f"{module_path}.os.path.exists", return_value=False)
    mock_repo_exists = mocker.patch(f"{module_path}.repo_exists", return_value=True)
    mock_loader = [mocker.patch(i) for i in mock_paths]

    repo_id = "/fake/google/repo-id"

    model_instance = model_class(
        repo_id=repo_id,
        context_length=64,
        batch_size=32,
        alias="test",
    )

    with model_instance._get_predictor(prediction_length=12) as p:
        predictor = p

    mock_os_exists.assert_called_once_with(repo_id)
    mock_repo_exists.assert_called_once_with(repo_id)

    if model_class is _TimesFMV1:
        assert predictor is mock_loader[1].return_value
        mock_loader[0].assert_called_once_with(huggingface_repo_id=repo_id)
    elif model_class is _TimesFMV2_p5:
        assert predictor is mock_loader[0].from_pretrained.return_value
        mock_loader[0].from_pretrained.assert_called_once_with(repo_id)


@pytest.mark.parametrize("model_class, _", MODEL_PARAMS)
def test_model_raises_OSError_on_failed_load(mocker, model_class, _):
    """Tests that an OSError is raised on a failed load attempt."""

    module_path = "timecopilot.models.foundation.timesfm"
    mock_os_exists = mocker.patch(f"{module_path}.os.path.exists", return_value=False)
    mock_repo_exists = mocker.patch(f"{module_path}.repo_exists", return_value=False)

    repo_id = "/this-is-a-fake/google/repo-id"

    model_instance = model_class(
        repo_id=repo_id,
        context_length=64,
        batch_size=32,
        alias="test",
    )
    with (
        pytest.raises(OSError, match="Failed to load model"),
        model_instance._get_predictor(prediction_length=12),
    ):
        pass

    mock_os_exists.assert_called_once_with(repo_id)
    mock_repo_exists.assert_called_once_with(repo_id)
