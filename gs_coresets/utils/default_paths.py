from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

from gs_coresets.utils.io_utils import expanduser_path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
ROOT_DIR = PROJECT_DIR
GRAPHDECO_DIR = (PROJECT_DIR / "external" / "gaussian_splatting").resolve()

DEFAULT_OUTPUT_DIR = (PROJECT_DIR / "outputs").resolve()
DEFAULT_BASELINE_DIR = DEFAULT_OUTPUT_DIR / "baseline"
DEFAULT_SENSITIVITY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "sensitivities"
DEFAULT_TESTS_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "tests"
DEFAULT_TESTS_BASELINE_DIR = DEFAULT_TESTS_OUTPUT_DIR / "baseline"
DEFAULT_TESTS_SENSITIVITY_OUTPUT_DIR = DEFAULT_TESTS_OUTPUT_DIR / "sensitivities"

DEFAULT_POINT_CLOUD_RELATIVE = Path("point_cloud") / "iteration_30000" / "point_cloud.ply"
DEFAULT_CAMERAS_FILENAME = "cameras.json"

DEFAULT_MODEL_DIR_ENV = "GS_CORESETS_DEFAULT_MODEL_DIR"
DEFAULT_OUTPUT_DIR_ENV = "GS_CORESETS_OUTPUT_DIR"
DEFAULT_SENS_OUTPUT_ENV = "GS_CORESETS_SENS_OUTPUT_DIR"
DEFAULT_SENS_TEST_OUTPUT_ENV = "GS_CORESETS_SENS_TEST_OUTPUT_DIR"


def resolve_project_dir() -> Path:
    return PROJECT_DIR


def resolve_graphdeco_dir() -> Path:
    return GRAPHDECO_DIR


def resolve_output_dir(output_dir: Optional[str] = None) -> Path:
    base = output_dir or os.environ.get(DEFAULT_OUTPUT_DIR_ENV) or str(DEFAULT_OUTPUT_DIR)
    return Path(expanduser_path(base))


def resolve_model_dir(model_dir: Optional[str]) -> str:
    base = model_dir or os.environ.get(DEFAULT_MODEL_DIR_ENV) or str(DEFAULT_BASELINE_DIR)
    return expanduser_path(base)


def resolve_point_cloud_path(ply_path: Optional[str], model_dir: str | Path) -> str:
    if ply_path:
        return expanduser_path(ply_path)
    return str((Path(expanduser_path(str(model_dir))) / DEFAULT_POINT_CLOUD_RELATIVE).resolve())


def resolve_cameras_path(cam_json_path: Optional[str], model_dir: str | Path) -> str:
    if cam_json_path:
        return expanduser_path(cam_json_path)
    return str((Path(expanduser_path(str(model_dir))) / DEFAULT_CAMERAS_FILENAME).resolve())


def resolve_sensitivity_output_dir(save_dir: Optional[str] = None) -> Path:
    base = save_dir or os.environ.get(DEFAULT_SENS_OUTPUT_ENV) or str(DEFAULT_SENSITIVITY_OUTPUT_DIR)
    return Path(expanduser_path(base)).resolve()


def resolve_test_sensitivity_output_dir(save_dir: Optional[str] = None) -> Path:
    base = save_dir or os.environ.get(DEFAULT_SENS_TEST_OUTPUT_ENV) or str(DEFAULT_TESTS_SENSITIVITY_OUTPUT_DIR)
    return Path(expanduser_path(base)).resolve()


PathLikeStr = Union[str, Path]


def ensure_in_syspath(path: PathLikeStr, *, append: bool = False) -> None:
    entry = str(path)
    if entry not in sys.path:
        if append:
            sys.path.append(entry)
        else:
            sys.path.insert(0, entry)


def ensure_project_dir_on_syspath(*, append: bool = False) -> None:
    ensure_in_syspath(PROJECT_DIR, append=append)


def ensure_graphdeco_dir_on_syspath(*, append: bool = False) -> None:
    ensure_in_syspath(GRAPHDECO_DIR, append=append)


def ensure_all_on_syspath() -> None:
    ensure_project_dir_on_syspath(append=False)
    ensure_graphdeco_dir_on_syspath(append=False)
