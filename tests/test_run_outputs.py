
import sys

PROJECT_ROOT = "/home/mhpereir/eulerian_heat_budget_surface"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
import json
from pathlib import Path

from src.run_outputs import prepare_run_paths, write_run_info
from src.specs import DomainRequest, DomainSpec, SurfaceBehaviour


def test_prepare_run_paths_uses_pbs_jobid(tmp_path):
    paths = prepare_run_paths(
        str(tmp_path),
        env={"PBS_JOBID": "2586030.venus"},
        now=datetime(2026, 3, 17, 12, 0, 0),
        pid=99,
    )

    assert paths.run_id == "2586030.venus"
    assert Path(paths.run_root) == tmp_path / "2586030.venus"
    assert Path(paths.plot_dir) == tmp_path / "2586030.venus" / "outputs_here"
    assert Path(paths.plot_dir).is_dir()
    assert Path(paths.metadata_path) == tmp_path / "2586030.venus" / "outputs_here" / "run_info.json"


def test_write_run_info_serializes_specs_to_json(tmp_path):
    paths = prepare_run_paths(
        str(tmp_path),
        env={"PBS_JOBID": "2586030.venus"},
        now=datetime(2026, 3, 17, 12, 0, 0),
        pid=99,
    )

    request = DomainRequest(
        bbox=(40.0, 60.0, -130.0, -110.0),
        margin_n=1,
        zg_top_pressure=60000.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    domain_spec = DomainSpec(
        lat_min=40.25,
        lat_max=59.75,
        lon_min=-129.75,
        lon_max=-110.25,
        zg_top_pressure=60000.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    surface_behaviour = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="combined",
    )

    metadata_path = write_run_info(
        paths,
        request=request,
        domain_spec=domain_spec,
        surface_behaviour=surface_behaviour,
        cli_args={"lat_min": 40.0, "in_surface_variables": False},
        env={"PBS_JOBID": "2586030.venus"},
        now=datetime(2026, 3, 17, 12, 30, 0),
    )

    payload = json.loads(Path(metadata_path).read_text())

    assert payload["run_id"] == "2586030.venus"
    assert payload["pbs_job_id"] == "2586030.venus"
    assert payload["plot_dir"] == str(tmp_path / "2586030.venus" / "outputs_here")
    assert payload["request"]["bbox"] == [40.0, 60.0, -130.0, -110.0]
    assert payload["domain_spec"]["lat_min"] == 40.25
    assert payload["surface_behaviour"]["surface_variable_mode"] == "combined"
    assert payload["cli_args"]["in_surface_variables"] is False
