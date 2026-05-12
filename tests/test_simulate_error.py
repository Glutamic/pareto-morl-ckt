"""Test that simulate() correctly detects ngspice failures."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open

from eval_engines.ngspice.ngspice_wrapper_parallel import NgSpiceWrapper


class TestSimulateExitCode:
    """Verify simulate() raises on non-zero exit and passes on zero exit."""

    def test_nonzero_exit_raises(self):
        """ngspice exiting with code 1 must raise RuntimeError."""
        # wait status = exit_code * 256 = 256
        with patch("os.system", return_value=256):
            with patch.object(NgSpiceWrapper, "__init__", return_value=None):
                wrapper = NgSpiceWrapper()
                with pytest.raises(RuntimeError, match=r"failed with status \d+"):
                    wrapper.simulate("/fake/path.cir")

    def test_zero_exit_passes(self):
        """ngspice exiting with code 0 must not raise."""
        with patch("os.system", return_value=0):
            with patch.object(NgSpiceWrapper, "__init__", return_value=None):
                wrapper = NgSpiceWrapper()
                info = wrapper.simulate("/fake/path.cir")
                assert info == 0

    def test_signal_kill_raises(self):
        """ngspice killed by SIGKILL (9) must raise RuntimeError."""
        # Signal in lower 8 bits: status = signal_number
        with patch("os.system", return_value=9):
            with patch.object(NgSpiceWrapper, "__init__", return_value=None):
                wrapper = NgSpiceWrapper()
                with pytest.raises(RuntimeError, match=r"failed with status \d+"):
                    wrapper.simulate("/fake/path.cir")


class TestSimulateLogCapture:
    """Verify ngspice output is captured to log file."""

    def test_error_includes_tail_content(self):
        """On failure, RuntimeError should include the last 20 lines of the log."""
        expected_lines = ["line {}\n".format(i) for i in range(25)]
        m_open = mock_open(read_data="".join(expected_lines))
        with patch("os.system", return_value=256):
            with patch("builtins.open", m_open):
                with patch.object(NgSpiceWrapper, "__init__", return_value=None):
                    wrapper = NgSpiceWrapper()
                    with pytest.raises(RuntimeError) as excinfo:
                        wrapper.simulate("/fake/path.cir")
                    assert "line 5" in str(excinfo.value)
                    assert "line 24" in str(excinfo.value)
                    assert "no output" not in str(excinfo.value)

    def test_error_fallback_when_log_unreadable(self):
        """On failure, if log cannot be read, show fallback message."""
        with patch("os.system", return_value=256):
            with patch("builtins.open", MagicMock(side_effect=FileNotFoundError)):
                with patch.object(NgSpiceWrapper, "__init__", return_value=None):
                    wrapper = NgSpiceWrapper()
                    with pytest.raises(RuntimeError, match="no output captured"):
                        wrapper.simulate("/fake/path.cir")


class TestFindInoise:
    """Verify find_inoise handles missing tran.csv gracefully."""

    def test_missing_file_raises_informative_error(self, tmp_path):
        """When tran.csv is missing, raise RuntimeError with clear message."""
        from eval_engines.ngspice.CircuitClass import CircuitClass
        obj = CircuitClass.__new__(CircuitClass)
        empty_dir = str(tmp_path)
        with pytest.raises(RuntimeError, match="tran.csv"):
            obj.find_inoise(empty_dir, 30e6)

    def test_valid_file_returns_delay_power(self, tmp_path):
        """When tran.csv exists with valid data, return parsed values."""
        from eval_engines.ngspice.CircuitClass import CircuitClass
        obj = CircuitClass.__new__(CircuitClass)
        csv_path = tmp_path / "tran.csv"
        csv_path.write_text("0\n5.0e-10\n1\n-2.0e-12\n")
        delay, power = obj.find_inoise(str(tmp_path), 30e6)
        assert delay == pytest.approx(500.0)  # 5e-10 / 1e-12
        assert power == pytest.approx(2.0)    # -(-2e-12) / 1e-12


class TestEnvSimulateFallback:
    """Verify env._simulate() handles RuntimeError from ngspice gracefully."""

    def test_sim_failure_returns_penalty_specs(self):
        """When simulation raises RuntimeError, return extreme penalty specs."""
        from unittest.mock import patch
        from env import MorlNgspiceEnv

        yaml_path = "eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_gf180.yaml"
        config = {
            "yaml_path": yaml_path,
            "env_name": "COMP",
            "total_timesteps": 1000,
            "episode_len": 10,
            "corner_sim": False,
        }

        with patch("eval_engines.ngspice.CircuitClass.CircuitClass.run",
                   side_effect=RuntimeError("ngspice failed with status 256")):
            env = MorlNgspiceEnv(env_config=config)
            obs, info = env.reset()
            assert obs is not None
            assert "cur_specs" in info
            assert np.all(info["cur_specs"] == 1e9)
