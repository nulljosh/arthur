"""Tests for arthur_watchdog.py daemon."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "daemon"))

import arthur_watchdog as wd


def _psutil_mock():
    """Context manager that patches psutil for CI compatibility.
    
    GitHub Actions runners lack /proc/meminfo entries psutil expects.
    Also forces total RAM to 32GB so Daemon._max_size stays '125M'.
    """
    vm = MagicMock()
    vm.total = 32 * 1024**3   # 32GB -> _max_size = '125M'
    vm.available = 8 * 1024**3

    swap = MagicMock()
    swap.used = 0

    return patch.multiple(
        'psutil',
        virtual_memory=MagicMock(return_value=vm),
        swap_memory=MagicMock(return_value=swap),
        cpu_percent=MagicMock(return_value=30.0),
    )


class TestPowerMode(unittest.TestCase):
    """ResourceMonitor.get_power_mode() returns correct mode based on resources."""

    def _mock_monitor(self, disk=20.0, cpu=30.0, ram=8.0):
        m = wd.ResourceMonitor()
        m.get_disk_free_gb = MagicMock(return_value=disk)
        m.get_cpu_percent = MagicMock(return_value=cpu)
        m.get_ram_available_gb = MagicMock(return_value=ram)
        return m

    def test_all_good_returns_full(self):
        mode, _ = self._mock_monitor().get_power_mode()
        self.assertEqual(mode, "full")

    def test_low_disk_returns_pause(self):
        mode, _ = self._mock_monitor(disk=3.0).get_power_mode()
        self.assertEqual(mode, "pause")

    def test_low_ram_returns_low(self):
        mode, _ = self._mock_monitor(ram=1.5).get_power_mode()
        self.assertEqual(mode, "low")

    def test_high_cpu_returns_low(self):
        mode, _ = self._mock_monitor(cpu=85.0).get_power_mode()
        self.assertEqual(mode, "low")

    def test_ram_at_threshold_returns_full(self):
        """RAM exactly at 2GB should be full (not strict less-than)."""
        mode, _ = self._mock_monitor(ram=2.0).get_power_mode()
        self.assertEqual(mode, "full")

    def test_ram_below_threshold_returns_low(self):
        mode, _ = self._mock_monitor(ram=1.9).get_power_mode()
        self.assertEqual(mode, "low")

    def test_cpu_at_limit_returns_full(self):
        """CPU exactly at 70% should be full (limit is >70)."""
        mode, _ = self._mock_monitor(cpu=70.0).get_power_mode()
        self.assertEqual(mode, "full")

    def test_disk_at_limit_returns_full(self):
        """Disk exactly at 5GB should be full."""
        mode, _ = self._mock_monitor(disk=5.0).get_power_mode()
        self.assertEqual(mode, "full")

    def test_disk_below_limit_returns_pause(self):
        mode, _ = self._mock_monitor(disk=4.9).get_power_mode()
        self.assertEqual(mode, "pause")

    def test_low_disk_overrides_good_ram_cpu(self):
        """Disk critical takes priority over everything."""
        mode, _ = self._mock_monitor(disk=2.0, cpu=10.0, ram=12.0).get_power_mode()
        self.assertEqual(mode, "pause")

    def test_multiple_low_conditions(self):
        """Low RAM + high CPU still returns low, not pause."""
        mode, _ = self._mock_monitor(ram=2.0, cpu=90.0).get_power_mode()
        self.assertEqual(mode, "low")

    def test_stats_returned(self):
        """get_power_mode returns stats dict."""
        _, stats = self._mock_monitor(disk=15.0, cpu=40.0, ram=6.0).get_power_mode()
        self.assertAlmostEqual(stats["disk"], 15.0)
        self.assertAlmostEqual(stats["cpu"], 40.0)
        self.assertAlmostEqual(stats["ram"], 6.0)


class TestBatchSize(unittest.TestCase):
    """ResourceMonitor.get_batch_size() returns correct size per mode."""

    def _make_monitor(self):
        with _psutil_mock():
            return wd.ResourceMonitor()

    def test_full_returns_8(self):
        self.assertEqual(self._make_monitor().get_batch_size("full"), 4)

    def test_low_returns_4(self):
        self.assertEqual(self._make_monitor().get_batch_size("low"), 2)

    def test_pause_returns_0(self):
        self.assertEqual(self._make_monitor().get_batch_size("pause"), 0)


class TestState(unittest.TestCase):
    """Daemon state loading and persistence."""

    def test_load_default_state(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            self.assertEqual(d.state, {"epoch": 0, "total": 3, "size": "65M"})

    def test_load_existing_state(self):
        state = {"epoch": 2, "total": 3, "size": "125M"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            f.flush()
            try:
                with _psutil_mock(), patch.object(wd, "STATE_FILE", Path(f.name)):
                    d = wd.Daemon()
                    self.assertEqual(d.state["epoch"], 2)
                    self.assertEqual(d.state["size"], "125M")
            finally:
                os.unlink(f.name)

    def test_save_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            f.flush()
            try:
                with _psutil_mock(), patch.object(wd, "STATE_FILE", Path(f.name)):
                    d = wd.Daemon()
                    d.state = {"epoch": 1, "total": 3, "size": "65M"}
                    d.save_state()
                    d2 = wd.Daemon()
                    self.assertEqual(d2.state["epoch"], 1)
            finally:
                os.unlink(f.name)

    def test_default_mode_is_full(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            self.assertEqual(d._current_mode, "full")


class TestPromotion(unittest.TestCase):
    """Size promotion logic: 65M -> 125M -> stop."""

    def _daemon_with_state(self, state):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.state = state
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            d.generate_roadmap = MagicMock()
            return d

    def test_promote_65m_to_125m(self):
        d = self._daemon_with_state({"epoch": 3, "total": 3, "size": "65M"})
        self.assertTrue(d.promote_size())
        self.assertEqual(d.state["size"], "125M")
        self.assertEqual(d.state["epoch"], 0)

    def test_125m_does_not_promote(self):
        d = self._daemon_with_state({"epoch": 3, "total": 3, "size": "125M"})
        self.assertFalse(d.promote_size())


class TestCommand(unittest.TestCase):
    """start_training() builds the correct command."""

    @patch("subprocess.Popen")
    def test_full_mode_command(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            with patch("builtins.open", mock_open()):
                d = wd.Daemon()
                d.start_training("full")

        cmd = mock_popen.call_args[0][0]
        cmd_str = " ".join(cmd)
        self.assertIn("train.py", cmd_str)
        self.assertIn("--size=65M", cmd_str)
        self.assertIn("--batch_size=4", cmd_str)
        self.assertNotIn("nice", cmd_str)

    @patch("subprocess.Popen")
    def test_low_mode_uses_nice(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            with patch("builtins.open", mock_open()):
                d = wd.Daemon()
                d.start_training("low")

        cmd = mock_popen.call_args[0][0]
        self.assertEqual(cmd[0], "nice")
        self.assertEqual(cmd[1], "-n")
        self.assertEqual(cmd[2], "15")
        cmd_str = " ".join(cmd)
        self.assertIn("--batch_size=2", cmd_str)

    @patch("subprocess.Popen")
    def test_pause_mode_does_not_start(self, mock_popen):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            with patch("builtins.open", mock_open()):
                d = wd.Daemon()
                d.start_training("pause")

        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    def test_start_sets_current_mode(self, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            with patch("builtins.open", mock_open()):
                d = wd.Daemon()
                d.start_training("low")
        self.assertEqual(d._current_mode, "low")


class TestProcessExit(unittest.TestCase):
    """Daemon._handle_process_exit() resets/increments failures and triggers cooldown."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            d._log_handle = None
            return d

    def test_success_resets_failures(self):
        d = self._daemon()
        d._consecutive_failures = 2
        d.proc = MagicMock(returncode=0)
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()
        d._handle_process_exit()
        self.assertEqual(d._consecutive_failures, 0)
        self.assertEqual(d.state["epoch"], 1)

    def test_failure_increments_counter(self):
        d = self._daemon()
        d._consecutive_failures = 0
        d.proc = MagicMock(returncode=1)
        d._handle_process_exit()
        self.assertEqual(d._consecutive_failures, 1)

    @patch("time.sleep")
    def test_max_failures_triggers_cooldown(self, mock_sleep):
        d = self._daemon()
        d._consecutive_failures = 2  # one below MAX (3)
        d.proc = MagicMock(returncode=1)
        d._handle_process_exit()
        mock_sleep.assert_called_once_with(600)

    def test_success_calls_validate_and_push(self):
        d = self._daemon()
        d.proc = MagicMock(returncode=0)
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()
        d._handle_process_exit()
        d.validate_checkpoint.assert_called_once_with("65M")
        d.on_epoch_complete.assert_called_once()


class TestHungProcess(unittest.TestCase):
    """Daemon._check_hung_process() kills stale training."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_kills_hung_process(self):
        d = self._daemon()
        d._training_start = time.time() - 90000  # 25 hours ago
        d.proc = MagicMock()
        d.proc.poll.return_value = None
        d._check_hung_process()
        d.proc.kill.assert_called_once()

    def test_does_not_kill_fresh_process(self):
        d = self._daemon()
        d._training_start = time.time() - 300  # 5 minutes ago
        d.proc = MagicMock()
        d.proc.poll.return_value = None
        d._check_hung_process()
        d.proc.kill.assert_not_called()


class TestStartPauseLoop(unittest.TestCase):
    """Verify start/pause behavior."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    @patch("subprocess.Popen")
    def test_pause_kills_running_process(self, mock_popen):
        d = self._daemon()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        d.proc = mock_proc
        d.pause_training()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=10)

    def test_pause_does_nothing_when_not_running(self):
        d = self._daemon()
        d.proc = None
        d.pause_training()

    def test_pause_does_nothing_when_already_exited(self):
        d = self._daemon()
        d.proc = MagicMock()
        d.proc.poll.return_value = 0
        d.pause_training()
        d.proc.terminate.assert_not_called()

    @patch("subprocess.Popen")
    def test_start_does_not_double_launch(self, mock_popen):
        d = self._daemon()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        d.proc = mock_proc
        d.start_training("full")
        mock_popen.assert_not_called()


class TestDaemonRunLoop(unittest.TestCase):
    """Test the main run() loop logic paths."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def _break_after_one(self, mock_sleep):
        call_count = [0]
        def limited_sleep(secs):
            call_count[0] += 1
            if call_count[0] >= 1:
                raise KeyboardInterrupt
        mock_sleep.side_effect = limited_sleep

    @patch("time.sleep")
    def test_full_mode_starts_training(self, mock_sleep):
        self._break_after_one(mock_sleep)
        d = self._daemon()
        d.monitor.get_power_mode = MagicMock(return_value=("full", {"disk": 20, "cpu": 30, "ram": 8}))
        d.start_training = MagicMock()
        d.state = {"epoch": 0, "total": 3, "size": "65M"}

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        d.start_training.assert_called_with("full")

    @patch("time.sleep")
    def test_low_mode_starts_training_with_low(self, mock_sleep):
        """Low mode still starts training (with batch=1)."""
        self._break_after_one(mock_sleep)
        d = self._daemon()
        d.monitor.get_power_mode = MagicMock(return_value=("low", {"disk": 20, "cpu": 80, "ram": 3}))
        d.start_training = MagicMock()
        d.state = {"epoch": 0, "total": 3, "size": "65M"}

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        d.start_training.assert_called_with("low")

    @patch("time.sleep")
    def test_pause_mode_does_not_start(self, mock_sleep):
        """Pause mode (disk critical) should not start training."""
        self._break_after_one(mock_sleep)
        d = self._daemon()
        d.monitor.get_power_mode = MagicMock(return_value=("pause", {"disk": 2, "cpu": 30, "ram": 8}))
        d.start_training = MagicMock()
        d.state = {"epoch": 0, "total": 3, "size": "65M"}

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        d.start_training.assert_not_called()

    @patch("time.sleep")
    def test_pause_kills_running_training(self, mock_sleep):
        """Pause mode kills running training (disk emergency)."""
        self._break_after_one(mock_sleep)
        d = self._daemon()
        d.monitor.get_power_mode = MagicMock(return_value=("pause", {"disk": 2, "cpu": 30, "ram": 8}))
        d.pause_training = MagicMock()
        d._check_hung_process = MagicMock()
        d._check_ram_pressure = MagicMock()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        d.proc = mock_proc

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        d.pause_training.assert_called()

    @patch("time.sleep")
    def test_mode_transition_updates_current_mode(self, mock_sleep):
        """Mode transition full->low updates _current_mode without restarting training."""
        self._break_after_one(mock_sleep)
        d = self._daemon()
        d._current_mode = "full"
        d.monitor.get_power_mode = MagicMock(return_value=("low", {"disk": 20, "cpu": 80, "ram": 3}))
        d.pause_training = MagicMock()
        d.start_training = MagicMock()
        d._check_hung_process = MagicMock()
        d._check_ram_pressure = MagicMock()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        d.proc = mock_proc

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        # Watchdog no longer restarts on mode change -- training continues at current batch
        d.pause_training.assert_not_called()
        self.assertEqual(d._current_mode, "low")

    @patch("time.sleep")
    def test_same_mode_does_not_restart(self, mock_sleep):
        """If mode hasn't changed, don't restart training."""
        self._break_after_one(mock_sleep)
        d = self._daemon()
        d._current_mode = "low"
        d.monitor.get_power_mode = MagicMock(return_value=("low", {"disk": 20, "cpu": 80, "ram": 3}))
        d.pause_training = MagicMock()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        d.proc = mock_proc

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        d.pause_training.assert_not_called()

    @patch("time.sleep")
    def test_completed_epochs_stops_loop(self, mock_sleep):
        """When all epochs done and no promotion, run() should exit."""
        d = self._daemon()
        d.monitor.get_power_mode = MagicMock(return_value=("full", {"disk": 20, "cpu": 30, "ram": 8}))
        d.state = {"epoch": 3, "total": 3, "size": "125M"}

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        d.proc = mock_proc
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()

        d.run()


class TestEpochCompletion(unittest.TestCase):
    """Epoch completion triggers auto-push and state update."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            d._log_handle = None
            return d

    def test_epoch_increments_on_success(self):
        d = self._daemon()
        d.proc = MagicMock(returncode=0)
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()
        d.state = {"epoch": 0, "total": 3, "size": "65M"}
        d._handle_process_exit()
        self.assertEqual(d.state["epoch"], 1)

    def test_epoch_does_not_increment_on_failure(self):
        d = self._daemon()
        d.proc = MagicMock(returncode=1)
        d.state = {"epoch": 0, "total": 3, "size": "65M"}
        d._handle_process_exit()
        self.assertEqual(d.state["epoch"], 0)

    def test_multiple_epochs_accumulate(self):
        d = self._daemon()
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()
        d.state = {"epoch": 0, "total": 5, "size": "65M"}

        for i in range(3):
            d.proc = MagicMock(returncode=0)
            d._training_start = None
            d._handle_process_exit()

        self.assertEqual(d.state["epoch"], 3)

    @patch("subprocess.run")
    def test_on_epoch_complete_calls_auto_push(self, mock_run):
        d = self._daemon()
        d.on_epoch_complete()
        push_calls = [c for c in mock_run.call_args_list if "auto_push.sh" in str(c)]
        self.assertEqual(len(push_calls), 1)
        cmd = push_calls[0][0][0]
        self.assertIn("auto_push.sh", cmd[-1])

    @patch("subprocess.run", side_effect=Exception("push failed"))
    def test_on_epoch_complete_handles_push_failure(self, mock_run):
        d = self._daemon()
        d.on_epoch_complete()


class TestFailureCooldown(unittest.TestCase):
    """Consecutive failures trigger cooldown, then reset on success."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            d._log_handle = None
            return d

    def test_single_failure_no_cooldown(self):
        d = self._daemon()
        d.proc = MagicMock(returncode=1)
        with patch("time.sleep") as mock_sleep:
            d._handle_process_exit()
            mock_sleep.assert_not_called()

    def test_two_failures_no_cooldown(self):
        d = self._daemon()
        d._consecutive_failures = 1
        d.proc = MagicMock(returncode=1)
        with patch("time.sleep") as mock_sleep:
            d._handle_process_exit()
            mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_three_failures_triggers_cooldown(self, mock_sleep):
        d = self._daemon()
        d._consecutive_failures = 2
        d.proc = MagicMock(returncode=1)
        d._handle_process_exit()
        mock_sleep.assert_called_once_with(600)
        self.assertEqual(d._consecutive_failures, 3)

    def test_success_after_failures_resets_counter(self):
        d = self._daemon()
        d._consecutive_failures = 2
        d.proc = MagicMock(returncode=0)
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()
        d._handle_process_exit()
        self.assertEqual(d._consecutive_failures, 0)

    @patch("time.sleep")
    def test_failure_still_returns_true(self, mock_sleep):
        d = self._daemon()
        d.proc = MagicMock(returncode=1)
        result = d._handle_process_exit()
        self.assertTrue(result)


class TestTrainingCommand(unittest.TestCase):
    """Training subprocess command construction edge cases."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    @patch("subprocess.Popen")
    def test_command_uses_python_executable(self, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        d = self._daemon()
        with patch("builtins.open", mock_open()):
            d.start_training("full")
        cmd = mock_popen.call_args[0][0]
        self.assertEqual(cmd[0], sys.executable)

    @patch("subprocess.Popen")
    def test_command_uses_correct_script(self, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        d = self._daemon()
        with patch("builtins.open", mock_open()):
            d.start_training("full")
        cmd = mock_popen.call_args[0][0]
        self.assertTrue(cmd[1].endswith("train.py"))

    @patch("subprocess.Popen")
    def test_low_mode_python_after_nice(self, mock_popen):
        """In low mode, python executable comes after nice -n 15."""
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        d = self._daemon()
        with patch("builtins.open", mock_open()):
            d.start_training("low")
        cmd = mock_popen.call_args[0][0]
        self.assertEqual(cmd[0], "nice")
        self.assertEqual(cmd[3], sys.executable)

    @patch("subprocess.Popen")
    def test_125m_uses_correct_steps(self, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        d = self._daemon()
        d.state["size"] = "125M"
        with patch("builtins.open", mock_open()):
            d.start_training("full")
        cmd_str = " ".join(mock_popen.call_args[0][0])
        self.assertIn("--steps=500", cmd_str)
        self.assertIn("--size=125M", cmd_str)

    @patch("subprocess.Popen")
    def test_sets_training_start_time(self, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        d = self._daemon()
        with patch("builtins.open", mock_open()):
            d.start_training("full")
        self.assertIsNotNone(d._training_start)

    @patch("subprocess.Popen")
    def test_saves_state_after_start(self, mock_popen):
        mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
        d = self._daemon()
        with patch("builtins.open", mock_open()):
            d.start_training("full")
        d.save_state.assert_called()
        self.assertIn("training_started", d.state)


class TestCheckpointValidation(unittest.TestCase):
    """Checkpoint validation after training."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_missing_checkpoint_returns_false(self):
        d = self._daemon()
        self.assertFalse(d.validate_checkpoint("nonexistent_size"))


class TestLogHandleCleanup(unittest.TestCase):
    """Log handle cleanup prevents file descriptor leaks."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_cleanup_closes_handle(self):
        d = self._daemon()
        mock_handle = MagicMock()
        d._log_handle = mock_handle
        d._cleanup_log_handle()
        mock_handle.close.assert_called_once()
        self.assertIsNone(d._log_handle)

    def test_cleanup_none_handle_is_safe(self):
        d = self._daemon()
        d._log_handle = None
        d._cleanup_log_handle()

    def test_cleanup_handles_os_error(self):
        d = self._daemon()
        mock_handle = MagicMock()
        mock_handle.close.side_effect = OSError("already closed")
        d._log_handle = mock_handle
        d._cleanup_log_handle()
        self.assertIsNone(d._log_handle)


class TestStateCorruption(unittest.TestCase):
    """Daemon should handle corrupt or missing state gracefully."""

    def test_corrupt_json_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f.flush()
            try:
                with _psutil_mock(), patch.object(wd, "STATE_FILE", Path(f.name)):
                    with self.assertRaises(json.JSONDecodeError):
                        wd.Daemon()
            finally:
                os.unlink(f.name)

    def test_empty_file_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            f.flush()
            try:
                with _psutil_mock(), patch.object(wd, "STATE_FILE", Path(f.name)):
                    with self.assertRaises(json.JSONDecodeError):
                        wd.Daemon()
            finally:
                os.unlink(f.name)

    def test_missing_keys_in_state(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"epoch": 1}, f)
            f.flush()
            try:
                with _psutil_mock(), patch.object(wd, "STATE_FILE", Path(f.name)):
                    d = wd.Daemon()
                    self.assertEqual(d.state["epoch"], 1)
            finally:
                os.unlink(f.name)


class TestPromotionEdgeCases(unittest.TestCase):
    """Promotion logic edge cases."""

    def _daemon(self, state):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.state = state
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            d.generate_roadmap = MagicMock()
            return d

    def test_promote_resets_epoch_to_zero(self):
        d = self._daemon({"epoch": 3, "total": 3, "size": "65M"})
        d.promote_size()
        self.assertEqual(d.state["epoch"], 0)

    def test_promote_saves_state(self):
        d = self._daemon({"epoch": 3, "total": 3, "size": "65M"})
        d.promote_size()
        d.save_state.assert_called_once()

    def test_full_lifecycle_65m_to_125m_to_done(self):
        d = self._daemon({"epoch": 0, "total": 3, "size": "65M"})
        d.validate_checkpoint = MagicMock()
        d.on_epoch_complete = MagicMock()

        for i in range(3):
            d.proc = MagicMock(returncode=0)
            d._log_handle = None
            d._training_start = None
            result = d._handle_process_exit()
            self.assertTrue(result)

        self.assertEqual(d.state["size"], "125M")
        self.assertEqual(d.state["epoch"], 0)

        for i in range(3):
            d.proc = MagicMock(returncode=0)
            d._log_handle = None
            d._training_start = None
            result = d._handle_process_exit()
            if i < 2:
                self.assertTrue(result)
            else:
                self.assertFalse(result)


class TestHungProcessEdgeCases(unittest.TestCase):
    """Hung process detection edge cases."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_no_training_start_is_safe(self):
        d = self._daemon()
        d._training_start = None
        d.proc = MagicMock()
        d._check_hung_process()
        d.proc.kill.assert_not_called()

    @patch("time.time")
    def test_exactly_at_limit_does_not_kill(self, mock_time):
        d = self._daemon()
        mock_time.return_value = 10000.0
        d._training_start = 10000.0 - 86400
        d.proc = MagicMock()
        d.proc.poll.return_value = None
        d._check_hung_process()
        d.proc.kill.assert_not_called()

    def test_one_second_over_limit_kills(self):
        d = self._daemon()
        d._training_start = time.time() - 86401
        d.proc = MagicMock()
        d.proc.poll.return_value = None
        d._check_hung_process()
        d.proc.kill.assert_called_once()


class TestWatchdogPauseFlag(unittest.TestCase):
    """Watchdog-initiated kills should not count as failures."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            d._log_handle = None
            return d

    def test_pause_sets_flag(self):
        d = self._daemon()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        d.proc = mock_proc
        d.pause_training()
        self.assertTrue(d._paused_by_watchdog)

    def test_watchdog_kill_not_counted_as_failure(self):
        d = self._daemon()
        d._paused_by_watchdog = True
        d._consecutive_failures = 0
        d.proc = MagicMock(returncode=-15)
        result = d._handle_process_exit()
        self.assertTrue(result)
        self.assertEqual(d._consecutive_failures, 0)
        self.assertFalse(d._paused_by_watchdog)

    def test_real_failure_still_counted(self):
        d = self._daemon()
        d._paused_by_watchdog = False
        d._consecutive_failures = 0
        d.proc = MagicMock(returncode=1)
        d._handle_process_exit()
        self.assertEqual(d._consecutive_failures, 1)

    @patch("time.sleep")
    def test_watchdog_kills_never_trigger_cooldown(self, mock_sleep):
        """Three watchdog pauses in a row should not trigger cooldown."""
        d = self._daemon()
        for _ in range(4):
            d._paused_by_watchdog = True
            d.proc = MagicMock(returncode=-15)
            d._handle_process_exit()
        self.assertEqual(d._consecutive_failures, 0)
        mock_sleep.assert_not_called()

    def test_flag_init_false(self):
        d = self._daemon()
        self.assertFalse(d._paused_by_watchdog)


class TestDiskPauseClearsProc(unittest.TestCase):
    """Disk-critical pause path must clear self.proc to prevent _handle_process_exit."""

    @patch("time.sleep")
    def test_disk_pause_clears_proc(self, mock_sleep):
        self._break_after_one(mock_sleep)
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()

        d.monitor.get_power_mode = MagicMock(return_value=("pause", {"disk": 2, "cpu": 30, "ram": 8}))
        d._check_hung_process = MagicMock()
        d._check_ram_pressure = MagicMock()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = None
        d.proc = mock_proc

        try:
            d.run()
        except KeyboardInterrupt:
            pass

        self.assertIsNone(d.proc)

    def _break_after_one(self, mock_sleep):
        call_count = [0]
        def limited_sleep(secs):
            call_count[0] += 1
            if call_count[0] >= 1:
                raise KeyboardInterrupt
        mock_sleep.side_effect = limited_sleep


class TestTrainingLogTail(unittest.TestCase):
    """_tail_training_log reads last N lines from training.log."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_reads_last_5_lines(self):
        d = self._daemon()
        content = "\n".join([f"line {i}" for i in range(10)])
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=content):
            result = d._tail_training_log(5)
        self.assertEqual(result, "line 5\nline 6\nline 7\nline 8\nline 9")

    def test_missing_log_returns_fallback(self):
        d = self._daemon()
        with patch.object(Path, "exists", return_value=False):
            result = d._tail_training_log()
        self.assertEqual(result, "(no training log)")


class TestSizeConfig(unittest.TestCase):
    """SIZE_CONFIG has expected entries."""

    def test_65m_config(self):
        cfg = wd.SIZE_CONFIG["65M"]
        self.assertEqual(cfg["steps"], 100000)
        self.assertIn("seq_len", cfg)

    def test_125m_config(self):
        cfg = wd.SIZE_CONFIG["125M"]
        self.assertEqual(cfg["steps"], 50000)
        self.assertIn("seq_len", cfg)


class TestSendIMessage(unittest.TestCase):
    """send_imessage() calls openclaw with correct args."""

    def _daemon_real_imessage(self):
        """Daemon with real send_imessage (not mocked) for testing the method itself."""
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            return d

    @patch("subprocess.run")
    def test_send_imessage_calls_openclaw(self, mock_run):
        d = self._daemon_real_imessage()
        d.send_imessage("test message")
        mock_run.assert_called_once_with(
            ["/opt/homebrew/bin/openclaw", "message", "send",
             "--channel", "imessage",
             "--target", "trommatic@icloud.com",
             "--message", "test message"],
            capture_output=True, timeout=15
        )

    @patch("subprocess.run", side_effect=Exception("openclaw not found"))
    def test_send_imessage_handles_failure(self, mock_run):
        d = self._daemon_real_imessage()
        d.send_imessage("test")  # should not raise


class TestCheckMilestones(unittest.TestCase):
    """Milestone detection from training log parsing."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_step_milestone_detected(self):
        d = self._daemon()
        d._parse_latest_training_progress = MagicMock(return_value=(1500, 3.5))
        d.check_milestones()
        # Should trigger step_1000 milestone
        calls = [c[0][0] for c in d.send_imessage.call_args_list]
        self.assertTrue(any("reached 1000 steps" in c for c in calls))

    def test_loss_milestone_detected(self):
        d = self._daemon()
        d._parse_latest_training_progress = MagicMock(return_value=(500, 2.8))
        d.check_milestones()
        calls = [c[0][0] for c in d.send_imessage.call_args_list]
        self.assertTrue(any("loss dropped below 3.0" in c for c in calls))
        self.assertTrue(any("loss dropped below 4.0" in c for c in calls))

    def test_milestone_not_re_sent(self):
        d = self._daemon()
        d._parse_latest_training_progress = MagicMock(return_value=(1500, 3.5))
        d.check_milestones()
        count1 = d.send_imessage.call_count
        d.check_milestones()
        count2 = d.send_imessage.call_count
        self.assertEqual(count1, count2)

    def test_no_progress_skips_check(self):
        d = self._daemon()
        d._parse_latest_training_progress = MagicMock(return_value=(None, None))
        d.check_milestones()
        d.send_imessage.assert_not_called()

    def test_milestones_persisted_to_state(self):
        d = self._daemon()
        d._parse_latest_training_progress = MagicMock(return_value=(1500, 3.5))
        d.check_milestones()
        self.assertIn('notified_milestones', d.state)
        self.assertIn('step_1000', d.state['notified_milestones'])


class TestParseTrainingProgress(unittest.TestCase):
    """_parse_latest_training_progress() extracts step/loss from log."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_parse_step_loss_format(self):
        d = self._daemon()
        log_content = "Step: 5000 Loss: 2.345\n"
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=log_content):
            step, loss = d._parse_latest_training_progress()
        self.assertEqual(step, 5000)
        self.assertAlmostEqual(loss, 2.345)

    def test_parse_tabular_format(self):
        d = self._daemon()
        log_content = "  1000    3.456  0.001\n"
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=log_content):
            step, loss = d._parse_latest_training_progress()
        self.assertEqual(step, 1000)
        self.assertAlmostEqual(loss, 3.456)

    def test_missing_log_returns_none(self):
        d = self._daemon()
        with patch.object(Path, "exists", return_value=False):
            step, loss = d._parse_latest_training_progress()
        self.assertIsNone(step)
        self.assertIsNone(loss)


class TestGenerateRoadmap(unittest.TestCase):
    """generate_roadmap() creates markdown roadmap file."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    def test_creates_roadmap_file(self):
        d = self._daemon()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(wd, "ARTHUR_ROOT", Path(tmpdir)):
                path = d.generate_roadmap("65M", 100000, 1.234)
                self.assertTrue(path.exists())
                content = path.read_text()
                self.assertIn("65M", content)
                self.assertIn("100000 steps", content)
                self.assertIn("1.2340", content)

    def test_65m_roadmap_mentions_125m(self):
        d = self._daemon()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(wd, "ARTHUR_ROOT", Path(tmpdir)):
                path = d.generate_roadmap("65M", 100000, 1.5)
                content = path.read_text()
                self.assertIn("125M", content)
                self.assertIn("Promoting", content)

    def test_125m_roadmap_mentions_instruction_tuning(self):
        d = self._daemon()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(wd, "ARTHUR_ROOT", Path(tmpdir)):
                path = d.generate_roadmap("125M", 50000, 0.8)
                content = path.read_text()
                self.assertIn("instruction", content.lower())
                self.assertIn("ONNX", content)

    def test_sends_imessage_summary(self):
        d = self._daemon()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(wd, "ARTHUR_ROOT", Path(tmpdir)):
                d.generate_roadmap("65M", 100000, 1.5)
                d.send_imessage.assert_called_once()
                msg = d.send_imessage.call_args[0][0]
                self.assertIn("65M", msg)


class TestEpochCompleteIMessage(unittest.TestCase):
    """on_epoch_complete() sends iMessage."""

    def _daemon(self):
        with _psutil_mock(), patch.object(wd, "STATE_FILE", Path("/nonexistent/state.json")):
            d = wd.Daemon()
            d.save_state = MagicMock()
            d.send_imessage = MagicMock()
            return d

    @patch("subprocess.run")
    def test_epoch_complete_sends_imessage(self, mock_run):
        d = self._daemon()
        d.send_imessage = MagicMock()
        d.state = {"epoch": 1, "total": 3, "size": "65M"}
        d.on_epoch_complete()
        d.send_imessage.assert_called_once()
        msg = d.send_imessage.call_args[0][0]
        self.assertIn("epoch 2/3", msg)
        self.assertIn("65M", msg)

    @patch("subprocess.run")
    def test_failure_cooldown_sends_imessage(self, mock_run):
        d = self._daemon()
        d.send_imessage = MagicMock()
        d._consecutive_failures = 2
        d.proc = MagicMock(returncode=1)
        d._log_handle = None
        with patch("time.sleep"):
            d._handle_process_exit()
        d.send_imessage.assert_called_once()
        msg = d.send_imessage.call_args[0][0]
        self.assertIn("failed 3x", msg)


class TestMilestoneState(unittest.TestCase):
    """Milestone state restores from daemon_state.json."""

    def test_milestones_restored_from_state(self):
        state = {"epoch": 0, "total": 3, "size": "65M", "notified_milestones": ["step_1000", "loss_4.0"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(state, f)
            f.flush()
            try:
                with _psutil_mock(), patch.object(wd, "STATE_FILE", Path(f.name)):
                    d = wd.Daemon()
                    self.assertIn("step_1000", d._notified_milestones)
                    self.assertIn("loss_4.0", d._notified_milestones)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
