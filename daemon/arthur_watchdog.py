#!/usr/bin/env python3
"""
Arthur LLM Training Watchdog Daemon
Autonomous continuous training with resource management.
Trains v3 models (65M -> 125M) on WikiText-2 with safe batch sizes.
"""

import os, sys, json, time, re, traceback, psutil, subprocess, logging
from pathlib import Path
from datetime import datetime

ARTHUR_ROOT = Path(__file__).parent.parent
LOG_DIR = ARTHUR_ROOT / "logs"
STATE_FILE = ARTHUR_ROOT / "daemon_state.json"

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "watchdog.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hard RAM ceiling: 10GB (62% of 16GB). Leaves 6GB for system.
RAM_HARD_CEILING_GB = 10.0
# Grace period (seconds) after training starts before RAM checks kick in.
# WikiText-103 dataset loading temporarily spikes RSS above steady-state.
RAM_CHECK_GRACE_PERIOD = 120
# Swap growth threshold: kill if swap grows >500MB in one check cycle (30s)
SWAP_GROWTH_THRESHOLD_MB = 500

# v3 training config per model size
SIZE_CONFIG = {
    '65M':  {'steps': 100000, 'seq_len': 128, 'grad_accum': 8},
    '125M': {'steps': 50000,  'seq_len': 128, 'grad_accum': 8},
}

class ResourceMonitor:
    def __init__(self):
        self.disk_min_gb = 5
        self.cpu_limit = 70
        self.ram_low_gb = 2
        self._last_swap_used = psutil.swap_memory().used

    def get_disk_free_gb(self):
        stat = os.statvfs('/')
        return stat.f_bavail * stat.f_frsize / 1024 / 1024 / 1024

    def get_cpu_percent(self):
        return psutil.cpu_percent(interval=1)

    def get_ram_available_gb(self):
        return psutil.virtual_memory().available / 1024 / 1024 / 1024

    def check_swap_growth(self):
        """Return swap growth in MB since last check."""
        current = psutil.swap_memory().used
        delta_mb = (current - self._last_swap_used) / 1024 / 1024
        self._last_swap_used = current
        return delta_mb

    def get_power_mode(self):
        disk = self.get_disk_free_gb()
        cpu = self.get_cpu_percent()
        ram = self.get_ram_available_gb()

        if disk < self.disk_min_gb:
            logger.warning(f"Disk critical: {disk:.1f}GB -- pause")
            return "pause", {"disk": disk, "cpu": cpu, "ram": ram}

        if ram < self.ram_low_gb or cpu > self.cpu_limit:
            logger.info(f"Low-power: disk={disk:.1f}GB cpu={cpu:.1f}% ram={ram:.1f}GB")
            return "low", {"disk": disk, "cpu": cpu, "ram": ram}

        logger.info(f"Full power: disk={disk:.1f}GB cpu={cpu:.1f}% ram={ram:.1f}GB")
        return "full", {"disk": disk, "cpu": cpu, "ram": ram}

    def get_batch_size(self, mode):
        if mode == "pause":
            return 0
        if mode == "low":
            return 2
        return 4

class Daemon:
    MAX_TRAINING_SECONDS = 86400  # 24 hours
    MAX_CONSECUTIVE_FAILURES = 3
    FAILURE_COOLDOWN = 600  # 10 minutes
    STEP_MILESTONES = [1000, 5000, 10000, 25000, 50000, 100000, 200000, 300000]
    LOSS_MILESTONES = [4.0, 3.0, 2.0, 1.5, 1.0]

    def __init__(self):
        self.monitor = ResourceMonitor()
        self.state = self.load_state()
        self.proc = None
        self._consecutive_failures = 0
        self._last_alert_step = None  # training step at which last failure alert was sent
        self._training_start = None
        self._log_handle = None
        self._current_mode = "full"
        self._paused_by_watchdog = False
        self._notified_milestones = set(self.state.get('notified_milestones', []))
        self._loop_count = 0

        # Lock to 65M on machines with <24GB RAM
        total_ram_gb = psutil.virtual_memory().total / 1024**3
        if total_ram_gb < 24:
            logger.info(f"{total_ram_gb:.0f}GB machine detected, locked to 65M model")
            if self.state.get('size') != '65M':
                self.state['size'] = '65M'
                self.state['epoch'] = 0
                self.save_state()
        self._max_size = '65M' if total_ram_gb < 24 else '125M'

    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {'epoch': 0, 'total': 3, 'size': '65M'}

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f)

    def start_training(self, mode="full"):
        if self.proc and self.proc.poll() is None:
            return

        batch = self.monitor.get_batch_size(mode)
        if batch == 0:
            return

        size = self.state.get('size', '65M')
        cfg = SIZE_CONFIG.get(size, SIZE_CONFIG['65M'])
        epoch = self.state['epoch']

        cmd = [
            sys.executable,
            str(ARTHUR_ROOT / "scripts/train.py"),
            f"--size={size}",
            f"--batch_size={batch}",
            f"--steps={cfg['steps']}",
            f"--seq_len={cfg['seq_len']}",
            f"--grad_accum={cfg.get('grad_accum', 1)}",
        ]

        # Always pass --resume; train.py handles missing checkpoint gracefully
        cmd.append("--resume")

        if mode == "low":
            cmd = ["nice", "-n", "15"] + cmd

        logger.info(f"v3 training: size={size} epoch={epoch}/{self.state['total']} batch={batch} mode={mode} resume=True")
        self._cleanup_log_handle()
        self._log_handle = open(LOG_DIR / "training.log", "a")
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        self.proc = subprocess.Popen(
            cmd, cwd=str(ARTHUR_ROOT),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env
        )
        self._training_start = time.time()
        self._current_mode = mode
        self.state['training_started'] = datetime.now().isoformat()
        self.save_state()

    def pause_training(self):
        if not self.proc or self.proc.poll() is not None:
            return
        logger.info("Pausing...")
        self._paused_by_watchdog = True
        self.proc.terminate()
        self.proc.wait(timeout=10)

    def promote_size(self):
        """After completing all epochs for current size, promote to next."""
        current = self.state.get('size', '65M')
        if current == '65M':
            if self._max_size == '65M':
                logger.info("65M complete -- skipping 125M promotion (machine too small)")
                step, loss = self._parse_latest_training_progress()
                self.generate_roadmap(current, step or 0, loss or 0.0)
                return False
            logger.info("65M complete -- promoting to 125M")
            step, loss = self._parse_latest_training_progress()
            self.generate_roadmap(current, step or 0, loss or 0.0)
            self.send_imessage(f"Arthur: 65M training complete. Promoting to 125M.")
            self.state['size'] = '125M'
            self.state['epoch'] = 0
            self.state['total'] = 3
            self.save_state()
            return True
        # 125M complete
        step, loss = self._parse_latest_training_progress()
        self.generate_roadmap(current, step or 0, loss or 0.0)
        return False

    def _cleanup_log_handle(self):
        if self._log_handle:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def validate_checkpoint(self, size):
        """Check that at least one checkpoint file exists after training."""
        latest = ARTHUR_ROOT / f"models/arthur_v3_{size}_latest.pt"
        best = ARTHUR_ROOT / f"models/arthur_v3_{size}_best.pt"
        found = False
        for path in [latest, best]:
            if path.exists():
                logger.info(f"Checkpoint OK: {path} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
                found = True
        if not found:
            logger.warning(f"No checkpoints found for {size}")
        return found

    def notify(self, message):
        """Send macOS notification."""
        try:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "Arthur"'],
                capture_output=True, timeout=5
            )
        except Exception:
            pass
        logger.info(f"Notified: {message}")

    def send_imessage(self, message):
        """Send iMessage to Joshua via OpenClaw."""
        try:
            subprocess.run(
                ["/opt/homebrew/bin/openclaw", "message", "send",
                 "--channel", "imessage",
                 "--target", "trommatic@icloud.com",
                 "--message", message],
                capture_output=True, timeout=15
            )
            logger.info(f"iMessage sent: {message}")
        except Exception as e:
            logger.error(f"iMessage failed: {e}")

    def _parse_latest_training_progress(self):
        """Parse last lines of training.log for latest step and loss."""
        log_path = LOG_DIR / "training.log"
        try:
            if not log_path.exists():
                return None, None
            lines = log_path.read_text().strip().splitlines()
            # Search from end for a line with step/loss data
            # Format: "Step 1000 | Loss: 3.456" or similar tabular output
            for line in reversed(lines[-50:]):
                m = re.search(r'(?:step|Step)\s*[=:]\s*(\d+).*?(?:loss|Loss)\s*[=:]\s*([\d.]+)', line)
                if m:
                    return int(m.group(1)), float(m.group(2))
                # Also match tabular: "  1000    3.456"
                m = re.search(r'^\s*(\d+)\s+([\d.]+)\s+', line)
                if m:
                    return int(m.group(1)), float(m.group(2))
        except OSError:
            pass
        return None, None

    def check_milestones(self):
        """Check for step/loss milestones and send iMessage notifications."""
        step, loss = self._parse_latest_training_progress()
        if step is None or loss is None:
            return

        for ms in self.STEP_MILESTONES:
            key = f"step_{ms}"
            if key not in self._notified_milestones and step >= ms:
                self.send_imessage(f"Arthur training: reached {ms} steps (loss: {loss:.3f})")
                self._notified_milestones.add(key)

        for ms in self.LOSS_MILESTONES:
            key = f"loss_{ms}"
            if key not in self._notified_milestones and loss <= ms:
                self.send_imessage(f"Arthur training: loss dropped below {ms} (now {loss:.3f} at step {step})")
                self._notified_milestones.add(key)

        self.state['notified_milestones'] = list(self._notified_milestones)
        self.save_state()

    def generate_roadmap(self, size, steps, loss):
        """Generate a roadmap markdown file after a training phase completes."""
        roadmap_dir = ARTHUR_ROOT / "roadmaps"
        roadmap_dir.mkdir(exist_ok=True)

        sizes = list(SIZE_CONFIG.keys())
        idx = sizes.index(size) if size in sizes else -1
        next_size = sizes[idx + 1] if idx + 1 < len(sizes) else None

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"# Arthur Roadmap: {size} -> Next Phase\nGenerated: {now}\n\n"
        content += f"## Completed\n- Base training: {size} model, {steps} steps, final loss: {loss:.4f}\n"
        content += f"- Checkpoint: models/arthur_v3_{size}_best.pt\n\n"
        content += "## Next Steps\n"

        if next_size:
            next_cfg = SIZE_CONFIG[next_size]
            content += f"### Promoting to {next_size}\n"
            content += f"- Begin {next_size} training ({next_cfg['steps']} steps)\n"
            content += f"- Sequence length: {next_cfg['seq_len']}\n\n"
        else:
            content += "### Base training complete (all sizes done)\n"
            content += "1. Build instruction dataset\n"
            content += '   - Identity prompts ("I am Arthur, a 90M parameter LLM")\n'
            content += "   - Q&A pairs from WikiText knowledge\n"
            content += "   - Basic math examples\n"
            content += "   - Conversation templates\n"
            content += "2. Fine-tune on instruction format (10-50K steps)\n"
            content += "3. Update web UI + CLI to use instruction-tuned checkpoint\n"
            content += "4. ONNX export + quantization for deployment\n"

        version = "v3"
        path = roadmap_dir / f"{version}_{size}.md"
        path.write_text(content)
        logger.info(f"Roadmap generated: {path}")

        summary = f"Arthur: {size} training complete. "
        if next_size:
            summary += f"Promoting to {next_size}."
        else:
            summary += "ALL BASE TRAINING COMPLETE. Ready for instruction tuning."
        self.send_imessage(summary)
        return path

    def on_epoch_complete(self):
        """Called when training run finishes."""
        size = self.state.get('size', '65M')
        epoch = self.state.get('epoch', 0) + 1  # +1 because state increments after this call
        total = self.state.get('total', 3)
        self.send_imessage(f"Arthur: epoch {epoch}/{total} complete for {size}. Pushing to GitHub.")
        try:
            subprocess.run(
                ["bash", str(ARTHUR_ROOT / "daemon/auto_push.sh")],
                cwd=str(ARTHUR_ROOT),
                capture_output=True,
                timeout=30
            )
            logger.info("Pushed to GitHub after epoch completion")
        except Exception as e:
            logger.error(f"Failed to push: {e}")

    def _tail_training_log(self, n=5):
        """Return last n lines of training.log for error context."""
        log_path = LOG_DIR / "training.log"
        try:
            if log_path.exists():
                lines = log_path.read_text().strip().splitlines()
                return "\n".join(lines[-n:])
        except OSError:
            pass
        return "(no training log)"

    def _handle_process_exit(self):
        """Handle a completed training process. Returns True if training should continue."""
        exit_code = self.proc.returncode
        self._cleanup_log_handle()
        size = self.state.get('size', '65M')

        if exit_code != 0:
            if self._paused_by_watchdog:
                logger.info(f"Training paused by watchdog (exit code {exit_code}), not counting as failure")
                self._paused_by_watchdog = False
                self.proc = None
                self._training_start = None
                return True
            self._consecutive_failures += 1
            tail = self._tail_training_log()
            logger.error(f"Training exited with code {exit_code} (failure {self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES})\nLast training output:\n{tail}")
            # Only notify after 3 consecutive failures to suppress transient alerts
            step, _ = self._parse_latest_training_progress()
            if self._consecutive_failures == self.MAX_CONSECUTIVE_FAILURES:
                logger.error(f"Too many failures, cooling down {self.FAILURE_COOLDOWN}s")
                self._last_alert_step = step
                self.send_imessage(f"Arthur: training failed {self._consecutive_failures}x in a row at step {step}, entering cooldown.")
                time.sleep(self.FAILURE_COOLDOWN)
            elif self._consecutive_failures > self.MAX_CONSECUTIVE_FAILURES:
                # After initial alert, only re-alert every 3 additional failures
                if (self._consecutive_failures % self.MAX_CONSECUTIVE_FAILURES) == 0:
                    logger.error(f"Persistent failures ({self._consecutive_failures} total), cooling down {self.FAILURE_COOLDOWN}s")
                    self._last_alert_step = step
                    self.send_imessage(f"Arthur: training still failing ({self._consecutive_failures} consecutive failures) at step {step}, entering cooldown.")
                    time.sleep(self.FAILURE_COOLDOWN)
            self.proc = None
            self._training_start = None
            return True

        self._consecutive_failures = 0
        self._last_alert_step = None
        self.validate_checkpoint(size)
        self.on_epoch_complete()
        self.state["epoch"] += 1
        self.save_state()
        logger.info(f"Epoch complete: {self.state['epoch']}/{self.state['total']} (size={size})")
        self.proc = None
        self._training_start = None

        if self.state['epoch'] >= self.state['total']:
            if not self.promote_size():
                logger.info("All training complete")
                self.notify("Arthur: all training complete")
                self.send_imessage("Arthur: ALL BASE TRAINING COMPLETE. Ready for instruction tuning.")
                return False
        return True

    def _check_hung_process(self):
        """Kill training if it exceeds the time limit."""
        if not self._training_start:
            return
        elapsed = time.time() - self._training_start
        if elapsed > self.MAX_TRAINING_SECONDS:
            logger.error(f"Training hung ({elapsed:.0f}s > {self.MAX_TRAINING_SECONDS}s), killing")
            self.proc.kill()
            self.proc.wait(timeout=10)

    def _check_ram_pressure(self):
        """Kill training if RSS exceeds hard ceiling or swap is thrashing."""
        if not self.proc or self.proc.poll() is not None:
            return
        # Grace period: don't kill during dataset loading
        if self._training_start and (time.time() - self._training_start) < RAM_CHECK_GRACE_PERIOD:
            return

        try:
            p = psutil.Process(self.proc.pid)
            rss_gb = p.memory_info().rss / 1024**3
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        if rss_gb > RAM_HARD_CEILING_GB:
            logger.error(f"Training RSS {rss_gb:.1f}GB exceeds hard ceiling {RAM_HARD_CEILING_GB}GB, killing")
            self.proc.kill()
            self.proc.wait(timeout=10)
            self._paused_by_watchdog = True
            return

        swap_delta_mb = self.monitor.check_swap_growth()
        if swap_delta_mb > SWAP_GROWTH_THRESHOLD_MB:
            logger.error(f"Swap grew {swap_delta_mb:.0f}MB in one cycle (threshold {SWAP_GROWTH_THRESHOLD_MB}MB), killing training")
            self.proc.kill()
            self.proc.wait(timeout=10)
            self._paused_by_watchdog = True

    def run(self):
        logger.info("Watchdog started (v3 mode, always-on)")
        while True:
            try:
                mode, stats = self.monitor.get_power_mode()
                training = self.proc and self.proc.poll() is None

                # Log mode transitions (no restart)
                if mode != self._current_mode:
                    if mode == "low":
                        logger.info(f"Switching to low-power mode (RAM: {stats['ram']:.1f}GB, CPU: {stats['cpu']:.1f}%)")
                    elif mode == "full" and self._current_mode == "low":
                        logger.info(f"Resuming full power (RAM: {stats['ram']:.1f}GB, CPU: {stats['cpu']:.1f}%)")
                    elif mode == "pause":
                        logger.warning(f"Pausing -- disk critical ({stats['disk']:.1f}GB)")

                self._loop_count += 1

                if training:
                    self._check_hung_process()
                    self._check_ram_pressure()
                    # Check milestones every ~5 min (10 iterations of 30s loop)
                    if self._loop_count % 10 == 0:
                        self.check_milestones()
                    # Only kill on pause (disk critical), not on mode change
                    if mode == "pause":
                        self.pause_training()
                        self.proc = None
                        self._training_start = None
                elif self.proc and self.proc.poll() is not None:
                    if not self._handle_process_exit():
                        break
                elif mode != "pause" and self.state['epoch'] < self.state['total']:
                    self.start_training(mode)

                self._current_mode = mode

                time.sleep(30)
            except Exception as e:
                logger.error(f"Error: {traceback.format_exc()}")
                time.sleep(30)

if __name__ == '__main__':
    Daemon().run()
