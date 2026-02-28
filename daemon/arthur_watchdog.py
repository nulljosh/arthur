#!/usr/bin/env python3
"""
Arthur LLM Training Watchdog Daemon
Autonomous continuous training with resource management.
"""

import os, sys, json, time, psutil, subprocess, logging
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

class ResourceMonitor:
    def __init__(self):
        self.storage_limit_gb = 5
        self.cpu_limit = 70
        self.ram_min_gb = 4
    
    def get_disk_free_gb(self):
        stat = os.statvfs('/')
        return stat.f_bavail * stat.f_frsize / 1024 / 1024 / 1024
    
    def get_cpu_percent(self):
        return psutil.cpu_percent(interval=1)
    
    def get_ram_available_gb(self):
        return psutil.virtual_memory().available / 1024 / 1024 / 1024
    
    def is_healthy(self):
        disk = self.get_disk_free_gb()
        cpu = self.get_cpu_percent()
        ram = self.get_ram_available_gb()
        
        if disk < self.storage_limit_gb:
            logger.warning(f"Low disk: {disk:.1f}GB")
            return False
        if cpu > self.cpu_limit:
            logger.warning(f"High CPU: {cpu:.1f}%")
            return False
        if ram < self.ram_min_gb:
            logger.warning(f"Low RAM: {ram:.1f}GB")
            return False
        
        logger.info(f"OK: disk={disk:.1f}GB cpu={cpu:.1f}% ram={ram:.1f}GB")
        return True
    
    def get_batch_size(self):
        ram = self.get_ram_available_gb()
        return 32 if ram > 8 else (16 if ram > 6 else 8)

class Daemon:
    def __init__(self):
        self.monitor = ResourceMonitor()
        self.state = self.load_state()
        self.proc = None
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {'epoch': 0, 'total': 50}
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f)
    
    def start_training(self):
        if self.proc and self.proc.poll() is None:
            return
        
        batch = self.monitor.get_batch_size()
        epoch = self.state['epoch']
        remaining = self.state['total'] - epoch
        
        cmd = [
            sys.executable,
            str(ARTHUR_ROOT / "src/train_v2.py"),
            f"--batch_size={batch}",
            f"--epochs={remaining}",
        ]
        
        logger.info(f"Training epoch {epoch}/{self.state['total']} (batch={batch})")
        self.proc = subprocess.Popen(
            cmd, cwd=str(ARTHUR_ROOT),
            stdout=open(LOG_DIR / "training.log", "a"),
            stderr=subprocess.STDOUT
        )
        self.state['training_started'] = datetime.now().isoformat()
        self.save_state()
    
    def pause_training(self):
        if not self.proc or self.proc.poll() is not None:
            return
        logger.info("Pausing...")
        self.proc.terminate()
        self.proc.wait(timeout=10)
    
    def run(self):
        logger.info("Watchdog started")
        while True:
            try:
                healthy = self.monitor.is_healthy()
                training = self.proc and self.proc.poll() is None
                
                if healthy and not training and self.state['epoch'] < self.state['total']:
                    self.start_training()
                elif not healthy and training:
                    self.pause_training()
                elif training and self.proc.poll() is not None:
                    self.state['epoch'] += 1
                    self.save_state()
                    logger.info(f"Epoch complete. Total: {self.state['epoch']}/{self.state['total']}")
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)

if __name__ == '__main__':
    Daemon().run()
