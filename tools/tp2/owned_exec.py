#!/usr/bin/env python3
"""Run one recorded engine over SSH; stdin EOF cleans up only that child."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.ds4.tp2_launch import process_identity, stop_process


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    options=parser.parse_args()
    command=options.command
    if command[:1]==['--']:command=command[1:]
    if not command:parser.error('engine command required')
    stopped=threading.Event()
    for name in ('SIGINT','SIGTERM','SIGHUP'):
        signal.signal(getattr(signal,name),lambda *_:stopped.set())
    process=subprocess.Popen(command,stdin=subprocess.PIPE,start_new_session=True)
    options.identity.write_text(json.dumps({'pid':process.pid,'identity':process_identity(process.pid),'args':command})+'\n')
    def forward():
        try:
            while block:=os.read(sys.stdin.fileno(),65536):
                process.stdin.write(block);process.stdin.flush()
        except (BrokenPipeError,OSError):
            pass
        finally:stopped.set()
    threading.Thread(target=forward,daemon=True).start()
    try:
        while process.poll() is None and not stopped.wait(.1):pass
        return process.returncode or 0
    finally:stop_process(process,timeout=5)

if __name__=='__main__':raise SystemExit(main())
