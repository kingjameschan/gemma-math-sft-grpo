#!/bin/bash
# Lock Windows screen (NOT sleep — WSL/training keep running). Called after E2E fully done.
/mnt/c/Windows/System32/rundll32.exe user32.dll,LockWorkStation
echo "screen locked at $(date)"
