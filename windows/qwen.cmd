@echo off
rem qwen - expose Qwen3.8-27B (WSL2) to the LAN and (re)start it.
rem Double-click me, or run "qwen" in any terminal.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0qwen.ps1" %*
