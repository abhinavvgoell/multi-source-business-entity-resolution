#!/usr/bin/env python3
"""Root entry point for the business entity-resolution pipeline.

The challenge submission must ship the implementation inside the runnable code
folder, so that copy is the single source of truth:

    student_resource/code/business_entity_resolution/src/business_entity_resolution.py

This wrapper executes it with the caller's arguments so the project-root
commands in README.md keep working and the two copies cannot drift apart.
"""
from __future__ import annotations

import runpy
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parent
    / "student_resource"
    / "code"
    / "business_entity_resolution"
    / "src"
    / "business_entity_resolution.py"
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
