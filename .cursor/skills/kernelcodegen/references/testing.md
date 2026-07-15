<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Kernel testing (bypass for now)

Kernel-on-HW testing is **bypassed** in the current kernelcodegen workflow.

- **Generate** and **compile** are the active steps; the kernel ELF is produced at `worklocal/build/kernel`.
- When kernel testing is added later: it may involve loading the kernel ELF onto AIE tiles (e.g. via host/runtime or board scripts) and checking tile execution. Until then, treat the “Test” step as optional and skip it when using this skill.

For host-side HW run and console verification, use the **hostcodegen** skill (e.g. piplinerun.sh or apppaltest.py).
