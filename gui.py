#!/usr/bin/env python3
"""
gui.py
------
Small Tkinter GUI wrapper around harmony_analysis.pipeline.run_analysis.
The GUI contains NO analysis logic itself — it only calls the same
`run_analysis()` function used by the CLI, so the core stays testable.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from harmony_analysis.pipeline import run_analysis


class HarmonyAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HARMONY Handover Analysis")
        self.geometry("560x360")
        self.resizable(False, False)

        self.experiment_folder = tk.StringVar()
        self.output_folder = tk.StringVar(value=str(Path.cwd() / "analysis_output"))
        self.status_var = tk.StringVar(value="Ready.")
        self.result_summary_var = tk.StringVar(value="")

        self._build_widgets()
        self._last_output_dir: Path | None = None

    def _build_widgets(self):
        pad = {"padx": 10, "pady": 6}

        frm_in = ttk.LabelFrame(self, text="1. Select Experiment Folder")
        frm_in.pack(fill="x", **pad)
        ttk.Entry(frm_in, textvariable=self.experiment_folder, width=55).pack(side="left", padx=6, pady=6)
        ttk.Button(frm_in, text="Browse...", command=self._pick_experiment_folder).pack(side="left", padx=6)

        frm_out = ttk.LabelFrame(self, text="2. Select Output Folder")
        frm_out.pack(fill="x", **pad)
        ttk.Entry(frm_out, textvariable=self.output_folder, width=55).pack(side="left", padx=6, pady=6)
        ttk.Button(frm_out, text="Browse...", command=self._pick_output_folder).pack(side="left", padx=6)

        frm_run = ttk.Frame(self)
        frm_run.pack(fill="x", **pad)
        self.analyze_btn = ttk.Button(frm_run, text="Analyze", command=self._start_analysis)
        self.analyze_btn.pack(side="left")
        self.open_btn = ttk.Button(frm_run, text="Open Output Folder", command=self._open_output_folder,
                                    state="disabled")
        self.open_btn.pack(side="left", padx=10)

        frm_progress = ttk.LabelFrame(self, text="Progress")
        frm_progress.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(frm_progress, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(frm_progress, textvariable=self.status_var, wraplength=520, justify="left").pack(
            fill="x", padx=10, pady=4)
        ttk.Label(frm_progress, textvariable=self.result_summary_var, wraplength=520, justify="left",
                  font=("TkDefaultFont", 10, "bold")).pack(fill="x", padx=10, pady=(4, 10))

    def _pick_experiment_folder(self):
        folder = filedialog.askdirectory(title="Select HARMONY_Experiments root folder")
        if folder:
            self.experiment_folder.set(folder)

    def _pick_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder.set(folder)

    def _start_analysis(self):
        root = self.experiment_folder.get().strip()
        out = self.output_folder.get().strip()
        if not root:
            messagebox.showerror("Missing input", "Please select an experiment folder first.")
            return
        if not Path(root).exists():
            messagebox.showerror("Invalid folder", f"Folder does not exist:\n{root}")
            return

        self.analyze_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self.progress.config(mode="indeterminate")
        self.progress.start(12)
        self.status_var.set("Starting analysis...")
        self.result_summary_var.set("")

        thread = threading.Thread(target=self._run_worker, args=(root, out), daemon=True)
        thread.start()

    def _run_worker(self, root: str, out: str):
        try:
            def progress_cb(msg, cur, total):
                self.after(0, self._update_progress, msg, cur, total)

            result = run_analysis(root=root, output_dir=out, progress_cb=progress_cb)
            self.after(0, self._on_success, result)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._on_error, exc)

    def _update_progress(self, msg: str, cur: int, total: int):
        self.status_var.set(msg)
        if total:
            if str(self.progress["mode"]) != "determinate":
                self.progress.stop()
                self.progress.config(mode="determinate", maximum=total)
            self.progress["value"] = cur

    def _on_success(self, result):
        self.progress.stop()
        self.progress.config(mode="determinate", maximum=max(result.n_trials_found, 1))
        self.progress["value"] = result.n_trials_found
        self.status_var.set("Analysis complete.")
        self.result_summary_var.set(
            f"Trials found: {result.n_trials_found}   "
            f"VALID: {result.n_valid}   VALID_WITH_WARNINGS: {result.n_warnings}   INVALID: {result.n_invalid}"
        )
        self._last_output_dir = Path(result.output_dir)
        self.analyze_btn.config(state="normal")
        self.open_btn.config(state="normal")

    def _on_error(self, exc: Exception):
        self.progress.stop()
        self.status_var.set("Error during analysis.")
        self.analyze_btn.config(state="normal")
        messagebox.showerror("Analysis failed", str(exc))

    def _open_output_folder(self):
        if not self._last_output_dir:
            return
        path = str(self._last_output_dir.resolve())
        try:
            if platform.system() == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not open folder", str(exc))


def main():
    app = HarmonyAnalysisGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
