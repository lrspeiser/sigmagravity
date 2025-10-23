#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $here "..\..")

# Galaxy figures (Track-2)
python "$here\generate_rc_gallery.py"
python "$here\generate_rar_plot.py"

# Cluster figures: use SigmaGravity harness output if available
$sg_fig = Join-Path $root "projects\SigmaGravity\figures\holdouts_pred_vs_obs.png"
$paper_fig_dir = Join-Path $root "many_path_model\paper_release\figures"
if (Test-Path $sg_fig) {
  New-Item -ItemType Directory -Force -Path $paper_fig_dir | Out-Null
  Copy-Item -Force $sg_fig (Join-Path $paper_fig_dir "holdouts_pred_vs_obs.png")
}
Write-Host "Figures written under many_path_model/paper_release/figures"