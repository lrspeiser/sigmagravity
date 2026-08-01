param(
    [string]$Distribution = "Ubuntu-24.04",
    [string]$CiaoEnvironment = "/home/henry/ciao-4.18"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DriveRoot = [System.IO.Path]::GetPathRoot($ProjectRoot)
if ($DriveRoot -notmatch '^[A-Za-z]:\\$') {
    throw "The project root must be on a Windows drive that WSL mounts under /mnt."
}
$DriveLetter = $DriveRoot.Substring(0, 1).ToLowerInvariant()
$RelativeProjectRoot = $ProjectRoot.Substring($DriveRoot.Length).Replace('\', '/')
$LinuxProjectRoot = "/mnt/$DriveLetter/$RelativeProjectRoot"

$ReductionScript = "$LinuxProjectRoot/scripts/reduce_rxj2129_chandra.sh"
$Command = "source '/home/henry/miniforge3/etc/profile.d/conda.sh' && conda activate '$CiaoEnvironment' && bash '$ReductionScript' '$LinuxProjectRoot'"
& wsl.exe -d $Distribution -- bash -lc $Command
if ($LASTEXITCODE -ne 0) {
    throw "The frozen CIAO reduction failed with exit code $LASTEXITCODE."
}
