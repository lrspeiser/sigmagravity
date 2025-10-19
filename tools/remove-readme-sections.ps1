Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$path = "README.md"
if (-not (Test-Path -LiteralPath $path)) {
  Write-Error "README.md not found in repo root."
  exit 1
}
$content = Get-Content -Raw -LiteralPath $path

# Remove 'Notes on nomenclature' section (bounded by next horizontal rule or heading)
$pat1 = '(?ms)^\s{0,3}#{1,6}\s*Notes on nomenclature[^\r\n]*\r?\n(?:.*?\r?\n)*?(?=^\s*---+\s*$|^\s{0,3}#{1,6}\s|\z)'
$content = [regex]::Replace($content, $pat1, "")
$pat1b = '(?ms)^(?:\s{0,3}#{1,6}\s*)?Notes on nomenclature[^\r\n]*\r?\n(?:.*?\r?\n)*?(?=^\s*---+\s*$|^\s{0,3}#{1,6}\s|\z)'
$content = [regex]::Replace($content, $pat1b, "")

# Remove 'One-sentence takeaway' section (bounded by next horizontal rule or heading), supporting any Unicode dash
$pat2 = '(?ms)^\s{0,3}#{1,6}\s*One\s*[-\u2010-\u2015]\s*sentence takeaway[^\r\n]*\r?\n(?:.*?\r?\n)*?(?=^\s*---+\s*$|^\s{0,3}#{1,6}\s|\z)'
$content = [regex]::Replace($content, $pat2, "")
$pat2b = '(?ms)^(?:\s{0,3}#{1,6}\s*)?One\s*[-\u2010-\u2015]\s*sentence takeaway[^\r\n]*\r?\n(?:.*?\r?\n)*?(?=^\s*---+\s*$|^\s{0,3}#{1,6}\s|\z)'
$content = [regex]::Replace($content, $pat2b, "")
# Fallback: match any chars between 'One' and 'sentence takeaway' to be robust to exotic dashes
$pat2c = '(?ms)^\s{0,3}#{1,6}\s*One.*?sentence takeaway[^\r\n]*\r?\n(?:.*?\r?\n)*?(?=^\s*---+\s*$|^\s{0,3}#{1,6}\s|\z)'
$content = [regex]::Replace($content, $pat2c, "")

# Explicitly remove 'Authors:' and 'Correspondence:' lines anywhere
$content = [regex]::Replace($content, '(?m)^\s*Authors:.*\r?\n', "")
$content = [regex]::Replace($content, '(?m)^\s*Correspondence:.*\r?\n', "")

# Tidy excessive blank lines
$content = [regex]::Replace($content, '(\r?\n){3,}', "`r`n`r`n")

Set-Content -LiteralPath $path -Value $content -NoNewline
Write-Host "README.md sections removed."
