param(
  [string]$Path = "README.md"
)

# Read file as raw text
$txt = Get-Content -Raw -LiteralPath $Path

# Convert all inline LaTeX math delimiters \( ... \) to $ ... $
# Simple literal replace: \( ... \) -> $ ... $
$txt = $txt.Replace('\(', '$').Replace('\)', '$')

# Write back
Set-Content -LiteralPath $Path -Value $txt -Encoding UTF8
