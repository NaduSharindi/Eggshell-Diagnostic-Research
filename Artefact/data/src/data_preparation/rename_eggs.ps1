# Renames all .jpg files in -Folder to E001.jpg, E002.jpg ... ordered by EXIF "Date taken".
# Writes mapping.csv (new name, date taken, old name). Run on a COPY, never on the raw backup.
param([Parameter(Mandatory=$true)][string]$Folder)
Add-Type -AssemblyName System.Drawing

function Get-DateTaken($path) {
    $fs = $null
    try {
        $fs  = [IO.File]::OpenRead($path)
        $img = [Drawing.Image]::FromStream($fs, $false, $false)
        $raw = [Text.Encoding]::ASCII.GetString($img.GetPropertyItem(36867).Value).Trim([char]0)
        $img.Dispose()
        return [datetime]::ParseExact($raw, 'yyyy:MM:dd HH:mm:ss', $null)
    } catch {
        return (Get-Item $path).LastWriteTime   # fallback if no EXIF date
    } finally { if ($fs) { $fs.Close() } }
}

$files = Get-ChildItem -Path $Folder -File | Where-Object { $_.Extension -match '^\.jpe?g$' }
Write-Host "Reading dates from $($files.Count) images..."
$list = $files | ForEach-Object {
    [pscustomobject]@{ File = $_; Taken = Get-DateTaken $_.FullName }
} | Sort-Object Taken, @{ Expression = { $_.File.Name } }

$pad = [Math]::Max(3, "$($list.Count)".Length)
$map = @(); $i = 0
# Pass 1: temporary names (avoids clashes with files already called E001 etc.)
foreach ($item in $list) {
    $i++
    $new = "E{0:D$pad}{1}" -f $i, $item.File.Extension.ToLower()
    $tmp = "__tmp_$i$($item.File.Extension)"
    $map += [pscustomobject]@{ new_file = $new; date_taken = $item.Taken.ToString('yyyy-MM-dd HH:mm:ss'); old_file = $item.File.Name; tmp = $tmp }
    Rename-Item -LiteralPath $item.File.FullName -NewName $tmp
}
# Pass 2: final names
foreach ($m in $map) { Rename-Item -LiteralPath (Join-Path $Folder $m.tmp) -NewName $m.new_file }

$map | Select-Object new_file, date_taken, old_file |
    Export-Csv -Path (Join-Path $Folder 'mapping.csv') -NoTypeInformation
Write-Host "Done: renamed $($map.Count) files. See mapping.csv"
