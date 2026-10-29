$ErrorActionPreference='Stop'
$source=$env:AICAFE_PACKAGE.TrimEnd('\')
$target='C:\Users\andre\Downloads\aicafe-scene-vlm'
$backup=$null
$installed=@()
try {
    if([IO.Path]::GetFullPath($source) -eq [IO.Path]::GetFullPath($target)){throw 'Extract this package into a separate folder first, then run this updater.'}
    if(!(Test-Path -LiteralPath (Join-Path $target 'app/server.py') -PathType Leaf)){throw "Application not found: $target"}
    $manifest=Get-Content -LiteralPath (Join-Path $source 'samples/manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if($manifest.clips.Count -ne 64){throw 'Expected the 64-sample package.'}
    $checks=Get-Content -LiteralPath (Join-Path $source 'CHECKSUMS.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $files=@($checks.PSObject.Properties | Where-Object {$_.Name.StartsWith('samples/') -or $_.Name.StartsWith('web/')})
    foreach($item in $files){
        if($item.Name.Contains('..') -or $item.Name.Contains(':')){throw 'Invalid package path'}
        if((Get-FileHash -LiteralPath (Join-Path $source $item.Name) -Algorithm SHA256).Hash -ne $item.Value){throw "Checksum failed: $($item.Name)"}
    }
    $backup=Join-Path $target ('update-backups/samples64-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8))
    foreach($item in $files){
        $dest=Join-Path $target $item.Name
        if(Test-Path -LiteralPath $dest){
            $copy=Join-Path $backup $item.Name
            New-Item -ItemType Directory -Path (Split-Path $copy) -Force | Out-Null
            Copy-Item -LiteralPath $dest -Destination $copy
        }
    }
    foreach($item in $files){
        $dest=Join-Path $target $item.Name
        New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
        $installed+=@{Path=$dest;Relative=$item.Name}
        Copy-Item -LiteralPath (Join-Path $source $item.Name) -Destination $dest -Force
        if((Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash -ne $item.Value){throw "Installed checksum failed: $($item.Name)"}
    }
    Write-Host '64 samples installed. Existing analysis results and uploads were preserved.' -ForegroundColor Green
    Write-Host "Backup: $backup"
    Write-Host 'Refresh http://127.0.0.1:8000 with Ctrl+F5.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    foreach($item in $installed){
        try {
            $original=Join-Path $backup $item.Relative
            if(Test-Path -LiteralPath $original){Copy-Item -LiteralPath $original -Destination $item.Path -Force}
            elseif(Test-Path -LiteralPath $item.Path){Remove-Item -LiteralPath $item.Path}
        }catch{Write-Host "Restore needs attention: $($item.Path). Backup: $backup" -ForegroundColor Red}
    }
    exit 1
}
exit 0
