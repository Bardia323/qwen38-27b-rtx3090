<#
qwen.ps1 — expose Qwen3.8-27B (vLLM on WSL2, F:\qwen38-27b-rtx3090) to your LAN.

Usage:
  double-click qwen.cmd        asks UAC once, then does everything
  qwen                         same, from any terminal (alias lives in %USERPROFILE%\bin)
  qwen -DryRun                 show what would happen, change nothing, no UAC

What it does (idempotent — safe to re-run any time):
  1. elevates to admin (portproxy + firewall need it; one UAC)
  2. portproxy:  0.0.0.0:18020 -> 127.0.0.1:18020   (WSL2's automatic localhost
     forwarding; target 127.0.0.1 so it keeps working even when the WSL IP changes)
  3. firewall:   allow inbound TCP 18020 from your LAN subnet only (not the world)
  4. prints the URL + API key your Mac needs
  5. if the server is NOT running: opens a WSL console at the repo, runs
     SPEC=dflash2 CTX=long PREFIX_CACHE=1 single-user/start_qwen.sh, waits until
     /health says 200, then sends one throwaway request to warm the Triton kernels
     If it IS running: just opens a WSL console at the repo. Never touches a
     running server.

There is no autostart. A 'Qwen vLLM autostart' scheduled task used to be registered
from here and never once ran (see the note at step 2b); it has been unregistered.
The portproxy and firewall rule persist across reboots by themselves, so the LAN
side stays configured -- only the server needs starting, by running this script.

Elevated runs write a log to qwen-run.log next to this script and keep their
window open until you press Enter.
#>
param(
  [switch]$DryRun,
  # -Model stock|uncensored skips the prompt. Anything else asks.
  [ValidateSet('stock', 'uncensored', 'ask')]
  [string]$Model = 'ask'
)

$ErrorActionPreference = 'Stop'

$REPO   = 'F:\qwen38-27b-rtx3090'
$WSLDIR = '/mnt/f/qwen38-27b-rtx3090'
$DISTRO = 'Ubuntu'
$PORT   = 18020
$LOG    = Join-Path $PSScriptRoot 'qwen-run.log'

# How the server is launched inside WSL.
#   SPEC=dflash2   the DFlash2 block drafter (7 drafts in one non-autoregressive pass)
#                  instead of Qwen's chained MTP head. Needs the W4A16 drafter in
#                  ~/models/Qwen3.8-27B-DFlash2-W4A16 and the patched vLLM; the start
#                  script finds both on its own and errors clearly if the drafter is gone.
#   CTX=long       int8 per-token-head KV, 131k context, 138,696 tokens of KV pool.
#   PREFIX_CACHE=1 reuse the KV of a shared prompt prefix across requests. CTX=long pays
#                  a heavy prefill (a 112k document takes ~251s), and this is what makes
#                  that a once-per-document cost instead of once per turn -- turn 2 of a
#                  chat over a 24k document goes ~23s -> ~1s. Costs one recurrent-state
#                  page per request, so the pool goes 138,696 -> 136,429 tokens.
# --- the two configurations, and the picker -----------------------------------------
#
# Both run SPEC=dflash2 (the DFlash2 block drafter) on int8 KV with prefix caching, which
# measured fastest of everything tried on this card.
#
# Two things are deliberately NOT offered, both measured on this box:
#   CTX=huge (KVarN 4/2-bit KV) reaches 164k but costs 3-5x decode -- the reproduce task
#     ran 11.7 tok/s against 55.3 on int8 KV. Use it by hand if you need the window.
#   DFLASH_TOKENS=15, the repo's 381 tok/s "reproduction mode", is a straight loss on the
#     int8 path for BOTH models: stock went 52.6 -> 32.5 reproduce and 105 -> 32 edit
#     while halving the window. Those published figures are CTX=fast (bf16 KV) at 25k.
#
# The uncensored build needs LM_ONLY=0 (exported text-only, no vision tower to skip) and a
# smaller window: its publisher left lm_head and embed_tokens in bf16, and even after
# int8-ing both it only fits ~98k.
$CONFIGS = [ordered]@{
  stock      = @{
    Label   = 'Stock Qwen3.8-27B   131k ctx  247 tok/s short, 105 edit, 53 reproduce'
    Context = 131072
    # Script defaults, including its pinned KV_MEM=5583457484 -- that value is tuned to
    # this checkpoint's footprint and yields 136,429 KV tokens. Do NOT pass KV_MEM= here:
    # auto-sizing from GPU_UTIL gives less and caps the window at ~102k.
    Launch  = 'SPEC=dflash2 CTX=long PREFIX_CACHE=1 bash single-user/start_qwen.sh'
  }
  uncensored = @{
    Label   = 'Heretic abliterated  98k ctx  212 tok/s short, 112 edit, 55 reproduce'
    Context = 98304
    # KV_MEM= (auto) on purpose: this build was converted from bf16 lm_head/embeddings to
    # int8, so its footprint differs from the stock checkpoint and the pinned value above
    # does not transfer.
    #
    # DRAFT is the STOCK drafter, not the abliterated one, and that is not a mistake.
    # Speculative decoding is exact -- the target verifies every token -- so a drafter
    # trained on the censored model cannot reintroduce refusals, and the abliteration is
    # measurably intact (identical output, refuses nothing). What it buys is speed: the
    # stock drafter is GPTQ-calibrated, the heretic-ara one only RTN, and on this target
    # that is 13.8 -> 55.1 tok/s reproduce, 73 -> 112 edit, 34 -> 64 summary, with
    # acceptance 53.9% -> 57.0%. Confirmed in both A/B orderings with a long-prompt warmup.
    Launch  = 'MODEL=$HOME/models/Qwen3.8-27B-Heretic-W4A16 ' +
              'DRAFT=$HOME/models/Qwen3.8-27B-DFlash2-W4A16 ' +
              'SPEC=dflash2 CTX=long LM_ONLY=0 KV_MEM= DFLASH_MAX_LEN=98304 ' +
              'PREFIX_CACHE=1 VLLM_V2_CUDAGRAPH_MEM_MIB=900 bash single-user/start_qwen.sh'
  }
}

function Select-QwenConfig([string]$choice) {
  if ($CONFIGS.Contains($choice)) { return $choice }
  Write-Host ''
  Write-Host 'Which model?'
  Write-Host ('  [1] ' + $CONFIGS.stock.Label)
  Write-Host ('  [2] ' + $CONFIGS.uncensored.Label)
  Write-Host ''
  while ($true) {
    $a = Read-Host 'Choose 1 or 2 (Enter = 1)'
    if ($a -eq '' -or $a -eq '1') { return 'stock' }
    if ($a -eq '2') { return 'uncensored' }
    Write-Host '  please enter 1 or 2'
  }
}

$PICK   = Select-QwenConfig $Model
$LAUNCH = $CONFIGS[$PICK].Launch
$CTXLEN = $CONFIGS[$PICK].Context

function ToU32([string]$a) {
  $o = @($a.Split('.') | ForEach-Object { [uint64]$_ })
  [uint32]($o[0] -shl 24 -bor $o[1] -shl 16 -bor $o[2] -shl 8 -bor $o[3])
}
function FromU32([uint32]$v) {
  $x = [uint64]$v
  "{0}.{1}.{2}.{3}" -f [int]($x -shr 24), [int]($x -shr 16 -band 0xFF), [int]($x -shr 8 -band 0xFF), [int]($x -band 0xFF)
}
function Get-LanInterface {
  Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
      $_.InterfaceAlias -notlike 'vEthernet*' -and
      $_.IPAddress -notlike '127.*' -and
      $_.IPAddress -notlike '169.254.*' -and
      $_.PrefixLength -lt 32
    } |
    Select-Object -First 1
}
function Test-ServerUp([int]$port) {
  try {
    $code = & curl.exe -s -o NUL -w '%{http_code}' --max-time 5 "http://127.0.0.1:$port/health"
    return "$code" -eq '200'
  } catch { return $false }
}
function Get-ApiKey {
  $p = Join-Path $REPO 'api_key.txt'
  if (Test-Path $p) { (Get-Content $p -Raw).Trim() } else { '' }
}

# --- detect the real LAN interface (skip WSL vEthernet, loopback, link-local) ---
$nic    = Get-LanInterface
$hostIP = $null
$subnet = $null
if ($nic) {
  $hostIP = $nic.IPAddress
  $ones   = [uint32]([math]::Pow(2, $nic.PrefixLength) - 1)
  $maskU  = [uint32]( [uint64]$ones -shl [int](32 - $nic.PrefixLength) )
  $subnet = (FromU32 ([uint32]([uint64](ToU32 $nic.IPAddress) -band [uint64]$maskU))) + '/' + $nic.PrefixLength
} else {
  Write-Warning 'Could not detect a LAN IPv4 interface; firewall rule will be scoped to Any (restrict manually).'
  $subnet = 'Any'
}

# Always transcript. This used to be gated on QWEN_ELEV_LOG, which nothing ever set, so
# qwen-run.log was never created and a failed run left no trace anywhere.
$elevatedLog = $true
try { Start-Transcript -Path $LOG -Append | Out-Null } catch { $elevatedLog = $false }

Write-Host ''
Write-Host '=== qwen: Qwen3.8-27B LAN bridge ==='
if ($nic) {
  Write-Host ('  this PC : {0}  (LAN {1}, interface "{2}")' -f $hostIP, $subnet, $nic.InterfaceAlias)
} else {
  Write-Host '  this PC : (no LAN interface found)'
}
Write-Host ('  port    : {0}' -f $PORT)

$up = Test-ServerUp $PORT
Write-Host ('  server  : {0}' -f $(if ($up) { 'already running (will not be restarted)' } else { 'not running (will be started)' }))
Write-Host ''

if ($DryRun) {
  Write-Host '[DryRun] no changes made. It would do:'
  $dryIP = (& wsl.exe -d $DISTRO -- hostname -I | Out-String).Trim().Split(' ')[0]
  Write-Host ('  1. netsh portproxy: 0.0.0.0:{0} -> {1}:{0}  (WSL guest IP, needs admin)' -f $PORT, $(if ($dryIP) { $dryIP } else { '<WSL IP>' }))
  Write-Host ('  2. firewall rule "Qwen vLLM LAN TCP {0}": allow inbound TCP {0} from {1}' -f $PORT, $subnet)
  if ($up) {
    Write-Host "  3. server is up -> open a WSL console at $WSLDIR"
  } else {
    Write-Host "  3. server down -> open a WSL console at $WSLDIR and run $LAUNCH"
  }
  Write-Host ('  4. Mac base_url would be: http://{0}:{1}/v1  (key: {2})' -f $hostIP, $PORT,
    $(if ((Get-ApiKey)) { 'in api_key.txt' } else { 'MISSING api_key.txt' }))
  if ($elevatedLog) { Stop-Transcript }
  return
}

# --- self-elevate (one UAC) ---
$ident = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = (New-Object Security.Principal.WindowsPrincipal($ident)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
  $self = $MyInvocation.MyCommand.Path
  Write-Host 'Need admin for the portproxy + firewall. Click Yes in the UAC window.'
  $env:QWEN_ELEV_LOG = '1'
  Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$self`"")
  Read-Host 'Elevation requested. Press Enter to close this window.'
  return
}

try {
  # --- 1. portproxy (needs IPNAT service + admin) ---
  $svc = Get-Service -Name IPNAT -ErrorAction SilentlyContinue
  if ($svc -and $svc.Status -ne 'Running') {
    Write-Host '  starting IPNAT service...'
    Start-Service -Name IPNAT
  }
  # The proxy MUST target the WSL VM's real IP, not 127.0.0.1.
  #
  # This used to say connectaddress=127.0.0.1, on the theory that WSL2's automatic
  # localhost forwarding would carry it into the distro and the rule would survive the
  # WSL IP changing. It does not work: binding 0.0.0.0:$PORT on the Windows side SHADOWS
  # localhost forwarding, so the proxy then forwards to itself. Every connection loops --
  # localhost, 127.0.0.1 and the LAN IP all failed from Windows while the server was
  # healthy inside WSL, which is what broke the Hermes TUI.
  #
  # NAT mode reassigns the WSL IP on every WSL restart, so it is looked up per run and the
  # rule is rebuilt. (Windows 11 22H2+ could use networkingMode=mirrored and skip all of
  # this; this box is Windows 10 19045, where that is unavailable.)
  $wslIP = (& wsl.exe -d $DISTRO -- hostname -I | Out-String).Trim().Split(' ')[0]
  if (-not $wslIP -or $wslIP -notmatch '^\d+\.\d+\.\d+\.\d+$') {
    throw "Could not determine the WSL IP for distro '$DISTRO' (got '$wslIP'). Is the distro running?"
  }
  netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$PORT *> $null
  netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$PORT connectaddress=$wslIP connectport=$PORT
  Write-Host ('  portproxy: 0.0.0.0:{0} -> {1}:{0}  (WSL2 guest, NAT mode)' -f $PORT, $wslIP)

  # --- 2. firewall, scoped to your LAN subnet ---
  $rule = "Qwen vLLM LAN TCP $PORT"
  $ruleOk = $false
  Remove-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
  try {
    New-NetFirewallRule -DisplayName $rule `
      -Description "Qwen3.8-27B vLLM on WSL2 ($REPO): LAN -> port $PORT" `
      -Direction Inbound -Action Allow -Protocol TCP -LocalPort $PORT -RemoteAddress $subnet | Out-Null
    $ruleOk = $true
  } catch {
    Write-Warning ("  New-NetFirewallRule failed: " + $_.Exception.Message + " -- falling back to netsh")
  }
  if (-not $ruleOk) {
    $rname = "Qwen vLLM LAN TCP $PORT"
    netsh advfirewall firewall delete rule name="$rname" *> $null
    netsh advfirewall firewall add rule name="$rname" dir=in action=allow protocol=TCP localport=$PORT remoteip=$subnet
    if ($LASTEXITCODE -eq 0) {
      $ruleOk = $true
      Write-Host ("  firewall (netsh): allow inbound TCP $PORT from $subnet")
    } else {
      Write-Warning "  netsh advfirewall add also failed (exit $LASTEXITCODE)."
    }
  } else {
    Write-Host ('  firewall : allow inbound TCP {0} from {1}' -f $PORT, $subnet)
  }

  # --- verify ---
  $verified = $false
  $check = Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
  if ($check) {
    $af = $check | Get-NetFirewallAddressFilter
    Write-Host ('  verified : rule active, remote address = {0}' -f ($af.RemoteAddress -join ', '))
    $verified = $true
  }
  if (-not $verified) {
    Write-Warning '  Could not verify the firewall rule — the Mac may be blocked by Windows Firewall. Check qwen-run.log.'
  }

  # --- 2b. no autostart ---
  # There used to be a 'Qwen vLLM autostart' scheduled task registered here, running
  # qwen.cmd 90s after every logon. It never worked: the task ran at RunLevel=Limited,
  # so this script took its self-elevation branch, raised a UAC prompt nobody was there
  # to click, and then blocked on Read-Host. LastTaskResult stayed 0x41303 ("never run")
  # and qwen-run.log was never written. Removed rather than repaired -- the server is
  # started by hand now. The portproxy and firewall rule above are persistent across
  # reboots on their own, so nothing needs to re-run at logon to keep the LAN reachable.

  # --- 3. start the server if it isn't running ---
  if ($up) {
    # A server is already up. It may not be the one just picked, and both serve the same
    # model id ("qwen3.8-27b"), so identify it by the checkpoint path vLLM reports.
    $running = ''
    try {
      $m = & curl.exe -s --max-time 10 -H "Authorization: Bearer $(Get-ApiKey)" `
             "http://127.0.0.1:$PORT/v1/models" | ConvertFrom-Json
      $running = [string]$m.data[0].root
      Write-Host ('  running   : {0} (max_model_len {1})' -f (Split-Path $running -Leaf), $m.data[0].max_model_len)
    } catch { Write-Host '  running   : (could not read /v1/models)' }

    $wantHeretic = ($PICK -eq 'uncensored')
    $isHeretic   = $running -match 'Heretic'
    if ($running -and ($wantHeretic -ne $isHeretic)) {
      Write-Warning ("  the running server is NOT the one you picked ($PICK).")
      $r = Read-Host '  restart it with the chosen config? [y/N]'
      if ($r -match '^(y|yes)$') {
        Write-Host '  stopping the running server...'
        & wsl.exe -d $DISTRO -- bash -c "pkill -f 'vllm serve'; sleep 5; pgrep -f 'vllm serve' >/dev/null && pkill -9 -f 'vllm serve'; true" *> $null
        Start-Sleep -Seconds 5
        $up = $false          # fall through to the start path below
      }
    }
  }
  if ($up) {
    Write-Host '  server already up - opening a WSL console at the repo.'
    Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $DISTRO, '--', 'bash', '-l', '-c', "cd $WSLDIR")
  } else {
    # Never start an engine while a previous one still holds the GPU. A vLLM engine that is
    # mid-teardown (or wedged) keeps ~23 GB reserved; the next boot then fights it for memory
    # during CUDA graph capture and hangs at 100% CPU, deaf to SIGTERM. Measured on this box:
    # capture took 3s on a clean card, 542s against a dying engine, and the third attempt
    # wedged outright. Draining first is what makes a restart reliable.
    Write-Host '  checking the GPU is free before starting...'
    # The drain script lives in windows/gpu-drain.sh -- see the comment there for why it is
    # not inlined here. Never let the check itself stop the launch: this script runs with
    # $ErrorActionPreference = 'Stop', so anything it surfaces would otherwise abort
    # straight to the finally block and the server would silently never start.
    try {
      $free = (& wsl.exe -d $DISTRO -- bash "$WSLDIR/windows/gpu-drain.sh" | Out-String).Trim()
    } catch {
      $free = "(drain check failed: $($_.Exception.Message))"
    }
    Write-Host ("  GPU in use before start: {0}" -f $free)

    Write-Host "  starting server: opening a WSL console at the repo, running $LAUNCH"
    # The launch line goes through a file, not `bash -l -c "<command>"`. Start-Process on
    # PowerShell 5.1 does not quote -ArgumentList elements that contain spaces, so bash
    # received `-c cd` and took the rest as positional arguments: the console opened, ran
    # nothing, and closed, while this script sat waiting on /health for 15 minutes.
    $launchSh = Join-Path $PSScriptRoot '.qwen-launch.sh'
    $script = "#!/bin/bash`nset -x`ncd $WSLDIR || exit 1`n$LAUNCH`n"
    [IO.File]::WriteAllText($launchSh, ($script -replace "`r`n", "`n"), (New-Object Text.UTF8Encoding $false))
    Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d', $DISTRO, '--', 'bash', '-l', "$WSLDIR/windows/.qwen-launch.sh")
    Write-Host '  waiting for /health (~90s warm; several minutes if the torch.compile cache is cold, e.g. after changing the config or repatching vLLM)...'
    $waited = 0
    while ($waited -lt 900 -and -not (Test-ServerUp $PORT)) {
      Start-Sleep -Seconds 5
      $waited += 5
      Write-Host ('    ... {0:0}m {1:0}s' -f [math]::Floor($waited / 60), ($waited % 60))
    }
    if (Test-ServerUp $PORT) {
      Write-Host '  server is up.'
      # /health goes green before the Triton kernels are warm: the first real request
      # JIT-compiles kernel_unified_attention, _prepare_dflash_inputs_kernel,
      # _compute_local_logits_stats_kernel, _rejection_kernel and _resample_kernel, which
      # makes it several times slower than steady state. Spend one throwaway request here
      # so the first request from the Mac is not the one that pays for it.
      Write-Host '  warming the speculative-decoding kernels (one throwaway request)...'
      $wk = Get-ApiKey
      $body = '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":24,"temperature":0}'
      & curl.exe -s -o NUL --max-time 180 "http://127.0.0.1:$PORT/v1/chat/completions" `
        -H "Authorization: Bearer $wk" -H 'Content-Type: application/json' -d $body
      if ($LASTEXITCODE -eq 0) { Write-Host '  warm.' } else { Write-Warning '  warmup request failed; the first real request will be slow.' }
    }
    else { Write-Warning '  timed out after 15 min - check the WSL console for errors.' }
  }

  # --- 4. summary for the Mac ---
  $key = Get-ApiKey
  Write-Host ''
  Write-Host '=== reachable from your Mac (same LAN) ==='
  Write-Host ('  base_url  : http://{0}:{1}/v1' -f $hostIP, $PORT)
  Write-Host ('  health    : http://{0}:{1}/health' -f $hostIP, $PORT)
  Write-Host '  model     : qwen3.8-27b'
  Write-Host ('  config    : {0} — {1}' -f $PICK, $CONFIGS[$PICK].Label)
  Write-Host ('  context   : {0} tokens' -f $CTXLEN)
  Write-Host ''
  Write-Host ('  Hermes config.yaml must say  context_length: {0}' -f $CTXLEN)
  Write-Host '    (C:\Users\bardi\AppData\Local\hermes\config.yaml)'
  Write-Host '    The two configs serve different windows; a stale value means Hermes'
  Write-Host '    assembles prompts the server rejects.'
  Write-Host ('  api_key   : {0}' -f $(if ($key) { $key } else { '(missing F:\qwen38-27b-rtx3090\api_key.txt)' }))
  Write-Host ''
  Write-Host '  test from the Mac:'
  Write-Host ('    curl http://{0}:{1}/v1/chat/completions -H "Authorization: Bearer {2}" -H "Content-Type: application/json" -d "{""model"":""qwen3.8-27b"",""messages"":[{""role"":""user"",""content"":""hi""}]}"' -f $hostIP, $PORT, $key)
  Write-Host ''
}
finally {
  if ($elevatedLog) { Stop-Transcript }
  Read-Host "Done (full log: $LOG). Press Enter to close."
}
