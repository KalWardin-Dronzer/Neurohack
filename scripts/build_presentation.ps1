param(
    [string]$OutputPath = "",
    [switch]$ExportPdf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputPath) {
    $OutputPath = Join-Path $projectRoot "dist\Long_Form_Memory_System_Presentation.pptx"
}

$outputDir = Split-Path -Parent $OutputPath
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Pull latest benchmark metrics for the results slide.
$metrics = @{
    recall_rate = "N/A"
    retrieval_relevance = "N/A"
    latency_p50_ms = "N/A"
    latency_p95_ms = "N/A"
    contradiction_rate = "N/A"
    total_probes = "N/A"
}

try {
    $pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $pythonPath) {
        $raw = & $pythonPath -m scripts.run_evaluation
        $parsed = $raw | ConvertFrom-Json
        $metrics.recall_rate = $parsed.recall_rate
        $metrics.retrieval_relevance = $parsed.retrieval_relevance
        $metrics.latency_p50_ms = $parsed.latency_p50_ms
        $metrics.latency_p95_ms = $parsed.latency_p95_ms
        $metrics.contradiction_rate = $parsed.contradiction_rate
        $metrics.total_probes = $parsed.total_probes
    }
} catch {
    Write-Warning "Could not load evaluation metrics automatically. Using N/A placeholders."
}

function Add-TextSlide {
    param(
        [Parameter(Mandatory = $true)][object]$Presentation,
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string[]]$Bullets
    )

    # 2 = ppLayoutText
    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 2)
    $slide.Shapes.Title.TextFrame.TextRange.Text = $Title

    $body = $slide.Shapes.Placeholders.Item(2).TextFrame.TextRange
    $body.Text = ($Bullets -join "`r`n")
    $body.Font.Size = 22
    $body.ParagraphFormat.Bullet.Visible = -1
    return $slide
}

function Add-TitleSlide {
    param(
        [Parameter(Mandatory = $true)][object]$Presentation,
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Subtitle
    )

    # 1 = ppLayoutTitle
    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 1)
    $slide.Shapes.Title.TextFrame.TextRange.Text = $Title
    $slide.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $Subtitle
    return $slide
}

$ppt = $null
$presentation = $null

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = 1
    $presentation = $ppt.Presentations.Add()

    $today = Get-Date -Format "yyyy-MM-dd"

    Add-TitleSlide -Presentation $presentation -Title "Long-Form Memory System for 1,000+ Turns" -Subtitle "Hackathon Submission | Generated on $today" | Out-Null

    Add-TextSlide -Presentation $presentation -Title "1) Problem Statement" -Bullets @(
        "Goal: Retain and recall user information across 1,000+ turns in real time.",
        "Constraints: no full conversation replay, no unbounded prompt growth, fully automated pipeline.",
        "Success condition: information from early turns must influence late-turn behavior accurately."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "2) System Architecture (Components + Dataflow)" -Bullets @(
        "Fast path (sync): store turn -> retrieve relevant memories -> generate response -> return.",
        "Background path (async): extract memory candidates -> normalize/update -> persist/index.",
        "Core components: API layer, MemoryEngine, Extractor, Store, HybridRetriever, Responder."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "3) Memory Extraction Format" -Bullets @(
        "Structured memory record fields: type, key, value, source_turn, confidence, status.",
        "Influence output at inference: memory_id, content, origin_turn, last_used_turn.",
        "Example: User prefers calls after 11 AM (origin_turn=1)."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "4) Persistence Strategy" -Bullets @(
        "Durable SQLite tables: turns, memories, memory_embeddings.",
        "Lexical index: SQLite FTS5 virtual table for fast keyword retrieval.",
        "Memory lifecycle: active/deprecated with supersedes_memory_id for conflict history."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "5) Retrieval & Injection Policy" -Bullets @(
        "Hybrid retrieval score = semantic similarity + lexical relevance + recency + confidence.",
        "Top-k relevant memories selected; one memory per key to avoid contradictions in context.",
        "Memory is injected implicitly into response generation; exposed only when debugging/evaluating."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "6) Evaluation Methodology" -Bullets @(
        "Synthetic long-conversation benchmark (up to 1,000 turns) with delayed recall probes.",
        "Conflict scenario verifies supersession behavior (old preference replaced by new preference).",
        "Measured metrics: recall_rate, retrieval_relevance, contradiction_rate, latency p50/p95."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "7) Key Quantitative Results" -Bullets @(
        "recall_rate: $($metrics.recall_rate)",
        "retrieval_relevance: $($metrics.retrieval_relevance)",
        "contradiction_rate: $($metrics.contradiction_rate)",
        "total_probes: $($metrics.total_probes)"
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "8) Latency Results (Real-Time Requirement)" -Bullets @(
        "latency_p50_ms: $($metrics.latency_p50_ms)",
        "latency_p95_ms: $($metrics.latency_p95_ms)",
        "Design choice enabling low latency: asynchronous extraction outside the user-critical path.",
        "Inference retrieves only top relevant memories, avoiding prompt overload."
    ) | Out-Null

    Add-TextSlide -Presentation $presentation -Title "9) Limitations, Future Work, and Demo Links" -Bullets @(
        "Limitations: rule-heavy extractor; semantic retrieval can be scaled further with FAISS/pgvector.",
        "Future work: richer LLM extractor, stronger evaluation suite, production-grade vector index.",
        "Demo command: python -m scripts.run_demo",
        "Metrics command: python -m scripts.run_evaluation",
        "API sample logs: API_SAMPLE_RESPONSES.md | Submission summary: SUBMISSION.md"
    ) | Out-Null

    $presentation.SaveAs($OutputPath)
    if ($ExportPdf) {
        $pdfPath = [System.IO.Path]::ChangeExtension($OutputPath, ".pdf")
        # 32 = ppSaveAsPDF
        $presentation.SaveAs($pdfPath, 32)
    }
} finally {
    if ($presentation -ne $null) {
        $presentation.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($ppt -ne $null) {
        $ppt.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "Created presentation: $OutputPath"
if ($ExportPdf) {
    Write-Output "Created PDF: $([System.IO.Path]::ChangeExtension($OutputPath, '.pdf'))"
}
