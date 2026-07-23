param(
    [Parameter(Mandatory=$true)][string]$Video,
    [string]$Output = "outputs/results.json",
    [string]$AnnotatedVideo = "outputs/annotated.mp4"
)

$ErrorActionPreference = "Stop"
python -m traffic_plate_study run `
    --video $Video `
    --output $Output `
    --config "config/default.yaml" `
    --annotated-video $AnnotatedVideo
