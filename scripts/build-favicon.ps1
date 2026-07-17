param(
  [string]$InputPath = "public/tonggu-logo.png",
  [string]$OutputPath = "public/tonggu-favicon.png"
)

Add-Type -AssemblyName System.Drawing

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$source = [System.Drawing.Bitmap]::FromFile($resolvedInput)
$size = 256
$canvas = New-Object System.Drawing.Bitmap($size, $size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$graphics = [System.Drawing.Graphics]::FromImage($canvas)

try {
  $graphics.Clear([System.Drawing.Color]::Transparent)
  $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

  $background = $source.GetPixel([int]($source.Width / 2), 8)
  $radius = 42
  $diameter = $radius * 2
  $path = New-Object System.Drawing.Drawing2D.GraphicsPath
  $path.AddArc(0, 0, $diameter, $diameter, 180, 90)
  $path.AddArc($size - $diameter, 0, $diameter, $diameter, 270, 90)
  $path.AddArc($size - $diameter, $size - $diameter, $diameter, $diameter, 0, 90)
  $path.AddArc(0, $size - $diameter, $diameter, $diameter, 90, 90)
  $path.CloseFigure()

  $brush = New-Object System.Drawing.SolidBrush($background)
  $graphics.FillPath($brush, $path)

  $targetWidth = 224
  $targetHeight = [int][Math]::Round($source.Height * ($targetWidth / $source.Width))
  $targetX = [int](($size - $targetWidth) / 2)
  $targetY = [int](($size - $targetHeight) / 2)
  $graphics.DrawImage($source, $targetX, $targetY, $targetWidth, $targetHeight)

  $outputDirectory = Split-Path -Parent $OutputPath
  if ($outputDirectory) { New-Item -ItemType Directory -Force $outputDirectory | Out-Null }
  $canvas.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
  if ($brush) { $brush.Dispose() }
  if ($path) { $path.Dispose() }
  $graphics.Dispose()
  $canvas.Dispose()
  $source.Dispose()
}
