import React, { useEffect, useRef } from "react"

// ── Character set — alphabets + numbers only ──
const CHARS =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$&+=<>~!?".split("")

// ── Color palettes — bioluminescent green theme ──
const PALETTES = {
  emerald:  { base: 150, range: 20, sat: [60, 90], light: [35, 80] },
  forest:   { base: 140, range: 15, sat: [50, 75], light: [28, 72] },
  lime:     { base: 165, range: 18, sat: [55, 85], light: [38, 78] },
  bio:      { base: 155, range: 25, sat: [65, 95], light: [40, 85] },
}
type PaletteKey = keyof typeof PALETTES

function pickPalette(): PaletteKey {
  const r = Math.random()
  if (r < 0.38) return "emerald"
  if (r < 0.60) return "forest"
  if (r < 0.80) return "lime"
  return "bio"
}

// ── Stream class ──
interface StreamData {
  col: number
  x: number
  y: number
  speed: number
  depth: number       // 0–1, affects brightness/size/blur
  layer: number       // 0=back, 1=mid, 2=front
  chars: string[]
  maxTrail: number
  palette: PaletteKey
  hueOffset: number
  charChangeRate: number
  isFlash: boolean    // rare bright flash column
  flashIntensity: number
  spawnTime: number
}

function createStream(
  col: number,
  fontSize: number,
  H: number,
  layer: number,
  time: number,
  forceFlash = false
): StreamData {
  const isFlash = forceFlash || Math.random() < 0.03
  const depthByLayer = [
    0.15 + Math.random() * 0.25,  // back: dim
    0.35 + Math.random() * 0.35,  // mid
    0.65 + Math.random() * 0.35,  // front: bright
  ]
  const speedByLayer = [
    0.6 + Math.random() * 1.0,    // back: slow
    1.5 + Math.random() * 2.0,    // mid
    2.5 + Math.random() * 3.5,    // front: fast
  ]
  const depth = isFlash ? 1 : depthByLayer[layer]
  const trailBase = [12, 18, 25][layer]

  return {
    col,
    x: col * fontSize + fontSize / 2,
    y: -Math.random() * H * 0.4 - 20,
    speed: isFlash ? 4 + Math.random() * 3 : speedByLayer[layer],
    depth,
    layer,
    chars: [],
    maxTrail: Math.floor(trailBase * (0.6 + depth * 0.6)),
    palette: isFlash ? "bio" : pickPalette(),
    hueOffset: (Math.random() - 0.5) * 40,
    charChangeRate: 0.05 + Math.random() * 0.08,
    isFlash,
    flashIntensity: isFlash ? 0.8 + Math.random() * 0.2 : 0,
    spawnTime: time,
  }
}

// ── Component ──
interface MatrixRainProps {
  fontSize?: number
  speed?: number
  opacity?: number
  className?: string
  style?: React.CSSProperties
  clearZoneRadius?: number
  /** How dense the rain is (0.5 = sparse, 2 = very dense) */
  density?: number
}

export function MatrixRain({
  fontSize = 14,
  speed = 40,
  opacity = 1,
  className = "",
  style,
  clearZoneRadius = 180,
  density = 1.4,
}: MatrixRainProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d", { alpha: false })
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    let W = 0
    let H = 0
    let cols = 0
    let streams: StreamData[] = []
    let animId = 0
    let lastFrame = 0

    // ── Bottom reflection pool ──
    const reflectionHeight = 60

    function resize() {
      W = window.innerWidth
      H = window.innerHeight
      canvas!.width = W * dpr
      canvas!.height = H * dpr
      canvas!.style.width = W + "px"
      canvas!.style.height = H + "px"
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0)
      cols = Math.ceil(W / fontSize)
      streams = []
    }
    resize()
    window.addEventListener("resize", resize)

    const clearFade = 130
    const spawnRates = [0.014, 0.018, 0.012] // per layer

    // ── Occasional flash timer ──
    let nextFlash = 2000 + Math.random() * 3000

    function render(time: number) {
      if (time - lastFrame < speed) {
        animId = requestAnimationFrame(render)
        return
      }
      lastFrame = time

      // ── Clear with persistence ──
      ctx!.globalCompositeOperation = "source-over"
      ctx!.fillStyle = "rgb(2, 2, 8)"
      ctx!.globalAlpha = 0.18
      ctx!.fillRect(0, 0, W, H)
      ctx!.globalAlpha = 1

      // ── Spawn streams across 3 layers ──
      for (let layer = 0; layer < 3; layer++) {
        for (let c = 0; c < cols; c++) {
          if (Math.random() < spawnRates[layer] * density) {
            const sameColSameLayer = streams.filter(
              (s) => s.col === c && s.layer === layer
            )
            if (sameColSameLayer.length < 1) {
              streams.push(createStream(c, fontSize, H, layer, time))
            }
          }
        }
      }

      // ── Flash columns ──
      if (time > nextFlash) {
        const flashCol = Math.floor(Math.random() * cols)
        streams.push(createStream(flashCol, fontSize, H, 2, time, true))
        nextFlash = time + 3000 + Math.random() * 5000
      }

      const centerX = W / 2
      const centerY = H / 2

      // Sort by layer so back draws first
      streams.sort((a, b) => a.layer - b.layer)

      // ── Update & draw ──
      streams = streams.filter((stream) => {
        stream.y += stream.speed

        // Push new head char if enough distance passed? Or just slide smoothly.
        // If we slide smoothly, we don't push characters every frame.
        // Wait, if we slide smoothly and compute Y dynamically, we don't need to push!
        // Just shimmer the characters.
        stream.y += stream.speed

        // Initialize characters if empty
        if (stream.chars.length === 0) {
          const charSet = stream.layer === 0 ? CHARS : CHARS
          for (let i = 0; i < stream.maxTrail; i++) {
            stream.chars.push(charSet[Math.floor(Math.random() * charSet.length)])
          }
        }

        // Shimmer: mutate trailing chars
        for (let i = 0; i < stream.chars.length; i++) {
          if (Math.random() < stream.charChangeRate) {
            stream.chars[i] =
              CHARS[Math.floor(Math.random() * CHARS.length)]
          }
        }

        const pal = PALETTES[stream.palette]

        for (let i = 0; i < stream.chars.length; i++) {
          const char = stream.chars[i]
          const y = stream.y - i * (fontSize * 1.5) // 1.5 = expanded vertical spacing
          if (y < -fontSize * 2 || y > H + fontSize) continue

          const progress = i / stream.maxTrail

          // ── Clear zone ──
          const dx = stream.x - centerX
          const dy = y - centerY
          const dist = Math.sqrt(dx * dx + dy * dy)
          let clearFactor = 1
          if (dist < clearZoneRadius + clearFade) {
            clearFactor = Math.max(
              0,
              (dist - clearZoneRadius) / clearFade
            )
            clearFactor = clearFactor * clearFactor
          }
          if (clearFactor < 0.015) continue

          // ── Color computation ──
          const hue =
            pal.base +
            stream.hueOffset +
            Math.sin(time * 0.0003 + stream.col * 0.15) * pal.range * 0.5

          const sat = pal.sat[0] + stream.depth * (pal.sat[1] - pal.sat[0])

          let alpha: number
          let lightness: number

          if (i === 0) {
            // ── HEAD: white-hot with colored glow ──
            if (stream.isFlash) {
              alpha = 1 * clearFactor * opacity
              lightness = 95
              ctx!.shadowColor = `hsla(${hue}, 90%, 75%, ${0.9 * clearFactor})`
              ctx!.shadowBlur = 25
            } else {
              alpha =
                (0.85 + Math.random() * 0.15) *
                stream.depth *
                clearFactor *
                opacity
              lightness = 78 + Math.random() * 17

              // Random bright flash on any head
              if (Math.random() < 0.06) {
                lightness = 95
                alpha = Math.min(1, alpha * 1.4)
              }

              if (stream.depth > 0.5) {
                ctx!.shadowColor = `hsla(${hue}, 75%, 70%, ${0.5 * stream.depth * clearFactor * opacity})`
                ctx!.shadowBlur = 12 * stream.depth
              }
            }
          } else if (i <= 2) {
            // ── Near-head: still bright ──
            const nearFade = 1 - i * 0.2
            alpha = nearFade * 0.7 * stream.depth * clearFactor * opacity
            lightness = 60 + Math.random() * 15

            if (stream.isFlash) {
              alpha *= 1.5
              lightness += 15
              ctx!.shadowColor = `hsla(${hue}, 70%, 60%, ${0.3 * clearFactor})`
              ctx!.shadowBlur = 8
            }
          } else {
            // ── Trail: gradient fade ──
            const fade = 1 - progress
            alpha =
              fade * fade * fade * 0.5 * stream.depth * clearFactor * opacity
            lightness = pal.light[0] + fade * (pal.light[1] - pal.light[0]) * 0.4

            if (stream.isFlash) {
              alpha *= 2
              ctx!.shadowColor = `hsla(${hue}, 50%, 50%, ${0.15 * clearFactor * fade})`
              ctx!.shadowBlur = 4
            }
          }

          if (alpha < 0.008) continue

          ctx!.globalAlpha = Math.min(1, alpha)

          // Head chars are white, trail chars are colored
          if (i === 0 && stream.depth > 0.6) {
            ctx!.fillStyle = `hsl(${hue}, ${sat * 0.3}%, ${lightness}%)`
          } else {
            ctx!.fillStyle = `hsl(${hue}, ${sat}%, ${lightness}%)`
          }

          const fs = fontSize * (0.75 + stream.depth * 0.3)
          ctx!.font = `${Math.round(fs)}px 'JetBrains Mono', monospace`
          ctx!.textAlign = "center"
          ctx!.fillText(char, stream.x, y)

          // Reset shadow after each char
          ctx!.shadowColor = "transparent"
          ctx!.shadowBlur = 0
        }

        // ── Bottom reflection (mirrored, faded) ──
        if (stream.depth > 0.4) {
          const lastChar = stream.chars[0]
          const lastCharY = stream.y
          if (lastChar && lastCharY > H - reflectionHeight * 2) {
            const reflY = H + (H - lastCharY) * 0.3
            if (reflY > H && reflY < H + reflectionHeight) {
              const reflAlpha =
                0.08 *
                stream.depth *
                (1 - (reflY - H) / reflectionHeight) *
                opacity
              if (reflAlpha > 0.01) {
                const pal2 = PALETTES[stream.palette]
                const hue2 = pal2.base + stream.hueOffset
                ctx!.globalAlpha = reflAlpha
                ctx!.fillStyle = `hsl(${hue2}, ${pal2.sat[0]}%, ${pal2.light[0] + 10}%)`
                ctx!.font = `${fontSize * 0.7}px 'JetBrains Mono', monospace`
                ctx!.textAlign = "center"
                ctx!.fillText(lastChar, stream.x, reflY)
              }
            }
          }
        }

        ctx!.globalAlpha = 1
        ctx!.shadowColor = "transparent"
        ctx!.shadowBlur = 0

        // Keep alive until entire trail below viewport
        const lastY = stream.y - stream.chars.length * (fontSize * 1.5)
        return lastY < H + reflectionHeight + 20
      })

      // ── Bottom glow line (pooling light) ──
      const grad = ctx!.createLinearGradient(0, H - 30, 0, H)
      grad.addColorStop(0, "rgba(120, 110, 255, 0)")
      grad.addColorStop(0.6, `rgba(120, 110, 255, ${0.02 * opacity})`)
      grad.addColorStop(1, `rgba(120, 110, 255, ${0.06 * opacity})`)
      ctx!.fillStyle = grad
      ctx!.fillRect(0, H - 30, W, 30)

      // ── Occasional horizontal scan line ──
      const scanY = (time * 0.03) % (H + 200) - 100
      if (scanY > 0 && scanY < H) {
        ctx!.globalAlpha = 0.03 * opacity
        const scanGrad = ctx!.createLinearGradient(0, scanY - 2, 0, scanY + 2)
        scanGrad.addColorStop(0, "transparent")
        scanGrad.addColorStop(0.5, "rgba(161, 157, 252, 1)")
        scanGrad.addColorStop(1, "transparent")
        ctx!.fillStyle = scanGrad
        ctx!.fillRect(0, scanY - 2, W, 4)
        ctx!.globalAlpha = 1
      }

      animId = requestAnimationFrame(render)
    }

    animId = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener("resize", resize)
    }
  }, [fontSize, speed, opacity, clearZoneRadius, density])

  return (
    <canvas
      ref={canvasRef}
      className={`fixed inset-0 pointer-events-none ${className}`}
      style={{ opacity, ...style }}
    />
  )
}
