import React, { useEffect, useState, useRef, useCallback } from "react"
import { MatrixRain } from "./MatrixRain"

export interface MatrixIntroProps {
  onComplete: () => void
}

type Phase = "rain" | "text" | "fadeout" | "done"

// ── Glitch-decode characters ──
const DECODE_CHARS = "ΣΦΩΔΘΛΞΠψαβγ@#$%&01?!+=<>"

// ── Timing ──
const TIMING = {
  rainOnly: 2800,       // rain builds before text starts
  textDuration: 7500,   // total time for text phase
  fadeout: 1400,        // fade everything out
  cleanup: 500,         // remove component
}

const TITLE = "LocMemory"
const SUBTITLE = "COGNITIVE MEMORY GRAPH"

export const MatrixIntro: React.FC<MatrixIntroProps> = ({ onComplete }) => {
  const [phase, setPhase] = useState<Phase>("rain")
  const [show, setShow] = useState(true)

  // Glitch-decode state for title
  const [titleChars, setTitleChars] = useState<string[]>(
    Array(TITLE.length).fill("")
  )
  const [resolvedCount, setResolvedCount] = useState(0)
  const [showCursor, setShowCursor] = useState(false)
  const [showSubtitle, setShowSubtitle] = useState(false)
  const [showLine, setShowLine] = useState(false)

  // Subtitle typewriter
  const [subStep, setSubStep] = useState(0)

  const scrambleTimers = useRef<ReturnType<typeof setInterval>[]>([])

  // ── Phase transitions ──
  useEffect(() => {
    const total = TIMING.rainOnly + TIMING.textDuration + TIMING.fadeout + TIMING.cleanup
    const t1 = setTimeout(() => setPhase("text"), TIMING.rainOnly)
    const t2 = setTimeout(
      () => setPhase("fadeout"),
      TIMING.rainOnly + TIMING.textDuration
    )
    const t3 = setTimeout(
      () => setPhase("done"),
      TIMING.rainOnly + TIMING.textDuration + TIMING.fadeout
    )
    const t4 = setTimeout(() => {
      setShow(false)
      onComplete()
    }, total)

    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
      clearTimeout(t3)
      clearTimeout(t4)
    }
    // eslint-disable-next-line
  }, [])

  // ── Glitch-decode title ──
  const startDecode = useCallback(() => {
    setShowCursor(true)

    TITLE.split("").forEach((finalChar, i) => {
      const charDelay = i * 450 // Slower: 450ms between each character resolve

      // Start scrambling this position
      const scramble = setInterval(() => {
        setTitleChars((prev) => {
          const next = [...prev]
          next[i] = DECODE_CHARS[Math.floor(Math.random() * DECODE_CHARS.length)]
          return next
        })
      }, 60)
      scrambleTimers.current.push(scramble)

      // Resolve after delay
      setTimeout(() => {
        clearInterval(scramble)
        setTitleChars((prev) => {
          const next = [...prev]
          next[i] = finalChar
          return next
        })
        setResolvedCount(i + 1)

        // After last char
        if (i === TITLE.length - 1) {
          setTimeout(() => {
            setShowLine(true)
            setShowSubtitle(true)
          }, 400)
          setTimeout(() => setShowCursor(false), 2000)
        }
      }, charDelay)
    })
  }, [])

  useEffect(() => {
    if (phase === "text") {
      startDecode()
    }
    return () => {
      scrambleTimers.current.forEach(clearInterval)
    }
  }, [phase, startDecode])

  // ── Subtitle typewriter ──
  useEffect(() => {
    if (!showSubtitle) return
    if (subStep >= SUBTITLE.length) return
    const t = setTimeout(() => setSubStep((s) => s + 1), 45)
    return () => clearTimeout(t)
  }, [showSubtitle, subStep])

  if (!show) return null

  const isFading = phase === "fadeout" || phase === "done"

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center"
      style={{
        background: "#020208",
        opacity: phase === "done" ? 0 : 1,
        transition: "opacity 1s ease-out",
      }}
    >
      {/* ── Scanlines ── */}
      <div
        className="fixed inset-0 pointer-events-none z-[2]"
        style={{
          background:
            "repeating-linear-gradient(to bottom, transparent 0px, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px)",
          opacity: isFading ? 0 : 0.7,
          transition: "opacity 1.2s ease",
        }}
      />

      {/* ── Vignette ── */}
      <div
        className="fixed inset-0 pointer-events-none z-[3]"
        style={{
          background:
            "radial-gradient(ellipse 65% 65% at 50% 50%, transparent 20%, rgba(2,2,8,0.4) 55%, rgba(2,2,8,0.85) 85%, rgba(2,2,8,0.98) 100%)",
        }}
      />

      {/* ── Ambient center glow ── */}
      <div
        className="fixed pointer-events-none z-[1]"
        style={{
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 600,
          height: 300,
          borderRadius: "50%",
          background:
            "radial-gradient(ellipse at center, rgba(120,110,255,0.07) 0%, rgba(80,70,200,0.03) 40%, transparent 70%)",
          opacity: isFading ? 0 : 1,
          transition: "opacity 1.4s ease",
          animation: isFading ? "none" : "ambientPulse 4s ease-in-out infinite",
        }}
      />

      {/* ── Rain layer ── */}
      <div
        className="absolute inset-0"
        style={{
          opacity: isFading ? 0 : 1,
          transition: "opacity 1.4s ease",
        }}
      >
        <MatrixRain
          fontSize={14}
          speed={35}
          clearZoneRadius={170}
          density={1.6}
          className="w-full h-full"
        />
      </div>

      {/* ── Branding ── */}
      <div
        className="absolute flex flex-col items-center z-[5]"
        style={{
          opacity: isFading ? 0 : 1,
          transition: "opacity 1.6s ease",
        }}
      >
        {/* Title */}
        <div
          className="flex items-center"
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "clamp(2.2rem, 5.5vw, 4.5rem)",
            fontWeight: 300,
            letterSpacing: "0.4em",
            color: "#fff",
            textShadow:
              "0 0 20px rgba(140,130,255,0.4), 0 0 40px rgba(140,130,255,0.15), 0 0 80px rgba(140,130,255,0.05)",
            minHeight: "1.2em",
            userSelect: "none",
          }}
        >
          {titleChars.map((char, i) => (
            <span
              key={i}
              style={{
                display: "inline-block",
                color: i < resolvedCount ? "#fff" : "rgba(161,157,252,0.7)",
                textShadow:
                  i < resolvedCount
                    ? "0 0 20px rgba(140,130,255,0.4), 0 0 40px rgba(140,130,255,0.15)"
                    : "0 0 10px rgba(140,130,255,0.5)",
                transition: "color 0.15s ease, text-shadow 0.15s ease",
                minWidth: char ? undefined : "0.5em",
              }}
            >
              {char}
            </span>
          ))}
          {showCursor && (
            <span
              style={{
                display: "inline-block",
                width: 2,
                height: "0.8em",
                background: "rgba(161,157,252,0.8)",
                marginLeft: 6,
                animation: "cursorBlink 1s steps(1) infinite",
              }}
            />
          )}
        </div>

        {/* Horizontal line */}
        <div
          style={{
            marginTop: 24,
            height: 1,
            width: showLine ? "clamp(140px, 28vw, 380px)" : 0,
            background:
              "linear-gradient(90deg, transparent, rgba(161,157,252,0.5) 20%, rgba(100,200,255,0.3) 50%, rgba(161,157,252,0.5) 80%, transparent)",
            boxShadow: "0 0 15px rgba(140,130,255,0.2)",
            opacity: showLine ? 1 : 0,
            transition: "width 1.8s cubic-bezier(0.25,0.46,0.45,0.94), opacity 0.6s ease",
          }}
        />

        {/* Subtitle */}
        <span
          style={{
            marginTop: 18,
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "clamp(0.55rem, 1.1vw, 0.78rem)",
            fontWeight: 300,
            letterSpacing: "0.55em",
            textTransform: "uppercase" as const,
            color: showSubtitle ? "rgba(161,157,252,0.4)" : "transparent",
            transition: "color 1.5s ease",
            userSelect: "none",
            whiteSpace: "pre" as const,
          }}
        >
          {SUBTITLE.slice(0, subStep)}
          {showSubtitle && subStep < SUBTITLE.length && (
            <span
              style={{
                animation: "cursorBlink 0.8s steps(1) infinite",
                color: "rgba(161,157,252,0.5)",
              }}
            >
              _
            </span>
          )}
        </span>
      </div>

      {/* ── Inline keyframes ── */}
      <style>{`
        @keyframes cursorBlink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        @keyframes ambientPulse {
          0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
          50% { transform: translate(-50%, -50%) scale(1.08); opacity: 1; }
        }
      `}</style>
    </div>
  )
}
