import { useEffect, useRef, useState } from "react"

/* ─── Neural constellation background ─── */
function ConstellationCanvas() {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let raf: number
    let W = 0, H = 0

    const resize = () => {
      W = canvas.width = canvas.offsetWidth
      H = canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener("resize", resize)

    const COUNT = 80
    const nodes = Array.from({ length: COUNT }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.18,
      vy: (Math.random() - 0.5) * 0.18,
      r: Math.random() * 1.2 + 0.4,
    }))

    const draw = () => {
      ctx.clearRect(0, 0, W, H)

      // connections
      for (let i = 0; i < COUNT; i++) {
        for (let j = i + 1; j < COUNT; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 130) {
            const a = (1 - dist / 130) * 0.18
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.strokeStyle = `rgba(0,196,188,${a})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }

      // nodes
      for (const n of nodes) {
        ctx.beginPath()
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        ctx.fillStyle = "rgba(0,196,188,0.45)"
        ctx.fill()

        n.x += n.vx
        n.y += n.vy
        if (n.x < 0 || n.x > W) n.vx *= -1
        if (n.y < 0 || n.y > H) n.vy *= -1
      }

      raf = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener("resize", resize)
    }
  }, [])

  return (
    <canvas
      ref={ref}
      className="pointer-events-none absolute inset-0 w-full h-full"
    />
  )
}

/* ─── Typewriter hook ─── */
function useTypewriter(lines: string[], startDelay = 800, charDelay = 38) {
  const [displayed, setDisplayed] = useState<string[]>(lines.map(() => ""))
  const [done, setDone] = useState(false)

  useEffect(() => {
    let cancelled = false
    let lineIdx = 0
    let charIdx = 0

    const step = () => {
      if (cancelled) return
      if (lineIdx >= lines.length) { setDone(true); return }

      const li = lineIdx
      const ci = charIdx
      setDisplayed(prev => {
        if (li >= lines.length || !lines[li]) return prev
        const next = [...prev]
        next[li] = lines[li].slice(0, ci + 1)
        return next
      })

      charIdx++
      if (charIdx >= lines[lineIdx].length) {
        lineIdx++
        charIdx = 0
        setTimeout(step, charDelay * 6)
      } else {
        setTimeout(step, charDelay)
      }
    }

    const t = setTimeout(step, startDelay)
    return () => { cancelled = true; clearTimeout(t) }
  }, [])

  return { displayed, done }
}

/* ─── Animated ring sigil ─── */
function Sigil() {
  return (
    <div className="relative flex items-center justify-center" style={{ width: 72, height: 72 }}>
      <svg width="72" height="72" viewBox="0 0 72 72" fill="none" className="absolute inset-0" style={{ animation: "spin-slow 12s linear infinite" }}>
        <circle cx="36" cy="36" r="30" stroke="rgba(0,196,188,0.25)" strokeWidth="1" strokeDasharray="6 4" />
      </svg>
      <svg width="72" height="72" viewBox="0 0 72 72" fill="none" className="absolute inset-0" style={{ animation: "spin-slow 8s linear infinite reverse" }}>
        <circle cx="36" cy="36" r="22" stroke="rgba(0,196,188,0.18)" strokeWidth="1" strokeDasharray="3 5" />
      </svg>
      <div
        className="relative z-10 h-8 w-8 rounded-full"
        style={{
          background: "radial-gradient(circle, rgba(0,196,188,0.6) 0%, rgba(0,196,188,0.1) 60%, transparent 100%)",
          boxShadow: "0 0 24px rgba(0,196,188,0.6), 0 0 48px rgba(0,196,188,0.2)",
          animation: "pulse-orb 3s ease-in-out infinite",
        }}
      />
    </div>
  )
}

/* ─── Creator card ─── */
function CreatorCard({
  initial, photo, displayName, fullName, surname, role, color, delay,
}: {
  initial: string
  photo: string
  displayName: string
  fullName: string
  surname: string
  role: string
  color: string
  delay: number
}) {
  const rgb = color === "teal" ? "0,196,188" : "179,136,255"
  const hex = color === "teal" ? "#00c4bc" : "#b388ff"
  const [imgFailed, setImgFailed] = useState(false)

  return (
    <div
      className="relative flex flex-col items-center gap-5 px-8 py-8"
      style={{
        animation: `fade-up 0.8s ease forwards`,
        animationDelay: `${delay}ms`,
        opacity: 0,
        width: 240,
        height: 340,
        flexShrink: 0,
      }}
    >
      {/* Glass card bg */}
      <div
        className="absolute inset-0 rounded-2xl"
        style={{
          background: `linear-gradient(135deg, rgba(${rgb},0.07) 0%, rgba(0,0,0,0.3) 100%)`,
          border: `1px solid rgba(${rgb},0.2)`,
          backdropFilter: "blur(12px)",
          boxShadow: `0 0 40px rgba(${rgb},0.08), inset 0 1px 0 rgba(${rgb},0.15)`,
        }}
      />

      {/* Corner accents */}
      <div className="absolute top-3 left-3 h-3 w-3" style={{ borderTop: `1px solid rgba(${rgb},0.5)`, borderLeft: `1px solid rgba(${rgb},0.5)` }} />
      <div className="absolute top-3 right-3 h-3 w-3" style={{ borderTop: `1px solid rgba(${rgb},0.5)`, borderRight: `1px solid rgba(${rgb},0.5)` }} />
      <div className="absolute bottom-3 left-3 h-3 w-3" style={{ borderBottom: `1px solid rgba(${rgb},0.5)`, borderLeft: `1px solid rgba(${rgb},0.5)` }} />
      <div className="absolute bottom-3 right-3 h-3 w-3" style={{ borderBottom: `1px solid rgba(${rgb},0.5)`, borderRight: `1px solid rgba(${rgb},0.5)` }} />

      {/* Avatar */}
      <div className="relative z-10 mt-2">
        <div
          className="h-20 w-20 rounded-full overflow-hidden flex items-center justify-center text-2xl font-bold"
          style={{
            border: `1px solid rgba(${rgb},0.4)`,
            boxShadow: `0 0 32px rgba(${rgb},0.25), 0 0 64px rgba(${rgb},0.1)`,
            background: imgFailed
              ? `radial-gradient(circle at 35% 35%, rgba(${rgb},0.3) 0%, rgba(${rgb},0.06) 70%)`
              : "transparent",
            color: hex,
            textShadow: `0 0 16px rgba(${rgb},1)`,
            fontFamily: "Georgia, serif",
            animation: `pulse-avatar 4s ease-in-out infinite`,
            animationDelay: `${delay}ms`,
          }}
        >
          {!imgFailed ? (
            <img
              src={photo}
              alt={fullName}
              onError={() => setImgFailed(true)}
              className="w-full h-full object-cover"
            />
          ) : initial}
        </div>
        {/* Outer ring */}
        <div
          className="absolute -inset-2 rounded-full"
          style={{
            border: `1px solid rgba(${rgb},0.12)`,
            animation: "spin-slow 10s linear infinite",
          }}
        />
      </div>

      {/* Name — fixed layout so both cards stay the same height */}
      <div className="relative z-10 flex flex-col items-center gap-1 flex-1 justify-center">
        <span
          className="text-[9px] tracking-[0.25em] uppercase"
          style={{ color: `rgba(${rgb},0.45)`, letterSpacing: "0.22em" }}
        >
          {fullName}
        </span>
        <span
          className="text-2xl tracking-[0.15em] uppercase font-semibold"
          style={{
            color: hex,
            textShadow: `0 0 16px rgba(${rgb},0.7), 0 0 32px rgba(${rgb},0.3)`,
            letterSpacing: "0.12em",
          }}
        >
          {displayName}
        </span>
        <span
          className="text-sm tracking-[0.2em] uppercase font-light"
          style={{ color: "#e8fff9", letterSpacing: "0.18em" }}
        >
          {surname}
        </span>
        <div className="mt-2 px-3 py-0.5 rounded-full" style={{ background: `rgba(${rgb},0.1)`, border: `1px solid rgba(${rgb},0.2)` }}>
          <span className="text-[8px] uppercase tracking-[0.3em]" style={{ color: `rgba(${rgb},0.8)` }}>
            {role}
          </span>
        </div>
      </div>
    </div>
  )
}

/* ─── Main page ─── */
export function AboutPage() {
  const lines = [
    "What I cannot create, I do not understand.",
    "Know how to solve every problem that has been solved.",
  ]
  const { displayed, done } = useTypewriter(lines, 1200, 36)

  return (
    <div
      className="relative min-h-screen w-full flex flex-col items-center justify-center font-mono overflow-auto py-16"
      style={{ background: "#020d0d" }}
    >
      {/* Deep radial glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: "radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,196,188,0.055) 0%, transparent 70%)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: "radial-gradient(ellipse 50% 40% at 50% 100%, rgba(179,136,255,0.04) 0%, transparent 70%)",
        }}
      />

      {/* Constellation */}
      <ConstellationCanvas />

      {/* Scanlines */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 4px)",
          zIndex: 1,
        }}
      />

      {/* Large corner brackets */}
      {[
        { top: 20, left: 20, t: true, l: true, c: "rgba(0,196,188,0.45)" },
        { top: 20, right: 20, t: true, r: true, c: "rgba(255,77,109,0.3)" },
        { bottom: 20, left: 20, b: true, l: true, c: "rgba(255,140,38,0.3)" },
        { bottom: 20, right: 20, b: true, r: true, c: "rgba(0,196,188,0.25)" },
      ].map((br, i) => (
        <div
          key={i}
          className="pointer-events-none absolute h-12 w-12"
          style={{
            top: br.top, left: br.left, bottom: br.bottom, right: br.right,
            borderTop: br.t ? `1px solid ${br.c}` : undefined,
            borderBottom: br.b ? `1px solid ${br.c}` : undefined,
            borderLeft: br.l ? `1px solid ${br.c}` : undefined,
            borderRight: br.r ? `1px solid ${br.c}` : undefined,
            filter: `drop-shadow(0 0 6px ${br.c})`,
            zIndex: 2,
          }}
        />
      ))}

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center gap-14 w-full max-w-4xl px-8">

        {/* Top badge */}
        <div
          className="flex flex-col items-center gap-3"
          style={{ animation: "fade-down 0.7s ease forwards", opacity: 0 }}
        >
          <div
            className="flex items-center gap-2 px-5 py-1.5"
            style={{
              border: "1px solid rgba(0,196,188,0.2)",
              boxShadow: "0 0 20px rgba(0,196,188,0.06)",
              background: "rgba(0,196,188,0.04)",
            }}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" style={{ boxShadow: "0 0 6px rgba(74,222,128,0.9)", animation: "pulse 2s ease-in-out infinite" }} />
            <span className="text-[9px] uppercase tracking-[0.45em]" style={{ color: "rgba(0,196,188,0.7)" }}>
              LocMemory &nbsp;·&nbsp; v0.1.0
            </span>
          </div>
          <span className="text-[8px] uppercase tracking-[0.35em]" style={{ color: "rgba(0,196,188,0.3)" }}>
            local intelligence · built by humans
          </span>
        </div>

        {/* Quote section */}
        <div
          className="relative w-full flex flex-col items-center"
          style={{ animation: "fade-up 0.8s ease forwards", animationDelay: "400ms", opacity: 0 }}
        >
          {/* Divider with label */}
          <div className="w-full flex items-center gap-5 mb-12">
            <div className="flex-1 h-px" style={{ background: "linear-gradient(to right, transparent, rgba(0,196,188,0.35))" }} />
            <div className="flex items-center gap-2">
              <div className="h-px w-3" style={{ background: "rgba(0,196,188,0.5)" }} />
              <span className="text-[8px] uppercase tracking-[0.45em]" style={{ color: "rgba(0,196,188,0.55)" }}>dedication</span>
              <div className="h-px w-3" style={{ background: "rgba(0,196,188,0.5)" }} />
            </div>
            <div className="flex-1 h-px" style={{ background: "linear-gradient(to left, transparent, rgba(0,196,188,0.35))" }} />
          </div>

          {/* Quote */}
          <div className="relative text-center w-full px-4">
            {/* Large ghost quote marks */}
            <span
              className="pointer-events-none absolute select-none"
              style={{
                top: -32, left: 8,
                fontSize: 120,
                lineHeight: 1,
                fontFamily: "Georgia, serif",
                color: "rgba(0,196,188,0.07)",
                zIndex: 0,
              }}
            >&ldquo;</span>
            <span
              className="pointer-events-none absolute select-none"
              style={{
                bottom: -32, right: 8,
                fontSize: 120,
                lineHeight: 1,
                fontFamily: "Georgia, serif",
                color: "rgba(0,196,188,0.07)",
                zIndex: 0,
              }}
            >&rdquo;</span>

            {/* Line 1 */}
            <p
              className="relative z-10 text-3xl md:text-4xl lg:text-5xl leading-tight mb-2"
              style={{
                fontFamily: "Georgia, 'Times New Roman', serif",
                fontWeight: 300,
                fontStyle: "italic",
                color: "#d4f7f5",
                letterSpacing: "0.01em",
                textShadow: "0 0 40px rgba(0,196,188,0.25)",
                minHeight: "1.3em",
              }}
            >
              {displayed[0]}
              {displayed[0].length < lines[0].length && (
                <span style={{ borderRight: "2px solid #00c4bc", animation: "blink 0.8s step-end infinite", marginLeft: 2 }} />
              )}
            </p>

            {/* Line 2 */}
            {displayed[0].length >= lines[0].length && (
              <p
                className="relative z-10 text-xl md:text-2xl lg:text-3xl leading-tight mt-5"
                style={{
                  fontFamily: "Georgia, 'Times New Roman', serif",
                  fontWeight: 300,
                  fontStyle: "italic",
                  color: "rgba(212,247,245,0.6)",
                  letterSpacing: "0.01em",
                  textShadow: "0 0 24px rgba(0,196,188,0.15)",
                  minHeight: "1.4em",
                }}
              >
                {displayed[1]}
                {displayed[1].length < lines[1].length && (
                  <span style={{ borderRight: "2px solid rgba(0,196,188,0.6)", animation: "blink 0.8s step-end infinite", marginLeft: 2 }} />
                )}
              </p>
            )}
          </div>

          {/* Attribution */}
          <div
            className="mt-12 flex items-center gap-4"
            style={{
              opacity: done ? 1 : 0,
              transition: "opacity 1.2s ease",
              transitionDelay: "0.4s",
            }}
          >
            <div className="h-px w-10" style={{ background: "linear-gradient(to right, transparent, rgba(0,196,188,0.4))" }} />
            <div className="flex flex-col items-center gap-1">
              <span className="text-[10px] uppercase tracking-[0.4em]" style={{ color: "rgba(0,196,188,0.6)" }}>
                Richard P. Feynman
              </span>
              <span className="text-[8px] tracking-[0.25em]" style={{ color: "rgba(0,196,188,0.3)" }}>
                Theoretical Physicist &nbsp;·&nbsp; Nobel Laureate
              </span>
            </div>
            <div className="h-px w-10" style={{ background: "linear-gradient(to left, transparent, rgba(0,196,188,0.4))" }} />
          </div>
        </div>

        {/* Sigil divider */}
        <div
          style={{
            opacity: done ? 1 : 0,
            transition: "opacity 1s ease",
            transitionDelay: "0.8s",
          }}
        >
          <Sigil />
        </div>

        {/* Creators */}
        <div
          className="flex flex-col items-center gap-8 w-full"
          style={{
            opacity: done ? 1 : 0,
            transition: "opacity 0.8s ease",
            transitionDelay: "1s",
          }}
        >
          <span className="text-[8px] uppercase tracking-[0.45em]" style={{ color: "rgba(0,196,188,0.3)" }}>
            created by
          </span>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6 w-full">
            <CreatorCard
              initial="T"
              photo="/Almouthana_Taha_khalfallah.png"
              displayName=""
              fullName="Almouthana Taha"
              surname="Khalfallah"
              role="Co-creator"
              color="teal"
              delay={1200}
            />

            {/* Central connector */}
            <div className="flex flex-col items-center gap-2 py-4">
              <div className="h-8 w-px sm:h-px sm:w-8" style={{ background: "linear-gradient(to bottom, transparent, rgba(0,196,188,0.3), transparent)" }} />
              <span className="text-[10px] tracking-[0.2em]" style={{ color: "rgba(0,196,188,0.3)" }}>&amp;</span>
              <div className="h-8 w-px sm:h-px sm:w-8" style={{ background: "linear-gradient(to bottom, transparent, rgba(0,196,188,0.3), transparent)" }} />
            </div>

            <CreatorCard
              initial="E"
              photo="/Eya_Dhrif.png"
              displayName=""
              fullName="Eya"
              surname="Dhrif"
              role="Co-creator"
              color="violet"
              delay={1500}
            />
          </div>

          {/* Tagline */}
          <div
            className="flex items-center gap-3 mt-2"
            style={{ animation: "fade-up 0.8s ease forwards", animationDelay: "1800ms", opacity: 0 }}
          >
            <div className="h-px w-8" style={{ background: "linear-gradient(to right, transparent, rgba(0,196,188,0.2))" }} />
            <span
              className="text-[9px] uppercase tracking-[0.4em] text-center"
              style={{ color: "rgba(0,196,188,0.35)", fontStyle: "italic" }}
            >
              built together &nbsp;·&nbsp; understood together
            </span>
            <div className="h-px w-8" style={{ background: "linear-gradient(to left, transparent, rgba(0,196,188,0.2))" }} />
          </div>
        </div>

        {/* Footer */}
        <div
          className="flex items-center gap-4 text-[7px] uppercase tracking-[0.3em]"
          style={{ color: "rgba(0,196,188,0.18)", animation: "fade-up 0.6s ease forwards", animationDelay: "2000ms", opacity: 0 }}
        >
          <span>SYS:LOCMEMORY</span>
          <span>·</span>
          <span>BUILD 2025.04</span>
          <span>·</span>
          <span>ALL RIGHTS RESERVED</span>
        </div>

      </div>

      <style>{`
        @keyframes fade-up {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fade-down {
          from { opacity: 0; transform: translateY(-14px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        @keyframes pulse-orb {
          0%, 100% { transform: scale(1);   box-shadow: 0 0 24px rgba(0,196,188,0.6), 0 0 48px rgba(0,196,188,0.2); }
          50%       { transform: scale(1.15); box-shadow: 0 0 36px rgba(0,196,188,0.9), 0 0 72px rgba(0,196,188,0.35); }
        }
        @keyframes pulse-avatar {
          0%, 100% { box-shadow: 0 0 32px rgba(0,196,188,0.25), 0 0 64px rgba(0,196,188,0.10); }
          50%       { box-shadow: 0 0 48px rgba(0,196,188,0.45), 0 0 80px rgba(0,196,188,0.20); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.7; }
          50%       { opacity: 1; }
        }
      `}</style>
    </div>
  )
}
