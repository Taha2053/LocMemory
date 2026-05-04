import { useEffect, useRef } from "react"
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  Color,
  Vector2,
  Vector3,
  Raycaster,
  Object3D,
  MathUtils,
  Mesh,
  BufferGeometry,
  InstancedMesh,
  MeshBasicMaterial,
  AdditiveBlending,
  NormalBlending,
  LineSegments,
  LineBasicMaterial,
  Float32BufferAttribute,
  WireframeGeometry,
  PlaneGeometry,
  CanvasTexture,
  Points,
  PointsMaterial,
  BackSide,
  FrontSide,
} from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import gsap from "gsap"
import { api } from "@/lib/api"
import { domainColor } from "@/lib/domainColors"

function createGlowTexture(): CanvasTexture {
  const size = 128
  const canvas = document.createElement("canvas")
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext("2d")!
  const half = size / 2
  const gradient = ctx.createRadialGradient(half, half, 0, half, half, half)
  gradient.addColorStop(0.00, "rgba(255,255,255,1.00)")
  gradient.addColorStop(0.25, "rgba(255,255,255,0.95)")
  gradient.addColorStop(0.45, "rgba(255,255,255,0.45)")
  gradient.addColorStop(0.70, "rgba(255,255,255,0.10)")
  gradient.addColorStop(1.00, "rgba(255,255,255,0.00)")
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)
  return new CanvasTexture(canvas)
}

function hashId(str: string): number {
  let h = 0xdeadbeef
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 2654435761)
  }
  return (h ^ (h >>> 16)) >>> 0
}

// Base scale in world units — using unit PlaneGeometry(1,1)
const MARKER_BASE_SCALE = 0.07
const ZOOM_DISTANCE = 0.35
// Pull every node this fraction of the way from the surface vertex toward the
// brain centroid, guaranteeing the marker + its halo stay inside the wireframe.
// 0.30 = 30% inward — enough headroom for the 6× bloom halo without making
// nodes feel detached from the cortex.
const INWARD_PULL = 0.32
const DEFAULT_CAM_Z = 1.9
const DEFAULT_CAM_Z_MOBILE = 2.9

// Pull a surface vertex inward toward centroid by INWARD_PULL fraction.
function pullInside(pos: Vector3, center: Vector3): Vector3 {
  return pos.clone().lerp(center, INWARD_PULL)
}

interface BrainSceneProps extends React.HTMLAttributes<HTMLDivElement> {
  modelUrl?: string
  onNodeSelect?: (id: string) => void
  selectedId?: string | null
  showEdges?: boolean
}

export function BrainScene({
  modelUrl = "/brain.glb",
  onNodeSelect,
  selectedId,
  showEdges = true,
  ...props
}: BrainSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const selectedRef = useRef<string | null>(selectedId ?? null)
  const onSelectRef = useRef(onNodeSelect)
  const zoomFnRef = useRef<((id: string | null) => void) | null>(null)

  useEffect(() => {
    selectedRef.current = selectedId ?? null
    zoomFnRef.current?.(selectedId ?? null)
  }, [selectedId])
  useEffect(() => { onSelectRef.current = onNodeSelect }, [onNodeSelect])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Mutable size object shared by raycaster, onResize, and ResizeObserver
    const size = { width: 800, height: 600 }

    const scene = new Scene()
    const camera = new PerspectiveCamera(75, size.width / size.height, 0.1, 100)
    const isMobile = window.innerWidth < 767
    const startZ = isMobile ? DEFAULT_CAM_Z_MOBILE : DEFAULT_CAM_Z
    camera.position.set(0, 0, startZ)

    const renderer = new WebGLRenderer({
      alpha: true,
      antialias: window.devicePixelRatio === 1,
      stencil: true,
    })
    renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio))
    container.appendChild(renderer.domElement)

    // Apply real container dimensions and update camera aspect
    const applySize = (w: number, h: number) => {
      size.width = w
      size.height = h
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }

    // Use ResizeObserver so we get the true post-layout size.
    // It fires synchronously before the first paint once the container
    // has non-zero dimensions, eliminating the need to wait for a resize.
    let sizeObserver: ResizeObserver | null = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          applySize(width, height)
          sizeObserver?.disconnect()
          sizeObserver = null
          break
        }
      }
    })
    sizeObserver.observe(container)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.enablePan = false
    // Zoom constraints: keep brain reasonably sized in viewport
    controls.minDistance = 1.2
    controls.maxDistance = 2.5
    controls.rotateSpeed = 0.6
    controls.zoomSpeed = 0.7
    controls.target.set(0, 0, 0)

    const raycaster = new Raycaster()
    let frameId = 0
    let disposed = false

    // Brain visual refs
    let brainWireframe: LineSegments | null = null
    let brainHullInner: Mesh | null = null
    let brainHullOuter: Mesh | null = null
    let brainDots: Points | null = null
    let stars: Points | null = null
    let nebula: Points | null = null
    // Stencil clipping group for nodes/edges
    let clipGroup: Object3D | null = null

    // Shared glow texture for all three bloom layers
    const glowTex = createGlowTexture()

    // Three-layer glow system:
    //   markersMesh = inner bright core (also hit-tested for raycasting)
    //   haloMesh    = mid soft halo (~2.5× core)
    //   bloomMesh   = outer feathered nebula (~6× core)
    let markersMesh: InstancedMesh | null = null
    let haloMesh: InstancedMesh | null = null
    let bloomMesh: InstancedMesh | null = null
    let edgesLine: LineSegments | null = null
    let nodeIds: string[] = []
    const nodeIndexById = new Map<string, number>()
    const nodePositions: Vector3[] = []
    const nodeTiers: number[] = []
    // Per-node domain colors for stroke
    const nodeCoreColors: Color[] = []
    // Per-node base scale (not used in flat design)
    const baseScales: number[] = []
    let hoveredIdx = -1
    let prevHoveredIdx = -1

    const tmpObj = new Object3D()

    let prevSelectedIdx = -1
    const applySelection = (id: string | null) => {
      const newIdx = id ? nodeIndexById.get(id) ?? -1 : -1

      if (markersMesh) {
        // Reset previous selection — restore the node's domain color.
        if (prevSelectedIdx >= 0 && prevSelectedIdx !== newIdx) {
          baseScales[prevSelectedIdx] = MARKER_BASE_SCALE
          markersMesh.setColorAt(prevSelectedIdx, nodeCoreColors[prevSelectedIdx])
          if (haloMesh?.instanceColor) haloMesh.setColorAt(prevSelectedIdx, nodeCoreColors[prevSelectedIdx])
          if (bloomMesh?.instanceColor) bloomMesh.setColorAt(prevSelectedIdx, nodeCoreColors[prevSelectedIdx])
        }

        // Apply new selection — brighter core, larger scale.
        if (newIdx >= 0) {
          const selCol = nodeCoreColors[newIdx].clone().lerp(new Color(0xffffff), 0.5)
          markersMesh.setColorAt(newIdx, selCol)
          if (haloMesh?.instanceColor) haloMesh.setColorAt(newIdx, nodeCoreColors[newIdx])
          if (bloomMesh?.instanceColor) bloomMesh.setColorAt(newIdx, nodeCoreColors[newIdx])
          baseScales[newIdx] = MARKER_BASE_SCALE * 1.35
        }

        if (markersMesh.instanceColor) markersMesh.instanceColor.needsUpdate = true
        if (haloMesh?.instanceColor) haloMesh.instanceColor.needsUpdate = true
        if (bloomMesh?.instanceColor) bloomMesh.instanceColor.needsUpdate = true

        prevSelectedIdx = newIdx
      }

      // Dim brain when a memory is focused
      const focused = newIdx >= 0
      if (brainWireframe) {
        gsap.to(brainWireframe.material as LineBasicMaterial, {
          opacity: focused ? 0.08 : 0.22,
          duration: 0.7,
          ease: "power2.out",
        })
      }
      if (brainDots) {
        gsap.to(brainDots.material as PointsMaterial, {
          opacity: focused ? 0.18 : 0.9,
          duration: 0.7,
          ease: "power2.out",
        })
      }
      if (brainHullInner) {
        gsap.to((brainHullInner.material as MeshBasicMaterial), {
          opacity: focused ? 0.25 : 0.55,
          duration: 0.7,
          ease: "power2.out",
        })
      }
      if (brainHullOuter) {
        gsap.to((brainHullOuter.material as MeshBasicMaterial), {
          opacity: focused ? 0.025 : 0.06,
          duration: 0.7,
          ease: "power2.out",
        })
      }
      if (edgesLine) {
        gsap.to(edgesLine.material as LineBasicMaterial, {
          opacity: focused ? 0.1 : 0.5,
          duration: 0.7,
          ease: "power2.out",
        })
      }

      // Camera fly
      gsap.killTweensOf(camera.position)
      gsap.killTweensOf(controls.target)
      if (newIdx >= 0) {
        const memPos = nodePositions[newIdx]
        const camTarget = memPos.clone().normalize().multiplyScalar(memPos.length() + ZOOM_DISTANCE)
        gsap.to(camera.position, {
          x: camTarget.x, y: camTarget.y, z: camTarget.z,
          duration: 0.9, ease: "power3.inOut",
          onUpdate: () => controls.update(),
        })
        gsap.to(controls.target, {
          x: memPos.x, y: memPos.y, z: memPos.z,
          duration: 0.9, ease: "power3.inOut",
          onUpdate: () => controls.update(),
        })
      } else {
        gsap.to(camera.position, {
          x: 0, y: 0, z: startZ,
          duration: 0.9, ease: "power3.inOut",
          onUpdate: () => controls.update(),
        })
        gsap.to(controls.target, {
          x: 0, y: 0, z: 0,
          duration: 0.9, ease: "power3.inOut",
          onUpdate: () => controls.update(),
        })
      }
    }
    zoomFnRef.current = applySelection

    const onMousemove = (e: MouseEvent) => {
      if (!markersMesh) return
      const rect = container.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / size.width) * 2 - 1
      const y = -((e.clientY - rect.top) / size.height) * 2 + 1
      raycaster.setFromCamera(new Vector2(x, y), camera)

      const markerHits = raycaster.intersectObject(markersMesh, false)
      if (markerHits.length > 0 && markerHits[0].instanceId !== undefined) {
        hoveredIdx = markerHits[0].instanceId
        container.style.cursor = "pointer"
      } else {
        hoveredIdx = -1
        container.style.cursor = ""
      }
    }

    let downX = 0, downY = 0
    const onPointerDown = (e: PointerEvent) => { downX = e.clientX; downY = e.clientY }
    const onClick = (e: MouseEvent) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return
      if (!markersMesh) return
      const rect = container.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / size.width) * 2 - 1
      const y = -((e.clientY - rect.top) / size.height) * 2 + 1
      raycaster.setFromCamera(new Vector2(x, y), camera)
      const hits = raycaster.intersectObject(markersMesh, false)
      if (hits.length > 0 && hits[0].instanceId !== undefined) {
        const id = nodeIds[hits[0].instanceId]
        onSelectRef.current?.(id)
      }
    }

    const onResize = () => {
      const w = container.clientWidth || 800
      const h = container.clientHeight || 600
      applySize(w, h)
    }

    const animate = () => {
      if (disposed) return
      controls.update()

      // Slow Y rotation — keep brain shells in lockstep
      if (brainWireframe) brainWireframe.rotation.y += 0.0008
      if (brainDots) brainDots.rotation.y += 0.0008
      if (brainHullInner) brainHullInner.rotation.y += 0.0008
      if (brainHullOuter) brainHullOuter.rotation.y += 0.0008
      if (stars) {
        stars.rotation.y -= 0.00015
        stars.rotation.x += 0.00008
        // Twinkling effect
        const twinkle = 0.5 + 0.5 * Math.sin((Date.now() / 800) * Math.PI * 2)
        ;(stars.material as PointsMaterial).opacity = 0.5 + twinkle * 0.4
      }
      // Nebula slow rotation
      if (nebula) {
        nebula.rotation.y -= 0.00005
        nebula.rotation.z += 0.00003
        const nebulaPulse = 0.3 + 0.15 * Math.sin((Date.now() / 3000) * Math.PI * 2)
        ;(nebula.material as PointsMaterial).opacity = nebulaPulse
      }

      // Slow atmospheric breath on the outer halo (~5s cycle)
      if (brainHullOuter) {
        const breath = 1.0 + 0.012 * Math.sin((Date.now() / 2500) * Math.PI)
        brainHullOuter.scale.setScalar(1.015 * breath)
      }

      // Update hover scale in baseScales
      if (markersMesh && nodeIds.length > 0 && hoveredIdx !== prevHoveredIdx) {
        if (prevHoveredIdx >= 0 && prevHoveredIdx !== prevSelectedIdx) {
          baseScales[prevHoveredIdx] = MARKER_BASE_SCALE
        }
        if (hoveredIdx >= 0 && hoveredIdx !== prevSelectedIdx) {
          baseScales[hoveredIdx] = MARKER_BASE_SCALE * 1.5
        }
        prevHoveredIdx = hoveredIdx
      }

      // Billboard all three glow layers every frame + apply breathing pulse
      if (markersMesh && nodePositions.length > 0) {
        // Slow breathing animation ~3s cycle
        const pulse = 0.92 + 0.08 * Math.sin((Date.now() / 1500) * Math.PI * 2)

        for (let i = 0; i < nodePositions.length; i++) {
          // Selected and hovered nodes don't pulse — they stay sharp and bright
          const isFixed = i === prevSelectedIdx || i === hoveredIdx
          const s = isFixed ? baseScales[i] : baseScales[i] * pulse

          tmpObj.position.copy(nodePositions[i])
          tmpObj.quaternion.copy(camera.quaternion) // Billboard: always face camera

          // Inner bright core
          tmpObj.scale.setScalar(s)
          tmpObj.updateMatrix()
          markersMesh.setMatrixAt(i, tmpObj.matrix)

          // Mid soft halo (2.5× larger)
          if (haloMesh) {
            tmpObj.scale.setScalar(s * 1.6)
            tmpObj.updateMatrix()
            haloMesh.setMatrixAt(i, tmpObj.matrix)
          }

          // Outer feathered bloom (6× larger)
          if (bloomMesh) {
            tmpObj.scale.setScalar(s * 2.4)
            tmpObj.updateMatrix()
            bloomMesh.setMatrixAt(i, tmpObj.matrix)
          }
        }

        markersMesh.instanceMatrix.needsUpdate = true
        if (haloMesh) haloMesh.instanceMatrix.needsUpdate = true
        if (bloomMesh) bloomMesh.instanceMatrix.needsUpdate = true
      }

      renderer.render(scene, camera)
      frameId = requestAnimationFrame(animate)
    }

    const loader = new GLTFLoader()

    loader.load(modelUrl, (gltf) => {
      if (disposed) return
      const brainMesh = gltf.scene.children[0] as Mesh

      // ── Inner translucent hull (gives the brain real volume / depth) ──
      // Backside rendering only — front faces are skipped so the wireframe and
      // nodes inside are not occluded. The dark teal interior reads as
      // bioluminescent depth behind the cortex pattern.
      const innerHullMat = new MeshBasicMaterial({
        color: new Color(0x00332b),
        transparent: true,
        opacity: 0.55,
        side: BackSide,
        depthWrite: false,
        blending: NormalBlending,
      })
      brainHullInner = new Mesh(brainMesh.geometry, innerHullMat)
      brainHullInner.scale.setScalar(0.99)
      brainHullInner.renderOrder = -2
      scene.add(brainHullInner)

      // ── Outer atmospheric shell (subtle fresnel-like emerald glow) ──
      // Front-side, additive — adds a gentle halo on the brain silhouette
      // that breathes with the pulse animation.
      const outerHullMat = new MeshBasicMaterial({
        color: new Color(0x00b87a),
        transparent: true,
        opacity: 0.06,
        side: FrontSide,
        depthWrite: false,
        blending: AdditiveBlending,
      })
      brainHullOuter = new Mesh(brainMesh.geometry, outerHullMat)
      brainHullOuter.scale.setScalar(1.015)
      brainHullOuter.renderOrder = -1
      scene.add(brainHullOuter)

      // ── Wireframe brain mesh (bioluminescent emerald green) ──
      const wireGeo = new WireframeGeometry(brainMesh.geometry)
      const wireMat = new LineBasicMaterial({
        color: new Color(0x00ff88),
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
        blending: AdditiveBlending,
      })
      brainWireframe = new LineSegments(wireGeo, wireMat)
      scene.add(brainWireframe)

      // ── Glowing node dots at each vertex ──
      const dotsGeo = new BufferGeometry()
      dotsGeo.setAttribute("position", brainMesh.geometry.getAttribute("position").clone())
      const dotsMat = new PointsMaterial({
        color: new Color(0x6effb8),
        size: 0.012,
        transparent: true,
        opacity: 0.9,
        blending: AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      })
      brainDots = new Points(dotsGeo, dotsMat)
      scene.add(brainDots)

      // ── Enhanced star field background ──
      const starVerts: number[] = []
      const starColors: number[] = []
      for (let i = 0; i < 800; i++) {
        const theta = Math.random() * Math.PI * 2
        const phi = Math.acos(2 * Math.random() - 1)
        const r = 4 + Math.random() * 5
        starVerts.push(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        )
        // Mix of emerald, cyan, white stars
        const colorChoice = Math.random()
        if (colorChoice < 0.4) {
          starColors.push(0, 1, 0.53) // emerald
        } else if (colorChoice < 0.7) {
          starColors.push(0, 0.9, 1) // cyan
        } else {
          starColors.push(1, 1, 1) // white
        }
      }
      const starGeo = new BufferGeometry()
      starGeo.setAttribute("position", new Float32BufferAttribute(starVerts, 3))
      starGeo.setAttribute("color", new Float32BufferAttribute(starColors, 3))
      const starMat = new PointsMaterial({
        size: 0.035,
        transparent: true,
        opacity: 0.7,
        vertexColors: true,
        depthWrite: false,
        sizeAttenuation: true,
      })
      stars = new Points(starGeo, starMat)
      scene.add(stars)

      // ── Nebula cloud background ──
      const nebulaVerts: number[] = []
      const nebulaColors: number[] = []
      for (let i = 0; i < 200; i++) {
        const theta = Math.random() * Math.PI * 2
        const phi = Math.acos(2 * Math.random() - 1)
        const r = 6 + Math.random() * 3
        nebulaVerts.push(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        )
        const colorChoice = Math.random()
        if (colorChoice < 0.4) {
          nebulaColors.push(0, 0.3, 0.2, 0.08)
        } else if (colorChoice < 0.7) {
          nebulaColors.push(0, 0.2, 0.3, 0.06)
        } else {
          nebulaColors.push(0.1, 0.05, 0.15, 0.05)
        }
      }
      const nebulaGeo = new BufferGeometry()
      nebulaGeo.setAttribute("position", new Float32BufferAttribute(nebulaVerts, 3))
      nebulaGeo.setAttribute("color", new Float32BufferAttribute(nebulaColors, 4))
      const nebulaMat = new PointsMaterial({
        size: 0.8,
        transparent: true,
        opacity: 0.4,
        vertexColors: true,
        depthWrite: false,
        sizeAttenuation: true,
        blending: AdditiveBlending,
      })
      nebula = new Points(nebulaGeo, nebulaMat)
      scene.add(nebula)

      // ── Load real memories overlaid on the brain ──
      api
        .graph()
        .then(({ nodes, links }) => {
          if (disposed || nodes.length === 0) return
          const positionsArr = brainMesh.geometry.attributes.position.array
          const positionCount = positionsArr.length / 3

          // Centroid = mean of vertex positions (more robust than bbox center
          // for irregular shapes).
          let cx = 0, cy = 0, cz = 0
          for (let i = 0; i < positionCount; i++) {
            cx += positionsArr[i * 3]
            cy += positionsArr[i * 3 + 1]
            cz += positionsArr[i * 3 + 2]
          }
          const brainCenter = new Vector3(cx / positionCount, cy / positionCount, cz / positionCount)

          // Robust "safe shell" radius: take the 70th-percentile distance from
          // centroid across all vertices. Stray geometry (brain-stem, optic
          // nerves, modeling outliers) sits beyond this; the dense cortex hull
          // sits inside. Anything we sample beyond safeRadius gets pulled
          // back along its own direction so it lands on the cortex envelope.
          const dists = new Float32Array(positionCount)
          for (let i = 0; i < positionCount; i++) {
            const dx = positionsArr[i * 3] - brainCenter.x
            const dy = positionsArr[i * 3 + 1] - brainCenter.y
            const dz = positionsArr[i * 3 + 2] - brainCenter.z
            dists[i] = Math.sqrt(dx * dx + dy * dy + dz * dz)
          }
          const sorted = Array.from(dists).sort((a, b) => a - b)
          const safeRadius = sorted[Math.floor(sorted.length * 0.70)]

          // ── Domain anchors ──────────────────────────────────────────────
          // Each domain gets a stable position inside the brain ("functional
          // region"), and all nodes of that domain cluster around it. This
          // gives the graph a brain-region feel — programming over here,
          // health over there — instead of nodes scattered randomly.
          //
          // Anchor direction = unit vector deterministically derived from
          // the domain name (Fibonacci-sphere style mapping over the hash),
          // then placed at 0.55 × safeRadius from centroid so anchors sit in
          // the cortex region with spacing between them.

          const uniqueDomains = Array.from(new Set(nodes.map((n) => n.domain || "unknown")))
          const domainAnchors = new Map<string, Vector3>()
          uniqueDomains.forEach((d) => {
            const h = hashId(d)
            // Map hash → spherical coords. Use top-half + low-half bits so
            // theta and phi are independent.
            const u = ((h & 0xffff) / 0xffff)            // 0..1
            const v = (((h >>> 16) & 0xffff) / 0xffff)   // 0..1
            const theta = u * Math.PI * 2                 // azimuth
            const phi = Math.acos(2 * v - 1)              // inclination (uniform on sphere)
            const dir = new Vector3(
              Math.sin(phi) * Math.cos(theta),
              Math.cos(phi),
              Math.sin(phi) * Math.sin(theta),
            )
            const anchor = brainCenter.clone().add(dir.multiplyScalar(safeRadius * 0.55))
            domainAnchors.set(d, anchor)
          })

          // Cluster radius = how tightly nodes pack around their domain anchor.
          // 22% of safeRadius gives visible clustering with enough internal
          // spread that individual nodes don't all overlap.
          const CLUSTER_RADIUS = safeRadius * 0.22
          const HARD_BOUND = safeRadius * 0.85

          nodeIds = nodes.map((n) => n.id)
          nodes.forEach((n, i) => {
            nodeIndexById.set(n.id, i)
            nodeTiers.push(n.tier)
            baseScales.push(MARKER_BASE_SCALE)

            const anchor = domainAnchors.get(n.domain || "unknown")!
            const h = hashId(n.id)

            // Deterministic offset within the cluster ball — uniform-ish
            // distribution by combining three independent hash bytes.
            const j1 = ((h >>> 3) & 0xff) / 255 - 0.5
            const j2 = ((h >>> 11) & 0xff) / 255 - 0.5
            const j3 = ((h >>> 19) & 0xff) / 255 - 0.5
            // Slight radial bias toward center of cluster for tighter grouping.
            const r = ((h >>> 27) & 0x1f) / 31  // 0..1
            const radial = Math.cbrt(r)         // cube-root → ball-uniform
            const len = Math.hypot(j1, j2, j3) || 1
            const pos = anchor.clone()
            pos.x += (j1 / len) * radial * CLUSTER_RADIUS
            pos.y += (j2 / len) * radial * CLUSTER_RADIUS
            pos.z += (j3 / len) * radial * CLUSTER_RADIUS

            // Hard clamp to the cortex shell so jittered anchors never escape.
            const offset = pos.clone().sub(brainCenter)
            const d = offset.length()
            if (d > HARD_BOUND) {
              offset.multiplyScalar(HARD_BOUND / d)
              pos.copy(brainCenter).add(offset)
            }

            nodePositions.push(pos)
          })

          // Shared unit plane — actual world size set via scale in animate loop
          const planeGeom = new PlaneGeometry(1, 1)

          // ── Layer 1: Solid colored disc (NormalBlending = no stacking) ──
          // Switching from Additive to Normal kills the central white-supernova
          // when many nodes overlap — each disc paints its domain color cleanly
          // instead of summing toward saturation.
          const coreMat = new MeshBasicMaterial({
            map: glowTex,
            color: 0xffffff,
            transparent: true,
            opacity: 0.95,
            blending: NormalBlending,
            depthWrite: false,
            toneMapped: false,
          })
          markersMesh = new InstancedMesh(planeGeom, coreMat, nodes.length)
          markersMesh.renderOrder = 3

          // ── Layer 2: Mid soft halo (additive but very low opacity) ──
          const haloMat = new MeshBasicMaterial({
            map: glowTex,
            color: 0xffffff,
            transparent: true,
            opacity: 0.10,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          haloMesh = new InstancedMesh(planeGeom, haloMat, nodes.length)
          haloMesh.renderOrder = 2

          // ── Layer 3: Outer feathered bloom (barely there) ──
          const bloomMat = new MeshBasicMaterial({
            map: glowTex,
            color: 0xffffff,
            transparent: true,
            opacity: 0.04,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          bloomMesh = new InstancedMesh(planeGeom, bloomMat, nodes.length)
          bloomMesh.renderOrder = 1

          // Bootstrap per-instance colors (by domain) and initial matrices
          for (let i = 0; i < nodes.length; i++) {
            const core = new Color(domainColor(nodes[i].domain))
            nodeCoreColors.push(core)
            markersMesh.setColorAt(i, core)
            haloMesh.setColorAt(i, core)
            bloomMesh.setColorAt(i, core)

            // Initial matrix (billboard orientation set correctly on first animate frame)
            tmpObj.position.copy(nodePositions[i])
            tmpObj.scale.setScalar(MARKER_BASE_SCALE)
            tmpObj.updateMatrix()
            markersMesh.setMatrixAt(i, tmpObj.matrix)
            tmpObj.scale.setScalar(MARKER_BASE_SCALE * 1.6)
            tmpObj.updateMatrix()
            haloMesh.setMatrixAt(i, tmpObj.matrix)
            tmpObj.scale.setScalar(MARKER_BASE_SCALE * 2.4)
            tmpObj.updateMatrix()
            bloomMesh.setMatrixAt(i, tmpObj.matrix)
          }

          markersMesh.instanceMatrix.needsUpdate = true
          haloMesh.instanceMatrix.needsUpdate = true
          bloomMesh.instanceMatrix.needsUpdate = true
          if (markersMesh.instanceColor) markersMesh.instanceColor.needsUpdate = true
          if (haloMesh.instanceColor) haloMesh.instanceColor.needsUpdate = true
          if (bloomMesh.instanceColor) bloomMesh.instanceColor.needsUpdate = true

          // Create clip group for nodes/edges
          clipGroup = new Object3D()
          scene.add(clipGroup)

          // Add bloom first (back), then halo, then core (front) to clip group
          if (bloomMesh) clipGroup.add(bloomMesh)
          if (haloMesh) clipGroup.add(haloMesh)
          clipGroup.add(markersMesh)

          if (selectedRef.current) applySelection(selectedRef.current)

          // Edges between memories
          if (showEdges && links.length > 0) {
            const verts: number[] = []
            const cols: number[] = []
            for (const l of links) {
              const sId = typeof l.source === "string" ? l.source : (l.source as any).id
              const tId = typeof l.target === "string" ? l.target : (l.target as any).id
              const si = nodeIndexById.get(sId)
              const ti = nodeIndexById.get(tId)
              if (si === undefined || ti === undefined) continue
              const sp = nodePositions[si]
              const tp = nodePositions[ti]
              verts.push(sp.x, sp.y, sp.z, tp.x, tp.y, tp.z)
              const sc = nodeCoreColors[si]
              const tc = nodeCoreColors[ti]
              const a = MathUtils.clamp(l.weight ?? 0.3, 0.1, 0.6)
              cols.push(sc.r * a, sc.g * a, sc.b * a, tc.r * a, tc.g * a, tc.b * a)
            }
            if (verts.length > 0) {
              const eg = new BufferGeometry()
              eg.setAttribute("position", new Float32BufferAttribute(verts, 3))
              eg.setAttribute("color", new Float32BufferAttribute(cols, 3))
              const em = new LineBasicMaterial({
                vertexColors: true,
                transparent: true,
                opacity: 0.3,
                depthWrite: false,
              })
              edgesLine = new LineSegments(eg, em)
              edgesLine.renderOrder = 0
              clipGroup.add(edgesLine)
            }
          }
        })
        .catch(() => {})

      window.addEventListener("mousemove", onMousemove, { passive: true })
      window.addEventListener("resize", onResize, { passive: true })
      container.addEventListener("pointerdown", onPointerDown)
      container.addEventListener("click", onClick)
      animate()
    })

    return () => {
      disposed = true
      cancelAnimationFrame(frameId)
      sizeObserver?.disconnect()
      sizeObserver = null
      window.removeEventListener("mousemove", onMousemove)
      window.removeEventListener("resize", onResize)
      container.removeEventListener("pointerdown", onPointerDown)
      container.removeEventListener("click", onClick)
      controls.dispose()
      glowTex.dispose()
      if (brainWireframe) {
        brainWireframe.geometry.dispose()
        ;(brainWireframe.material as LineBasicMaterial).dispose()
        scene.remove(brainWireframe)
      }
      if (brainHullInner) {
        ;(brainHullInner.material as MeshBasicMaterial).dispose()
        scene.remove(brainHullInner)
      }
      if (brainHullOuter) {
        ;(brainHullOuter.material as MeshBasicMaterial).dispose()
        scene.remove(brainHullOuter)
      }
      if (brainDots) {
        brainDots.geometry.dispose()
        ;(brainDots.material as PointsMaterial).dispose()
        scene.remove(brainDots)
      }
      if (stars) {
        stars.geometry.dispose()
        ;(stars.material as PointsMaterial).dispose()
        scene.remove(stars)
      }
      if (nebula) {
        nebula.geometry.dispose()
        ;(nebula.material as PointsMaterial).dispose()
        scene.remove(nebula)
      }
      if (stars) {
        stars.geometry.dispose()
        ;(stars.material as PointsMaterial).dispose()
        scene.remove(stars)
      }
      if (markersMesh) {
        markersMesh.geometry.dispose()
        ;(markersMesh.material as MeshBasicMaterial).dispose()
        scene.remove(markersMesh)
      }
      if (haloMesh) {
        haloMesh.geometry.dispose()
        ;(haloMesh.material as MeshBasicMaterial).dispose()
        scene.remove(haloMesh)
      }
      if (bloomMesh) {
        bloomMesh.geometry.dispose()
        ;(bloomMesh.material as MeshBasicMaterial).dispose()
        scene.remove(bloomMesh)
      }
      if (edgesLine) {
        edgesLine.geometry.dispose()
        ;(edgesLine.material as LineBasicMaterial).dispose()
        scene.remove(edgesLine)
      }
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [modelUrl, showEdges])

  return <div {...props} ref={containerRef} />
}
