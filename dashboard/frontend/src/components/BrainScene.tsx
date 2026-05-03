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
  LineSegments,
  LineBasicMaterial,
  Float32BufferAttribute,
  WireframeGeometry,
  PlaneGeometry,
  CanvasTexture,
  Points,
  PointsMaterial,
} from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import gsap from "gsap"
import { api } from "@/lib/api"

// Bioluminescent tier palette — emerald / cyan / lime / white-green
const TIER_PALETTE = [
  new Color(0x00ff88), // Core Context — deep emerald green
  new Color(0x00e5ff), // Anchor Memories — teal-cyan
  new Color(0xaaff00), // Leaf Memories — soft lime
  new Color(0xffffff), // Procedural — bright green-white
]

const TIER_HALO_PALETTE = [
  new Color(0x00cc66), // Core Context halo
  new Color(0x0099bb), // Anchor halo
  new Color(0x66cc00), // Leaf halo
  new Color(0x00ff66), // Procedural halo
]

function createGlowTexture(): CanvasTexture {
  const size = 128
  const canvas = document.createElement("canvas")
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext("2d")!
  const half = size / 2
  // Layered radial gradient: bright core → soft mid → feathered nebula fade
  const gradient = ctx.createRadialGradient(half, half, 0, half, half, half)
  gradient.addColorStop(0.00, "rgba(255,255,255,1.0)")
  gradient.addColorStop(0.12, "rgba(255,255,255,0.92)")
  gradient.addColorStop(0.30, "rgba(255,255,255,0.55)")
  gradient.addColorStop(0.55, "rgba(255,255,255,0.18)")
  gradient.addColorStop(0.78, "rgba(255,255,255,0.05)")
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
const MARKER_BASE_SCALE = 0.05
const ZOOM_DISTANCE = 0.35
// Scale nodes inward so they sit inside the brain boundary
const INWARD_SCALE = 0.78
const DEFAULT_CAM_Z = 1.9
const DEFAULT_CAM_Z_MOBILE = 2.9

// Clamp position to stay within brain bounding sphere
function clampToBrainSphere(pos: Vector3, center: Vector3, radius: number): Vector3 {
  const scaledRadius = radius * INWARD_SCALE
  const toPos = pos.clone().sub(center)
  const dist = toPos.length()
  if (dist > scaledRadius) {
    toPos.normalize().multiplyScalar(scaledRadius)
    return center.clone().add(toPos)
  }
  return pos.clone()
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
    let brainDots: Points | null = null
    let stars: Points | null = null
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
    // Per-node base scale — hover/selection overrides MARKER_BASE_SCALE
    const baseScales: number[] = []
    let hoveredIdx = -1
    let prevHoveredIdx = -1

    const tmpObj = new Object3D()

    let prevSelectedIdx = -1
    const applySelection = (id: string | null) => {
      const newIdx = id ? nodeIndexById.get(id) ?? -1 : -1

      if (markersMesh) {
        // Reset previous selection
        if (prevSelectedIdx >= 0 && prevSelectedIdx !== newIdx) {
          baseScales[prevSelectedIdx] = MARKER_BASE_SCALE
          const tIdx = Math.max(0, Math.min(3, nodeTiers[prevSelectedIdx] - 1))
          markersMesh.setColorAt(prevSelectedIdx, TIER_PALETTE[tIdx])
          if (haloMesh?.instanceColor) haloMesh.setColorAt(prevSelectedIdx, TIER_HALO_PALETTE[tIdx])
          if (bloomMesh?.instanceColor) bloomMesh.setColorAt(prevSelectedIdx, TIER_HALO_PALETTE[tIdx])
        }

        // Apply new selection — brighter core, larger scale
        if (newIdx >= 0) {
          const tIdx = Math.max(0, Math.min(3, nodeTiers[newIdx] - 1))
          markersMesh.setColorAt(
            newIdx,
            TIER_PALETTE[tIdx].clone().lerp(new Color(0xffffff), 0.5),
          )
          if (haloMesh?.instanceColor) haloMesh.setColorAt(newIdx, TIER_PALETTE[tIdx])
          if (bloomMesh?.instanceColor) bloomMesh.setColorAt(newIdx, TIER_PALETTE[tIdx])
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
          opacity: focused ? 0.06 : 0.15,
          duration: 0.7,
          ease: "power2.out",
        })
      }
      if (brainDots) {
        gsap.to(brainDots.material as PointsMaterial, {
          opacity: focused ? 0.15 : 0.85,
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

      // Slow Y rotation
      if (brainWireframe) brainWireframe.rotation.y += 0.0008
      if (brainDots) brainDots.rotation.y += 0.0008
      if (stars) {
        stars.rotation.y -= 0.0002
        stars.rotation.x += 0.0001
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
            tmpObj.scale.setScalar(s * 2.5)
            tmpObj.updateMatrix()
            haloMesh.setMatrixAt(i, tmpObj.matrix)
          }

          // Outer feathered bloom (6× larger)
          if (bloomMesh) {
            tmpObj.scale.setScalar(s * 6.0)
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

      // ── Wireframe brain mesh (bioluminescent emerald green) ──
      const wireGeo = new WireframeGeometry(brainMesh.geometry)
      const wireMat = new LineBasicMaterial({
        color: new Color(0x00ff88),
        transparent: true,
        opacity: 0.15,
        depthWrite: false,
      })
      brainWireframe = new LineSegments(wireGeo, wireMat)
      scene.add(brainWireframe)

      // ── Glowing node dots at each vertex ──
      const dotsGeo = new BufferGeometry()
      dotsGeo.setAttribute("position", brainMesh.geometry.getAttribute("position").clone())
      const dotsMat = new PointsMaterial({
        color: new Color(0x00ff88),
        size: 0.010,
        transparent: true,
        opacity: 0.85,
        blending: AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      })
      brainDots = new Points(dotsGeo, dotsMat)
      scene.add(brainDots)

      // ── Star field background ──
      const starVerts: number[] = []
      for (let i = 0; i < 350; i++) {
        const theta = Math.random() * Math.PI * 2
        const phi = Math.acos(2 * Math.random() - 1)
        const r = 5 + Math.random() * 4
        starVerts.push(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        )
      }
      const starGeo = new BufferGeometry()
      starGeo.setAttribute("position", new Float32BufferAttribute(starVerts, 3))
      const starMat = new PointsMaterial({
        color: 0xffffff,
        size: 0.02,
        transparent: true,
        opacity: 0.45,
        depthWrite: false,
        sizeAttenuation: true,
      })
      stars = new Points(starGeo, starMat)
      scene.add(stars)

      // ── Load real memories overlaid on the brain ──
      api
        .graph()
        .then(({ nodes, links }) => {
          if (disposed || nodes.length === 0) return
          const positionsArr = brainMesh.geometry.attributes.position.array
          const positionCount = positionsArr.length / 3

          // Compute bounding sphere from brain mesh for containment
          const posArray = positionsArr as unknown as { length: number; [i: number]: number }
          let minX = Infinity, maxX = -Infinity
          let minY = Infinity, maxY = -Infinity
          let minZ = Infinity, maxZ = -Infinity
          for (let i = 0; i < positionCount; i++) {
            const x = posArray[i * 3], y = posArray[i * 3 + 1], z = posArray[i * 3 + 2]
            minX = Math.min(minX, x); maxX = Math.max(maxX, x)
            minY = Math.min(minY, y); maxY = Math.max(maxY, y)
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
          }
          const brainCenter = new Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2)
          const brainRadius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2

          nodeIds = nodes.map((n) => n.id)
          nodes.forEach((n, i) => {
            nodeIndexById.set(n.id, i)
            nodeTiers.push(n.tier)
            baseScales.push(MARKER_BASE_SCALE)
            const vIdx = hashId(n.id) % positionCount
            const rawPos = new Vector3(
              positionsArr[vIdx * 3],
              positionsArr[vIdx * 3 + 1],
              positionsArr[vIdx * 3 + 2],
            )
            // Clamp to brain sphere to ensure nodes stay inside boundary
            const clampedPos = clampToBrainSphere(rawPos, brainCenter, brainRadius)
            nodePositions.push(clampedPos)
          })

          // Shared unit plane — actual world size set via scale in animate loop
          const planeGeom = new PlaneGeometry(1, 1)

          // ── Layer 1: Inner bright core glow (hit-tested for raycasting) ──
          const coreMat = new MeshBasicMaterial({
            map: glowTex,
            color: 0xffffff,
            transparent: true,
            opacity: 1.0,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          markersMesh = new InstancedMesh(planeGeom, coreMat, nodes.length)
          markersMesh.renderOrder = 3

          // ── Layer 2: Mid soft halo ──
          const haloMat = new MeshBasicMaterial({
            map: glowTex,
            color: 0xffffff,
            transparent: true,
            opacity: 0.55,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          haloMesh = new InstancedMesh(planeGeom, haloMat, nodes.length)
          haloMesh.renderOrder = 2

          // ── Layer 3: Outer feathered bloom ──
          const bloomMat = new MeshBasicMaterial({
            map: glowTex,
            color: 0xffffff,
            transparent: true,
            opacity: 0.22,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          bloomMesh = new InstancedMesh(planeGeom, bloomMat, nodes.length)
          bloomMesh.renderOrder = 1

          // Bootstrap per-instance colors and initial matrices
          for (let i = 0; i < nodes.length; i++) {
            const tierIdx = Math.max(0, Math.min(3, nodes[i].tier - 1))
            markersMesh.setColorAt(i, TIER_PALETTE[tierIdx])
            haloMesh.setColorAt(i, TIER_HALO_PALETTE[tierIdx])
            bloomMesh.setColorAt(i, TIER_HALO_PALETTE[tierIdx])

            // Initial matrix (billboard orientation set correctly on first animate frame)
            tmpObj.position.copy(nodePositions[i])
            tmpObj.scale.setScalar(MARKER_BASE_SCALE)
            tmpObj.updateMatrix()
            markersMesh.setMatrixAt(i, tmpObj.matrix)
            tmpObj.scale.setScalar(MARKER_BASE_SCALE * 2.5)
            tmpObj.updateMatrix()
            haloMesh.setMatrixAt(i, tmpObj.matrix)
            tmpObj.scale.setScalar(MARKER_BASE_SCALE * 6.0)
            tmpObj.updateMatrix()
            bloomMesh.setMatrixAt(i, tmpObj.matrix)
          }

          markersMesh.instanceMatrix.needsUpdate = true
          haloMesh.instanceMatrix.needsUpdate = true
          bloomMesh.instanceMatrix.needsUpdate = true
          if (markersMesh.instanceColor) markersMesh.instanceColor.needsUpdate = true
          if (haloMesh.instanceColor) haloMesh.instanceColor.needsUpdate = true
          if (bloomMesh.instanceColor) bloomMesh.instanceColor.needsUpdate = true

          // Create clip group for nodes/edges (scaled inward via INWARD_SCALE)
          clipGroup = new Object3D()
          scene.add(clipGroup)

          // Add bloom first (back), then halo, then core (front) to clip group
          clipGroup.add(bloomMesh)
          clipGroup.add(haloMesh)
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
              const sc = TIER_PALETTE[Math.max(0, Math.min(3, nodes[si].tier - 1))]
              const tc = TIER_PALETTE[Math.max(0, Math.min(3, nodes[ti].tier - 1))]
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
                opacity: 0.5,
                depthWrite: false,
              })
              edgesLine = new LineSegments(eg, em)
              edgesLine.renderOrder = 1
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
      if (brainDots) {
        brainDots.geometry.dispose()
        ;(brainDots.material as PointsMaterial).dispose()
        scene.remove(brainDots)
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
