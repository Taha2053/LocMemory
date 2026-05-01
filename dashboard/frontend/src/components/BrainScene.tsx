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
  BoxGeometry,
  BufferGeometry,
  InstancedMesh,
  MeshBasicMaterial,
  AdditiveBlending,
  LineSegments,
  LineBasicMaterial,
  Float32BufferAttribute,
  WireframeGeometry,
  Points,
  PointsMaterial,
} from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import gsap from "gsap"
import { api } from "@/lib/api"

// Data-viz tier palette — green / orange / amber / rose
const TIER_PALETTE = [
  new Color(0x00ff88), // Core Context — bright emerald
  new Color(0xff8c26), // Anchor Memories — warm orange
  new Color(0xffd700), // Leaf Memories — amber gold
  new Color(0xff4d6d), // Procedural — rose red
]

function hashId(str: string): number {
  let h = 0xdeadbeef
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 2654435761)
  }
  return (h ^ (h >>> 16)) >>> 0
}

const MARKER_BASE_SCALE = 1.0
const ZOOM_DISTANCE = 0.35
const DEFAULT_CAM_Z = 1.9
const DEFAULT_CAM_Z_MOBILE = 2.9

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

    const size = {
      width: container.clientWidth || 800,
      height: container.clientHeight || 600,
    }

    const scene = new Scene()
    const camera = new PerspectiveCamera(75, size.width / size.height, 0.1, 100)
    const isMobile = window.innerWidth < 767
    const startZ = isMobile ? DEFAULT_CAM_Z_MOBILE : DEFAULT_CAM_Z
    camera.position.set(0, 0, startZ)

    const renderer = new WebGLRenderer({
      alpha: true,
      antialias: window.devicePixelRatio === 1,
    })
    renderer.setSize(size.width, size.height)
    renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio))
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.enablePan = false
    controls.minDistance = 0.6
    controls.maxDistance = 5
    controls.rotateSpeed = 0.6
    controls.zoomSpeed = 0.7
    controls.target.set(0, 0, 0)

    const raycaster = new Raycaster()
    let frameId = 0
    let disposed = false

    // Brain visual refs
    let brainWireframe: LineSegments | null = null
    let brainDots: Points | null = null

    // Memory overlay state
    let markersMesh: InstancedMesh | null = null
    let haloMesh: InstancedMesh | null = null
    let edgesLine: LineSegments | null = null
    let nodeIds: string[] = []
    const nodeIndexById = new Map<string, number>()
    const nodePositions: Vector3[] = []
    const nodeTiers: number[] = []
    let hoveredIdx = -1
    let prevHoveredIdx = -1

    const tmpObj = new Object3D()

    const setMarkerMatrix = (idx: number, scale: number) => {
      if (!markersMesh) return
      tmpObj.position.copy(nodePositions[idx])
      tmpObj.scale.setScalar(scale)
      tmpObj.updateMatrix()
      markersMesh.setMatrixAt(idx, tmpObj.matrix)
    }

    const setHaloMatrix = (idx: number, scale: number) => {
      if (!haloMesh) return
      tmpObj.position.copy(nodePositions[idx])
      tmpObj.scale.setScalar(scale)
      tmpObj.updateMatrix()
      haloMesh.setMatrixAt(idx, tmpObj.matrix)
    }

    let prevSelectedIdx = -1
    const applySelection = (id: string | null) => {
      const newIdx = id ? nodeIndexById.get(id) ?? -1 : -1

      if (markersMesh && haloMesh) {
        if (prevSelectedIdx >= 0 && prevSelectedIdx !== newIdx) {
          setMarkerMatrix(prevSelectedIdx, MARKER_BASE_SCALE)
          setHaloMatrix(prevSelectedIdx, 1.0)
          const tIdx = Math.max(0, Math.min(3, nodeTiers[prevSelectedIdx] - 1))
          markersMesh.setColorAt(prevSelectedIdx, TIER_PALETTE[tIdx])
        }

        if (newIdx >= 0) {
          const tIdx = Math.max(0, Math.min(3, nodeTiers[newIdx] - 1))
          markersMesh.setColorAt(
            newIdx,
            TIER_PALETTE[tIdx].clone().lerp(new Color(0xffffff), 0.5),
          )
          setMarkerMatrix(newIdx, MARKER_BASE_SCALE * 1.35)
          setHaloMatrix(newIdx, 1.6)
        }

        markersMesh.instanceMatrix.needsUpdate = true
        haloMesh.instanceMatrix.needsUpdate = true
        if (markersMesh.instanceColor) markersMesh.instanceColor.needsUpdate = true

        prevSelectedIdx = newIdx
      }

      // Dim brain when a memory is focused
      const focused = newIdx >= 0
      if (brainWireframe) {
        gsap.to(brainWireframe.material as LineBasicMaterial, {
          opacity: focused ? 0.06 : 0.2,
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
      size.width = container.clientWidth || 800
      size.height = container.clientHeight || 600
      camera.aspect = size.width / size.height
      camera.updateProjectionMatrix()
      renderer.setSize(size.width, size.height)
    }

    const animate = () => {
      if (disposed) return
      controls.update()

      // Slow Y rotation
      if (brainWireframe) brainWireframe.rotation.y += 0.0008
      if (brainDots) brainDots.rotation.y += 0.0008

      // Hover scale on markers
      if (markersMesh && nodeIds.length > 0 && hoveredIdx !== prevHoveredIdx) {
        if (prevHoveredIdx >= 0 && prevHoveredIdx !== prevSelectedIdx) {
          setMarkerMatrix(prevHoveredIdx, MARKER_BASE_SCALE)
        }
        if (hoveredIdx >= 0 && hoveredIdx !== prevSelectedIdx) {
          setMarkerMatrix(hoveredIdx, MARKER_BASE_SCALE * 1.5)
        }
        markersMesh.instanceMatrix.needsUpdate = true
        prevHoveredIdx = hoveredIdx
      }

      renderer.render(scene, camera)
      frameId = requestAnimationFrame(animate)
    }

    const loader = new GLTFLoader()

    loader.load(modelUrl, (gltf) => {
      if (disposed) return
      const brainMesh = gltf.scene.children[0] as Mesh

      // ── Wireframe brain mesh ──
      const wireGeo = new WireframeGeometry(brainMesh.geometry)
      const wireMat = new LineBasicMaterial({
        color: new Color(0x00ff88),
        transparent: true,
        opacity: 0.2,
        depthWrite: false,
      })
      brainWireframe = new LineSegments(wireGeo, wireMat)
      scene.add(brainWireframe)

      // ── Glowing node dots at each vertex ──
      const dotsGeo = new BufferGeometry()
      dotsGeo.setAttribute("position", brainMesh.geometry.getAttribute("position").clone())
      const dotsMat = new PointsMaterial({
        color: new Color(0x00ff88),
        size: 0.012,
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
      const stars = new Points(starGeo, starMat)
      scene.add(stars)

      // ── Load real memories overlaid on the brain ──
      api
        .graph()
        .then(({ nodes, links }) => {
          if (disposed || nodes.length === 0) return
          const positionsArr = brainMesh.geometry.attributes.position.array
          const positionCount = positionsArr.length / 3

          nodeIds = nodes.map((n) => n.id)
          nodes.forEach((n, i) => {
            nodeIndexById.set(n.id, i)
            nodeTiers.push(n.tier)
            const vIdx = hashId(n.id) % positionCount
            nodePositions.push(
              new Vector3(
                positionsArr[vIdx * 3],
                positionsArr[vIdx * 3 + 1],
                positionsArr[vIdx * 3 + 2],
              ),
            )
          })

          // Memory markers — wireframe cubes
          const markerGeom = new WireframeGeometry(new BoxGeometry(0.045, 0.045, 0.045))
          const markerMat = new MeshBasicMaterial({
            wireframe: true,
            transparent: true,
            opacity: 1.0,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          markersMesh = new InstancedMesh(markerGeom, markerMat, nodes.length)
          markersMesh.renderOrder = 2

          for (let i = 0; i < nodes.length; i++) {
            setMarkerMatrix(i, MARKER_BASE_SCALE)
            const tierIdx = Math.max(0, Math.min(3, nodes[i].tier - 1))
            markersMesh.setColorAt(i, TIER_PALETTE[tierIdx])
          }
          markersMesh.instanceMatrix.needsUpdate = true
          if (markersMesh.instanceColor) markersMesh.instanceColor.needsUpdate = true
          scene.add(markersMesh)

          // Halos — soft glow zones
          const haloGeom = new WireframeGeometry(new BoxGeometry(0.085, 0.085, 0.085))
          const haloMat = new MeshBasicMaterial({
            wireframe: true,
            transparent: true,
            opacity: 0.4,
            blending: AdditiveBlending,
            depthWrite: false,
            toneMapped: false,
          })
          haloMesh = new InstancedMesh(haloGeom, haloMat, nodes.length)
          haloMesh.renderOrder = 1.5

          for (let i = 0; i < nodes.length; i++) {
            setHaloMatrix(i, 1.0)
            const tIdx = Math.max(0, Math.min(3, nodes[i].tier - 1))
            const halo = TIER_PALETTE[tIdx].clone().lerp(new Color(0xffffff), 0.4)
            haloMesh.setColorAt(i, halo)
          }
          haloMesh.instanceMatrix.needsUpdate = true
          if (haloMesh.instanceColor) haloMesh.instanceColor.needsUpdate = true
          scene.add(haloMesh)

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
              scene.add(edgesLine)
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
      window.removeEventListener("mousemove", onMousemove)
      window.removeEventListener("resize", onResize)
      container.removeEventListener("pointerdown", onPointerDown)
      container.removeEventListener("click", onClick)
      controls.dispose()
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
