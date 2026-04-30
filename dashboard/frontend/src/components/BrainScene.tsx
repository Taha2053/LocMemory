import { useEffect, useRef } from "react"
import {
  Scene,
  WebGLRenderer,
  PerspectiveCamera,
  BoxGeometry,
  ShaderMaterial,
  Color,
  Vector2,
  Vector3,
  Raycaster,
  Object3D,
  MathUtils,
  LoadingManager,
  Mesh,
  BufferGeometry,
  InstancedMesh,
  MeshBasicMaterial,
  AdditiveBlending,
  LineSegments,
  LineBasicMaterial,
  Float32BufferAttribute,
} from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { InstancedUniformsMesh } from "three-instanced-uniforms-mesh"
import gsap from "gsap"
import { api } from "@/lib/api"

const VERTEX_SHADER = /* glsl */ `
  uniform vec3 uPointer;
  uniform vec3 uColor;
  uniform float uRotation;
  uniform float uSize;
  uniform float uHover;
  uniform float uSizeMul;

  varying vec3 vColor;

  #define PI 3.14159265359

  mat2 rotate(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
  }

  void main() {
    vec4 mvPosition = vec4(position, 1.0);
    mvPosition = instanceMatrix * mvPosition;

    float d = distance(uPointer, mvPosition.xyz);
    float c = smoothstep(0.45, 0.1, d);

    float scale = (uSize + c * 8.0 * uHover) * uSizeMul;
    vec3 pos = position;
    pos *= scale;
    pos.xz *= rotate(PI * c * uRotation + PI * uRotation * 0.43);
    pos.xy *= rotate(PI * c * uRotation + PI * uRotation * 0.71);

    mvPosition = instanceMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * modelViewMatrix * mvPosition;

    vColor = uColor;
  }
`

const FRAGMENT_SHADER = /* glsl */ `
  uniform float uDim;
  varying vec3 vColor;
  void main() {
    // brighten brain voxels — clamp to 1.0, then dim when a memory is focused
    gl_FragColor = vec4(min(vColor * 1.25, vec3(1.0)) * uDim, 1.0);
  }
`

// LocMemory tier palette — used for memory markers + edges
const TIER_PALETTE = [
  new Color(0x3b82f6), // Core Context — electric blue
  new Color(0x06b6d4), // Anchor Memories — cyan
  new Color(0x9ec5e8), // Leaf Memories — soft white-blue
  new Color(0xa855f7), // Procedural Memories — purple
]

// Lighter pastel palette — used only for the brain voxels so the brain reads brighter
const BRAIN_PALETTE = [
  new Color(0x9ec5e8),
  new Color(0x7dd3fc),
  new Color(0xc4b5fd),
  new Color(0xddd6fe),
  new Color(0xe0f2fe),
]

// Stable string hash so a node id always maps to the same brain vertex
function hashId(str: string): number {
  let h = 0xdeadbeef
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 2654435761)
  }
  return (h ^ (h >>> 16)) >>> 0
}

const MARKER_BASE_SCALE = 1.0
const ZOOM_DISTANCE = 0.35
const DEFAULT_CAM_Z = 1.2
const DEFAULT_CAM_Z_MOBILE = 2.3

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

    // ── Orbit controls — free rotation up/down/left/right + zoom ──
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.enablePan = false
    controls.minDistance = 0.4
    controls.maxDistance = 4
    controls.rotateSpeed = 0.6
    controls.zoomSpeed = 0.7
    controls.target.set(0, 0, 0)

    const raycaster = new Raycaster()
    const mouse = new Vector2()
    const point = new Vector3()
    const uniforms = { uHover: 0 }
    let hover = false
    let brainMesh: Mesh | null = null
    let instancedMesh: InstancedUniformsMesh | null = null
    let brainMaterial: ShaderMaterial | null = null
    let frameId = 0
    let disposed = false

    // ── memory-overlay state ───────────────────────────
    let markersMesh: InstancedMesh | null = null
    let haloMesh: InstancedMesh | null = null
    let edgesLine: LineSegments | null = null
    let nodeIds: string[] = []
    const nodeIndexById = new Map<string, number>()
    const nodePositions: Vector3[] = []
    const nodeTiers: number[] = []
    let hoveredIdx = -1
    let prevHoveredIdx = -1

    const animateHoverUniform = (value: number) => {
      gsap.to(uniforms, {
        uHover: value,
        duration: 0.25,
        onUpdate: () => {
          if (!instancedMesh) return
          for (let i = 0; i < instancedMesh.count; i++) {
            instancedMesh.setUniformAt("uHover", i, uniforms.uHover)
          }
        },
      })
    }

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

    // ── selection-driven zoom + static highlight (no per-frame vibration) ──
    let prevSelectedIdx = -1
    const applySelection = (id: string | null) => {
      const newIdx = id ? nodeIndexById.get(id) ?? -1 : -1

      if (markersMesh && haloMesh) {
        // restore previous
        if (prevSelectedIdx >= 0 && prevSelectedIdx !== newIdx) {
          setMarkerMatrix(prevSelectedIdx, MARKER_BASE_SCALE)
          setHaloMatrix(prevSelectedIdx, 1.0)
          const tIdx = Math.max(0, Math.min(3, nodeTiers[prevSelectedIdx] - 1))
          markersMesh.setColorAt(prevSelectedIdx, TIER_PALETTE[tIdx])
        }

        if (newIdx >= 0) {
          const tIdx = Math.max(0, Math.min(3, nodeTiers[newIdx] - 1))
          // brighter color, larger halo, larger marker — all static
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

      // brain fade + shrink so the selected memory becomes the focal point
      const focused = newIdx >= 0
      if (brainMaterial) {
        gsap.to(brainMaterial.uniforms.uDim, {
          value: focused ? 0.18 : 1.0,
          duration: 0.7,
          ease: "power2.out",
        })
        gsap.to(brainMaterial.uniforms.uSizeMul, {
          value: focused ? 0.4 : 1.0,
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

      // camera fly
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
      if (!brainMesh) return
      const rect = container.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / size.width) * 2 - 1
      const y = -((e.clientY - rect.top) / size.height) * 2 + 1
      mouse.set(x, y)

      raycaster.setFromCamera(mouse, camera)

      // Marker picking
      if (markersMesh) {
        const markerHits = raycaster.intersectObject(markersMesh, false)
        if (markerHits.length > 0 && markerHits[0].instanceId !== undefined) {
          hoveredIdx = markerHits[0].instanceId
          container.style.cursor = "pointer"
        } else {
          hoveredIdx = -1
          container.style.cursor = ""
        }
      }

      const intersects = raycaster.intersectObject(brainMesh)
      if (intersects.length === 0) {
        if (hover) { hover = false; animateHoverUniform(0) }
      } else {
        if (!hover) { hover = true; animateHoverUniform(1) }
        gsap.to(point, {
          x: intersects[0].point.x,
          y: intersects[0].point.y,
          z: intersects[0].point.z,
          overwrite: true,
          duration: 0.3,
          onUpdate: () => {
            if (!instancedMesh) return
            for (let i = 0; i < instancedMesh.count; i++) {
              instancedMesh.setUniformAt("uPointer", i, point)
            }
          },
        })
      }
    }

    let downX = 0, downY = 0
    const onPointerDown = (e: PointerEvent) => { downX = e.clientX; downY = e.clientY }
    const onClick = (e: MouseEvent) => {
      // only treat as click if pointer barely moved (avoid firing during orbit drag)
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

      // hover scale on markers (only when not the selected one)
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

    const loadingManager = new LoadingManager()
    const loader = new GLTFLoader(loadingManager)

    loader.load(modelUrl, (gltf) => {
      if (disposed) return
      brainMesh = gltf.scene.children[0] as Mesh

      const geometry = new BoxGeometry(0.004, 0.004, 0.004, 1, 1, 1)
      const material = new ShaderMaterial({
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        wireframe: true,
        uniforms: {
          uPointer: { value: new Vector3() },
          uColor: { value: new Color() },
          uRotation: { value: 0 },
          uSize: { value: 0 },
          uHover: { value: uniforms.uHover },
          uDim: { value: 1.0 },
          uSizeMul: { value: 1.0 },
        },
      })
      brainMaterial = material

      if (brainMesh.geometry instanceof BufferGeometry) {
        const count = brainMesh.geometry.attributes.position.count
        instancedMesh = new InstancedUniformsMesh(geometry, material, count)
        scene.add(instancedMesh)

        const dummy = new Object3D()
        const positions = brainMesh.geometry.attributes.position.array
        for (let i = 0; i < positions.length; i += 3) {
          const idx = i / 3
          dummy.position.set(positions[i], positions[i + 1], positions[i + 2])
          dummy.updateMatrix()
          instancedMesh.setMatrixAt(idx, dummy.matrix)
          instancedMesh.setUniformAt("uRotation", idx, MathUtils.randFloat(-1, 1))
          instancedMesh.setUniformAt("uSize", idx, MathUtils.randFloat(0.3, 3))
          const colorIndex = MathUtils.randInt(0, BRAIN_PALETTE.length - 1)
          instancedMesh.setUniformAt("uColor", idx, BRAIN_PALETTE[colorIndex])
        }

        // ── load real memories overlaid on the brain ────────────
        api
          .graph()
          .then(({ nodes, links }) => {
            if (disposed || !brainMesh || nodes.length === 0) return
            const positionsArr = (brainMesh.geometry as BufferGeometry)
              .attributes.position.array
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

            // Memory markers — bigger, luminous wireframe cubes
            const markerGeom = new BoxGeometry(0.045, 0.045, 0.045, 1, 1, 1)
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

            // Halo around each memory — soft glow zone
            const haloGeom = new BoxGeometry(0.085, 0.085, 0.085, 1, 1, 1)
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

            // re-apply current selection now that meshes exist
            if (selectedRef.current) applySelection(selectedRef.current)

            // Edges
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
      }

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
