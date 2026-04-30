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
} from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import { InstancedUniformsMesh } from "three-instanced-uniforms-mesh"
import gsap from "gsap"

const VERTEX_SHADER = /* glsl */ `
  uniform vec3 uPointer;
  uniform vec3 uColor;
  uniform float uRotation;
  uniform float uSize;
  uniform float uHover;

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

    float scale = uSize + c * 8.0 * uHover;
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
  varying vec3 vColor;
  void main() {
    gl_FragColor = vec4(vColor, 1.0);
  }
`

// LocMemory tier palette
const TIER_PALETTE = [
  new Color(0x3b82f6), // Core Context — electric blue
  new Color(0x06b6d4), // Anchor Memories — cyan
  new Color(0x9ec5e8), // Leaf Memories — soft white-blue
  new Color(0xa855f7), // Procedural Memories — purple
]

interface BrainSceneProps extends React.HTMLAttributes<HTMLDivElement> {
  modelUrl?: string
}

export function BrainScene({ modelUrl = "/brain.glb", ...props }: BrainSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const size = {
      width: container.clientWidth || 800,
      height: container.clientHeight || 600,
    }

    const scene = new Scene()
    const camera = new PerspectiveCamera(75, size.width / size.height, 0.1, 100)
    camera.position.set(0, 0, 1.2)

    const renderer = new WebGLRenderer({
      alpha: true,
      antialias: window.devicePixelRatio === 1,
    })
    renderer.setSize(size.width, size.height)
    renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio))
    container.appendChild(renderer.domElement)

    const raycaster = new Raycaster()
    const mouse = new Vector2()
    const point = new Vector3()
    const uniforms = { uHover: 0 }
    let hover = false
    let isMobile = window.innerWidth < 767
    let brainMesh: Mesh | null = null
    let instancedMesh: InstancedUniformsMesh | null = null
    let frameId = 0
    let disposed = false

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

    const onMousemove = (e: MouseEvent) => {
      if (!brainMesh) return
      const rect = container.getBoundingClientRect()
      const x = ((e.clientX - rect.left) / size.width) * 2 - 1
      const y = -((e.clientY - rect.top) / size.height) * 2 + 1
      mouse.set(x, y)

      gsap.to(camera.position, { x: x * 0.2, y: -y * 0.2, duration: 0.5 })

      raycaster.setFromCamera(mouse, camera)
      const intersects = raycaster.intersectObject(brainMesh)

      if (intersects.length === 0) {
        if (hover) {
          hover = false
          animateHoverUniform(0)
        }
      } else {
        if (!hover) {
          hover = true
          animateHoverUniform(1)
        }
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

    const onResize = () => {
      size.width = container.clientWidth || 800
      size.height = container.clientHeight || 600
      camera.aspect = size.width / size.height
      camera.updateProjectionMatrix()
      renderer.setSize(size.width, size.height)
      isMobile = window.innerWidth < 767
    }

    const animate = () => {
      if (disposed) return
      camera.lookAt(0, 0, 0)
      camera.position.z = isMobile ? 2.3 : 1.2
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
        },
      })

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
          const colorIndex = MathUtils.randInt(0, TIER_PALETTE.length - 1)
          instancedMesh.setUniformAt("uColor", idx, TIER_PALETTE[colorIndex])
        }
      }

      window.addEventListener("mousemove", onMousemove, { passive: true })
      window.addEventListener("resize", onResize, { passive: true })
      animate()
    })

    return () => {
      disposed = true
      cancelAnimationFrame(frameId)
      window.removeEventListener("mousemove", onMousemove)
      window.removeEventListener("resize", onResize)
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [modelUrl])

  return <div {...props} ref={containerRef} />
}
