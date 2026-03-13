/**
 * skeleton.js — Three.js 3D skeleton renderer
 *
 * COCO-17 joint connections define the bones.
 * Each keypoint gets a glowing sphere; each bone a line.
 * Coordinates are normalised [0,1] → mapped to world space [-2, 2].
 */

const KEYPOINT_NAMES = [
  'nose','l_eye','r_eye','l_ear','r_ear',
  'l_shoulder','r_shoulder','l_elbow','r_elbow',
  'l_wrist','r_wrist','l_hip','r_hip',
  'l_knee','r_knee','l_ankle','r_ankle',
];

const SKELETON_BONES = [
  [0,1],[0,2],[1,3],[2,4],               // head
  [5,6],[5,7],[7,9],[6,8],[8,10],        // arms
  [5,11],[6,12],[11,12],                 // torso
  [11,13],[13,15],[12,14],[14,16],       // legs
];

class SkeletonRenderer {
  constructor(canvasId) {
    this.canvas  = document.getElementById(canvasId);
    this.keypts  = Array(17).fill(null).map(() => ({ x:.5, y:.5, z:.5, confidence:0 }));
    this._init();
  }

  _init() {
    // Scene
    this.scene  = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080c14);
    this.scene.fog = new THREE.FogExp2(0x080c14, 0.12);

    // Camera
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
    this.camera.position.set(0, 1.2, 5);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Orbit controls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;

    // Lights
    this.scene.add(new THREE.AmbientLight(0x203050, 3));
    const point = new THREE.PointLight(0x38bdf8, 3, 10);
    point.position.set(0, 3, 2);
    this.scene.add(point);

    // Grid
    const grid = new THREE.GridHelper(6, 12, 0x1a2840, 0x111d2e);
    grid.position.y = -2;
    this.scene.add(grid);

    // Joint spheres
    const jGeo  = new THREE.SphereGeometry(0.07, 12, 12);
    const jMat  = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x1a6080, roughness: .3 });
    this.jointMeshes = KEYPOINT_NAMES.map(() => {
      const m = new THREE.Mesh(jGeo, jMat.clone());
      this.scene.add(m);
      return m;
    });

    // Bone lines
    const boneMat = new THREE.LineBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.7 });
    this.boneLines = SKELETON_BONES.map(([a, b]) => {
      const geo  = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const line = new THREE.Line(geo, boneMat.clone());
      this.scene.add(line);
      return { line, a, b };
    });

    this._resize();
    window.addEventListener('resize', () => this._resize());
    this._animate();
  }

  _resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Map normalised [0,1] → world [-2, 2] (y is flipped) */
  _toWorld(kp) {
    return new THREE.Vector3(
      (kp.x - 0.5) * 4,
      (0.5 - kp.y) * 4,
      (kp.z - 0.5) * 2,
    );
  }

  updateKeypoints(keypoints) {
    if (!keypoints || keypoints.length !== 17) return;
    this.keypts = keypoints;

    keypoints.forEach((kp, i) => {
      const pos = this._toWorld(kp);
      this.jointMeshes[i].position.copy(pos);
      const conf = kp.confidence;
      this.jointMeshes[i].material.emissiveIntensity = conf * 2;
      this.jointMeshes[i].scale.setScalar(0.5 + conf * 0.5);
    });

    this.boneLines.forEach(({ line, a, b }) => {
      const posA = this._toWorld(keypoints[a]);
      const posB = this._toWorld(keypoints[b]);
      const positions = line.geometry.attributes.position;
      positions.setXYZ(0, posA.x, posA.y, posA.z);
      positions.setXYZ(1, posB.x, posB.y, posB.z);
      positions.needsUpdate = true;
    });
  }

  setAutoRotate(val) { this.controls.autoRotate = val; this.controls.autoRotateSpeed = 1.5; }

  resetCamera() {
    this.camera.position.set(0, 1.2, 5);
    this.controls.reset();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
