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
  constructor(canvasId, maxPeople = 3) {
    this.canvas    = document.getElementById(canvasId);
    this.maxPeople = maxPeople;
    this.skeletons = []; 
    this.nodes     = []; // Node markers
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

    // Initialise skeleton pool
    const colors = [0x38bdf8, 0x10b981, 0xf59e0b]; // Different colors for people
    const boneColors = [0x818cf8, 0x34d399, 0xfbbf24];

    for (let p = 0; p < this.maxPeople; p++) {
      const joints = KEYPOINT_NAMES.map(() => {
        const geo  = new THREE.SphereGeometry(0.07, 12, 12);
        const mat  = new THREE.MeshStandardMaterial({ 
          color: colors[p % colors.length], 
          emissive: colors[p % colors.length], 
          roughness: .3 
        });
        const m = new THREE.Mesh(geo, mat);
        m.visible = false;
        this.scene.add(m);
        return m;
      });

      const bones = SKELETON_BONES.map(([a, b]) => {
        const mat = new THREE.LineBasicMaterial({ 
          color: boneColors[p % boneColors.length], 
          transparent: true, 
          opacity: 0.7 
        });
        const geo  = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geo, mat);
        line.visible = false;
        this.scene.add(line);
        return { line, a, b };
      });

      this.skeletons.push({ joints, bones });
    }

    // Node markers (ESP32 boxes)
    const nGeo = new THREE.BoxGeometry(0.15, 0.1, 0.2);
    const nMat = new THREE.MeshStandardMaterial({ color: 0x64748b, emissive: 0x334155 });
    for (let i = 0; i < 8; i++) { // Support up to 8 nodes
      const m = new THREE.Mesh(nGeo, nMat);
      m.visible = false;
      this.scene.add(m);
      this.nodes.push(m);
    }

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

  updateSkeletons(skeletons) {
    if (!skeletons || !Array.isArray(skeletons)) return;

    this.skeletons.forEach((sGroup, pIdx) => {
      const data = skeletons[pIdx];
      
      // If no data for this person slot, hide them
      if (!data || data.length !== 17) {
        sGroup.joints.forEach(j => j.visible = false);
        sGroup.bones.forEach(b => b.line.visible = false);
        return;
      }

      // Check if person is "active" (at least some confidence)
      const avgConf = data.reduce((acc, kp) => acc + kp.confidence, 0) / 17;
      const active = avgConf > 0.15; 

      sGroup.joints.forEach((mesh, i) => {
        mesh.visible = active;
        if (!active) return;
        const kp = data[i];
        const pos = this._toWorld(kp);
        mesh.position.copy(pos);
        mesh.material.emissiveIntensity = kp.confidence * 2;
        mesh.scale.setScalar(0.5 + kp.confidence * 0.5);
      });

      sGroup.bones.forEach(({ line, a, b }) => {
        line.visible = active;
        if (!active) return;
        const posA = this._toWorld(data[a]);
        const posB = this._toWorld(data[b]);
        const positions = line.geometry.attributes.position;
        positions.setXYZ(0, posA.x, posA.y, posA.z);
        positions.setXYZ(1, posB.x, posB.y, posB.z);
        positions.needsUpdate = true;
      });
    });
  }

  updateNodes(nodeCoords) {
    // nodeCoords: { id: {x, y, z}, ... }
    this.nodes.forEach(m => m.visible = false);
    Object.entries(nodeCoords).forEach(([id, pos], idx) => {
      if (idx >= this.nodes.length) return;
      const m = this.nodes[idx];
      m.visible = true;
      m.position.set(pos.x, pos.y, pos.z);
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
