"""Professional dark-medical theme + 3D DNA hero banner for the Streamlit app.

Usage in app/streamlit_app.py (right after st.set_page_config):
    from ui_theme import apply_theme, hero_3d
    apply_theme()
    hero_3d()
"""
import streamlit as st
import streamlit.components.v1 as components

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -10%, rgba(45,212,191,.14), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(96,165,250,.14), transparent 55%),
    #0b1220;
}
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1050px; padding-top: 1.5rem; }
h1 { font-weight: 700; letter-spacing: -.02em; color: #f1f5f9; }
h2, h3 { font-weight: 600; color: #e2e8f0; letter-spacing: -.01em; }
[data-testid="stFileUploader"] section {
  background: rgba(255,255,255,.03);
  border: 1.5px dashed rgba(45,212,191,.5);
  border-radius: 14px;
}
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid rgba(255,255,255,.08); }
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(135deg, #14b8a6, #2563eb);
  color: #fff; border: 0; border-radius: 10px;
  padding: .65rem 1.8rem; font-weight: 600;
  box-shadow: 0 8px 22px rgba(20,184,166,.25);
  transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 28px rgba(37,99,235,.3); }
[data-testid="stMetric"] {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 14px; padding: 14px 18px;
}
[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { border-radius: 10px; }
img { border-radius: 12px; }
</style>
"""

_HERO = """
<style>
  html, body { margin: 0; background: transparent; overflow: hidden; font-family: Inter, sans-serif; }
  #wrap { position: relative; width: 100%; height: 100vh;
          border: 1px solid rgba(255,255,255,.08); border-radius: 18px; overflow: hidden;
          background: linear-gradient(135deg, rgba(20,184,166,.10), rgba(37,99,235,.10)); }
  #c { position: absolute; inset: 0; }
  #tag { position: absolute; left: 22px; bottom: 16px; color: #94a3b8; font-size: 12px;
         letter-spacing: .14em; text-transform: uppercase; z-index: 2; }
</style>
<div id="wrap"><div id="c"></div><div id="tag">Imaging &nbsp;·&nbsp; Clinical data &nbsp;·&nbsp; Deep learning</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
try {
  const el = document.getElementById('c');
  let W = el.clientWidth, H = el.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  cam.position.set(0, 0, 14);
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const l1 = new THREE.PointLight(0x2dd4bf, 1.3, 70); l1.position.set(8, 6, 10); scene.add(l1);
  const l2 = new THREE.PointLight(0x60a5fa, 1.1, 70); l2.position.set(-8, -4, 8); scene.add(l2);

  const pivot = new THREE.Group(); pivot.rotation.z = Math.PI / 2 + 0.25; scene.add(pivot);
  const helix = new THREE.Group(); pivot.add(helix);
  const N = 60, R = 1.5, rise = 0.42;
  const sph = new THREE.SphereGeometry(0.2, 24, 24);
  const rodG = new THREE.CylinderGeometry(0.05, 0.05, 1, 8);
  const mA = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, metalness: .3, roughness: .25, emissive: 0x0f766e, emissiveIntensity: .45 });
  const mB = new THREE.MeshStandardMaterial({ color: 0x60a5fa, metalness: .3, roughness: .25, emissive: 0x1d4ed8, emissiveIntensity: .4 });
  const mR = new THREE.MeshStandardMaterial({ color: 0xcbd5e1, transparent: true, opacity: .5 });
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < N; i++) {
    const a = i * 0.5, y = (i - N / 2) * rise;
    const p1 = new THREE.Vector3(Math.cos(a) * R, y, Math.sin(a) * R);
    const p2 = new THREE.Vector3(Math.cos(a + Math.PI) * R, y, Math.sin(a + Math.PI) * R);
    const s1 = new THREE.Mesh(sph, mA); s1.position.copy(p1);
    const s2 = new THREE.Mesh(sph, mB); s2.position.copy(p2);
    const rod = new THREE.Mesh(rodG, mR);
    rod.position.copy(p1).add(p2).multiplyScalar(0.5);
    rod.scale.y = p1.distanceTo(p2);
    rod.quaternion.setFromUnitVectors(up, p2.clone().sub(p1).normalize());
    helix.add(s1, s2, rod);
  }

  const crossMat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: .55, roughness: .4 });
  const crossA = new THREE.BoxGeometry(0.7, 0.2, 0.2), crossB = new THREE.BoxGeometry(0.2, 0.7, 0.2);
  const crosses = [];
  for (let i = 0; i < 14; i++) {
    const g = new THREE.Group();
    g.add(new THREE.Mesh(crossA, crossMat), new THREE.Mesh(crossB, crossMat));
    g.position.set((Math.random() - .5) * 22, (Math.random() - .5) * 8, -2 - Math.random() * 5);
    g.userData = { s: .4 + Math.random() * .8, r: (Math.random() - .5) * .02, o: Math.random() * 6 };
    g.scale.setScalar(.6 + Math.random() * .7);
    scene.add(g); crosses.push(g);
  }

  let mx = 0, my = 0;
  window.addEventListener('pointermove', e => { mx = e.clientX / window.innerWidth - .5; my = e.clientY / window.innerHeight - .5; });
  window.addEventListener('resize', () => {
    W = el.clientWidth; H = el.clientHeight;
    renderer.setSize(W, H); cam.aspect = W / H; cam.updateProjectionMatrix();
  });
  let t = 0;
  (function loop() {
    requestAnimationFrame(loop); t += 0.01;
    helix.rotation.y += 0.008;
    pivot.position.x += (mx * 1.6 - pivot.position.x) * .05;
    pivot.position.y += (-my * 1.0 - pivot.position.y) * .05;
    crosses.forEach(c => { c.rotation.x += c.userData.r; c.rotation.y += c.userData.r * 1.5; c.position.y += Math.sin(t * c.userData.s + c.userData.o) * .004; });
    renderer.render(scene, cam);
  })();
} catch (e) {
  document.getElementById('tag').textContent = 'Multi-Modal Medical AI';
}
</script>
"""


def apply_theme() -> None:
    """Inject the global CSS. Call once, right after st.set_page_config()."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero_3d(height: int = 280) -> None:
    """Render the animated 3D DNA-helix banner."""
    components.html(_HERO, height=height)
