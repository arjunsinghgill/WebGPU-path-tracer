// ═══════════════════════════════════════════════════════════════════════
//  WebGPU Path Tracer
//  Progressive Monte Carlo path tracing on the GPU via WebGPU compute.
//  Features:
//    · Cook-Torrance GGX microfacet BRDF (specular + diffuse)
//    · Dielectric glass (Fresnel + refraction)
//    · Lambertian diffuse
//    · Emissive surfaces
//    · Russian roulette path termination
//    · Firefly clamping
//    · Depth of Field
//    · Procedural sky (Rayleigh-inspired gradient + sun disk)
//    · ACES / Filmic / Linear tone mapping
//    · Progressive sample accumulation
//    · PCG hash RNG (per-pixel, per-sample)
//    · 6 debug passes: beauty, albedo, normal, depth, shadow, emission
// ═══════════════════════════════════════════════════════════════════════

// ─── PARAMETERS ──────────────────────────────────────────────────────
const P = {
  maxSpp: 512, bounces: 8, clamp: 4.0,
  fov: 55, focal: 3.8, aperture: 0.0,
  skyMode: 0,   // 0=sky 1=sunset 2=studio 3=black
  skyStr: 1.0,  sunEl: 45,
  exposure: 1.0, toneMap: 0, gamma: true,
  pass: 0,
  // camera orbit
  camTheta: 0.25, camPhi: 1.05, camR: 4.5,
  panX: 0, panY: 0,
};

// ─── SCENE DEFINITION ────────────────────────────────────────────────
// Material: { color[3], rough, type(0=diff,1=metal,2=glass,3=emit), ior, emit, _pad }
const MATS = [
  { name:'White Diffuse', color:[.93,.93,.93], rough:.9,  type:0, ior:1.5, emit:0  }, // 0
  { name:'Red Wall',      color:[.80,.06,.06], rough:.9,  type:0, ior:1.5, emit:0  }, // 1
  { name:'Green Wall',    color:[.06,.72,.06], rough:.9,  type:0, ior:1.5, emit:0  }, // 2
  { name:'Area Light',    color:[1.0, 1.0, 1.0],rough:1, type:3, ior:1.5, emit:15 }, // 3
  { name:'Gold Metal',    color:[1.0,.78,.00], rough:.12, type:1, ior:1.5, emit:0  }, // 4
  { name:'Glass Sphere',  color:[.96,.97,1.0], rough:.0,  type:2, ior:1.52,emit:0  }, // 5
  { name:'Blue Diffuse',  color:[.12,.30,.90], rough:.75, type:0, ior:1.5, emit:0  }, // 6
  { name:'Mirror Sphere', color:[.97,.97,.97], rough:.02, type:1, ior:1.5, emit:0  }, // 7
  { name:'Floor',         color:[.82,.80,.74], rough:.65, type:0, ior:1.5, emit:0  }, // 8
  { name:'Ceiling',       color:[.90,.90,.90], rough:.9,  type:0, ior:1.5, emit:0  }, // 9
  { name:'Back Wall',     color:[.88,.88,.88], rough:.85, type:0, ior:1.5, emit:0  }, // 10
  { name:'Copper Box',    color:[.95,.64,.54], rough:.3,  type:1, ior:1.5, emit:0  }, // 11
  { name:'Orange Emit',   color:[1.0,.45,.05], rough:.5,  type:3, ior:1.5, emit:6  }, // 12
];

// Spheres: [cx, cy, cz, radius, matIdx]
const SPHERES_DEF = [
  [ 0.0, -0.62,  0.25, 0.38, 5],   // glass
  [-0.48,-0.05, -0.2,  0.35, 4],   // gold
  [ 0.48,-0.62,  -0.3, 0.38, 7],   // mirror
  [ 0.0,  0.62, -0.45, 0.10, 12],  // orange emissive
];

// Scene objects for the UI
const SCENE_OBJS = [
  { name:'Floor',       mat:8,  color:'#d1ccbe', type:'MESH' },
  { name:'Ceiling',     mat:9,  color:'#e6e6e6', type:'MESH' },
  { name:'Back Wall',   mat:10, color:'#e0e0e0', type:'MESH' },
  { name:'Left Wall',   mat:1,  color:'#cc0f0f', type:'MESH' },
  { name:'Right Wall',  mat:2,  color:'#0fb70f', type:'MESH' },
  { name:'Area Light',  mat:3,  color:'#ffffff', type:'LIGHT' },
  { name:'Gold Metal',  mat:4,  color:'#ffc800', type:'SPHERE' },
  { name:'Glass Ball',  mat:5,  color:'#c8e8ff', type:'SPHERE' },
  { name:'Mirror',      mat:7,  color:'#f7f7f7', type:'SPHERE' },
  { name:'Orange Orb',  mat:12, color:'#ff7310', type:'LIGHT' },
  { name:'Copper Box',  mat:11, color:'#f4a48a', type:'MESH' },
];

// ─── WGSL PATH TRACING SHADER ────────────────────────────────────────
const WGSL = /* wgsl */`

// ═══ Structs ═══════════════════════════════════════════════════════════

struct Uniforms {
  // Camera
  camPos    : vec3f,  pad0 : f32,
  camRight  : vec3f,  pad1 : f32,
  camUp     : vec3f,  pad2 : f32,
  camFwd    : vec3f,  pad3 : f32,
  fovTan    : f32,
  aperture  : f32,
  focalDist : f32,
  // Render
  sampleIdx  : u32,
  maxBounces : u32,
  clampVal   : f32,
  renderPass : u32,
  // Film
  exposure   : f32,
  toneMap    : u32,
  doGamma    : u32,
  // Sky
  skyMode    : u32,
  skyStr     : f32,
  sunEl      : f32,
  // Resolution
  resX       : u32,
  resY       : u32,
  pad4       : f32,
  pad5       : f32,
};

struct Tri {
  v0  : vec3f, pad0 : f32,
  v1  : vec3f, pad1 : f32,
  v2  : vec3f, mat  : u32,   // mat packed into pad slot
  n   : vec3f, pad2 : f32,
};

struct Sphere {
  center : vec3f,
  radius : f32,
  mat    : u32,
  pad0 : f32, pad1 : f32, pad2 : f32,
};

struct Mat {
  color : vec3f,
  rough : f32,
  mtype : u32,   // 0=diffuse 1=metal 2=glass 3=emit
  ior   : f32,
  emit  : f32,
  pad   : f32,
};

// ═══ Bindings ══════════════════════════════════════════════════════════

@group(0) @binding(0) var<uniform>            uni    : Uniforms;
@group(0) @binding(1) var<storage, read>      tris   : array<Tri>;
@group(0) @binding(2) var<storage, read>      sphs   : array<Sphere>;
@group(0) @binding(3) var<storage, read>      mats   : array<Mat>;
@group(0) @binding(4) var<storage, read_write> accum : array<vec4f>;
@group(0) @binding(5) var outTex : texture_storage_2d<rgba16float, write>;

// ═══ PCG RNG ═══════════════════════════════════════════════════════════

var<private> rng : u32;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rngInit(px: u32, py: u32, frame: u32) {
  rng = pcg(px ^ pcg(py ^ pcg(frame * 2654435761u + 1013904223u)));
}

fn rng1() -> f32 {
  rng = pcg(rng);
  return f32(rng) / 4294967296.0;
}
fn rng2() -> vec2f { return vec2f(rng1(), rng1()); }

// ═══ Sampling ══════════════════════════════════════════════════════════

const PI     : f32 = 3.14159265358979;
const TWO_PI : f32 = 6.28318530717959;
const INV_PI : f32 = 0.31830988618379;

fn cosineSampleHemisphere(N: vec3f) -> vec3f {
  let r  = rng2();
  let phi  = TWO_PI * r.x;
  let cosT = sqrt(r.y);
  let sinT = sqrt(1.0 - r.y);
  var T = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.x) > 0.9);
  T = normalize(cross(N, T));
  let B = cross(N, T);
  return normalize(T * cos(phi)*sinT + B * sin(phi)*sinT + N * cosT);
}

fn sampleGGXVNDF(N: vec3f, V: vec3f, alpha: f32) -> vec3f {
  // Visible normal distribution function sampling (Walter 2007 + Heitz 2018)
  let r  = rng2();
  let phi = TWO_PI * r.x;
  let a2  = alpha * alpha;
  let cosT = sqrt((1.0 - r.y) / (1.0 + (a2 - 1.0) * r.y + 1e-8));
  let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
  var T = select(vec3f(1,0,0), vec3f(0,1,0), abs(N.x) > 0.9);
  T = normalize(cross(N, T));
  let B = cross(N, T);
  let H = normalize(T * cos(phi)*sinT + B * sin(phi)*sinT + N * cosT);
  return normalize(2.0 * dot(V, H) * H - V);
}

fn diskSample() -> vec2f {
  let r   = rng2();
  let phi = TWO_PI * r.x;
  return sqrt(r.y) * vec2f(cos(phi), sin(phi));
}

// ═══ BSDF / Shading ════════════════════════════════════════════════════

fn schlick(cosT: f32, ior: f32) -> f32 {
  var r0 = (1.0 - ior) / (1.0 + ior);
  r0 *= r0;
  return r0 + (1.0 - r0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

fn schlickVec(cosT: f32, F0: vec3f) -> vec3f {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

fn D_GGX(NdotH: f32, a2: f32) -> f32 {
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d + 1e-8);
}

fn G_Smith(NdotV: f32, NdotL: f32, a2: f32) -> f32 {
  let gv = NdotV + sqrt(NdotV * NdotV * (1.0 - a2) + a2);
  let gl = NdotL + sqrt(NdotL * NdotL * (1.0 - a2) + a2);
  return 1.0 / (gv * gl + 1e-8);
}

struct BSDFResult {
  dir        : vec3f,
  throughput : vec3f,
  pdf        : f32,
};

fn evalBSDF(wo: vec3f, N: vec3f, mat: Mat, frontFace: bool) -> BSDFResult {
  var r : BSDFResult;

  switch mat.mtype {

    // ── Lambertian Diffuse ───────────────────────────────────────────
    case 0u: {
      let wi   = cosineSampleHemisphere(N);
      let pdf  = max(dot(N, wi), 0.0) * INV_PI;
      r.dir        = wi;
      r.throughput = mat.color; // albedo (pdf * cosine cancel)
      r.pdf        = pdf;
    }

    // ── GGX Metal ───────────────────────────────────────────────────
    case 1u: {
      let alpha = max(mat.rough * mat.rough, 0.001);
      let wi    = sampleGGXVNDF(N, wo, alpha);
      let H     = normalize(wo + wi);
      let NdotV = max(dot(N, wo), 1e-4);
      let NdotL = max(dot(N, wi), 0.0);
      let NdotH = max(dot(N, H),  0.0);
      let VdotH = max(dot(wo, H), 0.0);
      let a2    = alpha * alpha;
      let F0    = mat.color;
      let F     = schlickVec(VdotH, F0);
      let D     = D_GGX(NdotH, a2);
      let G     = G_Smith(NdotV, NdotL, a2);
      let spec  = F * D * G;
      r.dir        = wi;
      r.pdf        = D * NdotH / (4.0 * VdotH + 1e-8);
      r.throughput = select(vec3f(0), spec * NdotL / max(r.pdf, 1e-8), r.pdf > 1e-8 && NdotL > 0.0);
    }

    // ── Dielectric Glass ────────────────────────────────────────────
    case 2u: {
      let ior     = mat.ior;
      let etaRat  = select(ior, 1.0/ior, frontFace);
      let cosI    = abs(dot(wo, N));
      let sin2T   = etaRat * etaRat * (1.0 - cosI * cosI);
      let Fr      = schlick(cosI, ior);
      var wi : vec3f;
      if (sin2T > 1.0 || rng1() < Fr) {
        // Total internal reflection or Fresnel reflect
        wi = 2.0 * dot(wo, N) * N - wo;
      } else {
        // Snell refract
        let cosT = sqrt(max(0.0, 1.0 - sin2T));
        wi = (etaRat * cosI - cosT) * N - etaRat * wo;
      }
      r.dir        = normalize(wi);
      r.throughput = mat.color;
      r.pdf        = 1.0;
    }

    // ── Emissive (terminates path, handled separately) ───────────────
    case 3u: {
      r.dir        = cosineSampleHemisphere(N);
      r.throughput = vec3f(0);
      r.pdf        = 1.0;
    }

    default: {
      r.dir = N; r.throughput = vec3f(0); r.pdf = 1.0;
    }
  }

  return r;
}

// ═══ Sky ════════════════════════════════════════════════════════════════

fn skyColor(dir: vec3f) -> vec3f {
  let d = normalize(dir);
  let sunDir = normalize(vec3f(cos(uni.sunEl), sin(uni.sunEl), 0.3));

  switch uni.skyMode {

    case 1u: { // Sunset
      let t  = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
      let hz = vec3f(1.0, 0.38, 0.05) * 2.5;
      let zen = vec3f(0.08, 0.12, 0.6);
      let gnd = vec3f(0.2, 0.16, 0.12);
      var c = mix(hz, zen, pow(t, 0.5));
      if (d.y < 0.0) { c = mix(c, gnd, clamp(-d.y * 3.0, 0.0, 1.0)); }
      let sd = dot(d, sunDir);
      if (sd > 0.9998) { c += vec3f(8.0, 5.0, 2.0); }
      else if (sd > 0.997) {
        let g = (sd - 0.997) / 0.003;
        c += vec3f(3.0, 1.5, 0.3) * pow(g, 3.0);
      }
      return c * uni.skyStr;
    }

    case 2u: { // Studio
      let t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
      return mix(vec3f(0.06, 0.06, 0.07), vec3f(0.55, 0.58, 0.7), t * t) * uni.skyStr;
    }

    case 3u: { return vec3f(0); } // Black

    default: { // Physical sky
      let t   = clamp(d.y, 0.0, 1.0);
      let hz  = vec3f(0.55, 0.72, 0.98);
      let zen = vec3f(0.08, 0.25, 0.92);
      var c   = mix(hz, zen, pow(t, 0.7));
      if (d.y < 0.0) { c = mix(c, vec3f(0.15, 0.14, 0.12), clamp(-d.y * 4.0, 0.0, 1.0)); }
      // Sun
      let sd = dot(d, sunDir);
      if (sd > 0.9999) { c += vec3f(10.0, 9.0, 7.0); }
      else if (sd > 0.998) {
        let g = (sd - 0.998) / 0.0019;
        c += vec3f(2.5, 1.8, 0.7) * pow(g, 4.0);
      }
      // Horizon glow
      let hg = 1.0 - abs(d.y);
      c += vec3f(0.5, 0.4, 0.2) * pow(hg, 12.0) * 0.4;
      return c * uni.skyStr;
    }
  }
  return vec3f(0);
}

// ═══ Intersection ═══════════════════════════════════════════════════════

struct Hit {
  t   : f32,
  pos : vec3f,
  n   : vec3f,
  mat : u32,
  ff  : bool,
};

fn missHit() -> Hit { var h: Hit; h.t = 1e30; return h; }

fn triIntersect(ro: vec3f, rd: vec3f, tri: Tri) -> f32 {
  let e1 = tri.v1 - tri.v0;
  let e2 = tri.v2 - tri.v0;
  let h  = cross(rd, e2);
  let a  = dot(e1, h);
  if (abs(a) < 1e-7) { return -1.0; }
  let f  = 1.0 / a;
  let s  = ro - tri.v0;
  let u  = f * dot(s, h);
  if (u < 0.0 || u > 1.0) { return -1.0; }
  let q  = cross(s, e1);
  let v  = f * dot(rd, q);
  if (v < 0.0 || u + v > 1.0) { return -1.0; }
  let t  = f * dot(e2, q);
  return select(-1.0, t, t > 1e-4);
}

fn sphIntersect(ro: vec3f, rd: vec3f, s: Sphere) -> f32 {
  let oc = ro - s.center;
  let b  = dot(oc, rd);
  let c  = dot(oc, oc) - s.radius * s.radius;
  let d  = b * b - c;
  if (d < 0.0) { return -1.0; }
  let sq = sqrt(d);
  var t  = -b - sq;
  if (t < 1e-4) { t = -b + sq; }
  return select(-1.0, t, t > 1e-4);
}

fn sceneIntersect(ro: vec3f, rd: vec3f) -> Hit {
  var best = missHit();

  let nT = arrayLength(&tris);
  for (var i = 0u; i < nT; i++) {
    let t = triIntersect(ro, rd, tris[i]);
    if (t > 0.0 && t < best.t) {
      best.t   = t;
      best.pos = ro + rd * t;
      best.n   = tris[i].n;
      best.mat = tris[i].mat;
      best.ff  = dot(rd, best.n) < 0.0;
    }
  }

  let nS = arrayLength(&sphs);
  for (var i = 0u; i < nS; i++) {
    let t = sphIntersect(ro, rd, sphs[i]);
    if (t > 0.0 && t < best.t) {
      best.t   = t;
      best.pos = ro + rd * t;
      best.n   = normalize(best.pos - sphs[i].center);
      best.mat = sphs[i].mat;
      best.ff  = dot(rd, best.n) < 0.0;
    }
  }

  if (!best.ff) { best.n = -best.n; }
  return best;
}

// ═══ Path Tracing ═══════════════════════════════════════════════════════

fn tracePath(ro_in: vec3f, rd_in: vec3f) -> vec4f {
  var ro         = ro_in;
  var rd         = rd_in;
  var throughput = vec3f(1.0);
  var radiance   = vec3f(0.0);

  // Pass data (first hit only)
  var albedoPass  = vec3f(0);
  var normalPass  = vec3f(0);
  var depthPass   = 0.0;
  var shadowPass  = 1.0;
  var emitPass    = vec3f(0);
  var firstHit    = true;

  for (var b = 0u; b <= uni.maxBounces; b++) {
    let hit = sceneIntersect(ro, rd);

    if (hit.t >= 1e29) {
      radiance += throughput * skyColor(rd);
      break;
    }

    let mat = mats[hit.mat];

    // Emission
    if (mat.mtype == 3u) {
      radiance += throughput * mat.color * mat.emit;
      if (firstHit) { emitPass = mat.color * mat.emit; }
      break;
    }

    if (firstHit) {
      firstHit   = false;
      albedoPass = mat.color;
      normalPass = hit.n * 0.5 + 0.5;
      depthPass  = clamp(1.0 - hit.t * 0.15, 0.0, 1.0);
    }

    // Evaluate BSDF
    let bsdf = evalBSDF(-rd, hit.n, mat, hit.ff);

    // Shadow pass: cast a shadow test ray on first bounce
    if (b == 0u) {
      let testRo  = hit.pos + hit.n * 2e-4;
      let testHit = sceneIntersect(testRo, bsdf.dir);
      if (testHit.t < 1e29 && mats[testHit.mat].mtype != 3u) {
        shadowPass = 0.25;
      }
    }

    if (bsdf.pdf < 1e-8) { break; }
    throughput *= bsdf.throughput;

    // Firefly clamp
    let lum = dot(throughput, vec3f(0.2126, 0.7152, 0.0722));
    if (lum > uni.clampVal) { throughput *= uni.clampVal / lum; }

    // Russian roulette (after 3 bounces)
    if (b >= 3u) {
      let q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.01, 1.0);
      if (rng1() > q) { break; }
      throughput /= q;
    }

    rd = bsdf.dir;
    // Offset origin to avoid self-intersection
    ro = hit.pos + bsdf.dir * 1e-4;
  }

  // Select pass output
  switch uni.renderPass {
    case 1u: { return vec4f(albedoPass, 1); }
    case 2u: { return vec4f(normalPass, 1); }
    case 3u: { return vec4f(vec3f(depthPass), 1); }
    case 4u: { return vec4f(vec3f(shadowPass), 1); }
    case 5u: { return vec4f(emitPass, 1); }
    default: { return vec4f(radiance, 1); }
  }
}

// ═══ Tone Mapping ═══════════════════════════════════════════════════════

fn ACES(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3f(0), vec3f(1));
}

fn Filmic(x: vec3f) -> vec3f {
  let A = 0.22; let B = 0.30; let C = 0.10;
  let D = 0.20; let E = 0.01; let F = 0.30;
  let W = 11.2;
  let num = ((x*(A*x+C*B)+D*E) / (x*(A*x+B)+D*F)) - E/F;
  let denom_w = ((W*(A*W+C*B)+D*E) / (W*(A*W+B)+D*F)) - E/F;
  return clamp(num / denom_w, vec3f(0), vec3f(1));
}

fn tonemap(c: vec3f) -> vec3f {
  var col = c * uni.exposure;
  switch uni.toneMap {
    case 0u: { col = ACES(col); }
    case 1u: { col = Filmic(col); }
    default: { col = clamp(col, vec3f(0), vec3f(1)); }
  }
  if (uni.doGamma == 1u) { col = pow(max(col, vec3f(0)), vec3f(1.0 / 2.2)); }
  return col;
}

// ═══ Compute Entry Point ════════════════════════════════════════════════

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px  = gid.xy;
  let res = vec2u(uni.resX, uni.resY);
  if (any(px >= res)) { return; }

  rngInit(px.x, px.y, uni.sampleIdx);

  // Stratified jitter
  let jitter = rng2() - 0.5;
  let uv     = (vec2f(px) + jitter) / vec2f(res) * 2.0 - 1.0;
  let aspect = f32(uni.resX) / f32(uni.resY);

  // Primary ray
  var ro = uni.camPos;
  var rd = normalize(
    uni.camFwd +
    uni.camRight * uv.x * uni.fovTan * aspect +
    uni.camUp    * uv.y * uni.fovTan
  );

  // Depth of Field
  if (uni.aperture > 1e-5) {
    let fp   = ro + rd * uni.focalDist;
    let disk = diskSample() * uni.aperture;
    ro += uni.camRight * disk.x + uni.camUp * disk.y;
    rd  = normalize(fp - ro);
  }

  let color = tracePath(ro, rd);

  // Progressive accumulate
  let idx  = px.y * uni.resX + px.x;
  let prev = accum[idx];
  let n    = f32(uni.sampleIdx) + 1.0;
  let avg  = (prev.xyz * (n - 1.0) + color.xyz) / n;
  accum[idx] = vec4f(avg, 1.0);

  // Write tone-mapped to output texture
  textureStore(outTex, vec2i(px), vec4f(tonemap(avg), 1.0));
}

// ═══ Display (full-screen quad) ══════════════════════════════════════════

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let quad = array<vec2f, 6>(
    vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1),
    vec2f(-1,1),  vec2f(1,-1), vec2f(1,1));
  return vec4f(quad[vi], 0.0, 1.0);
}

@group(0) @binding(0) var texSampler : sampler;
@group(0) @binding(1) var texOut     : texture_2d<f32>;

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(texOut));
  return textureSample(texOut, texSampler, pos.xy / size);
}
`;

// ─── NOISE CHART ──────────────────────────────────────────────────────
const noiseCvs   = document.getElementById('noise-canvas');
const noiseCtx   = noiseCvs.getContext('2d');
const noiseHist  = [];

function drawNoiseChart(v) {
  noiseHist.push(v);
  if (noiseHist.length > 90) noiseHist.shift();
  const w = noiseCvs.parentElement.clientWidth;
  const h = 40;
  noiseCvs.width = w; noiseCvs.height = h;
  noiseCtx.clearRect(0,0,w,h);
  if (noiseHist.length < 2) return;
  const mx = Math.max(...noiseHist, 1e-4);
  noiseCtx.strokeStyle = 'rgba(255,94,26,0.9)';
  noiseCtx.lineWidth = 1.5;
  noiseCtx.beginPath();
  for (let i = 0; i < noiseHist.length; i++) {
    const x = (i / (noiseHist.length - 1)) * w;
    const y = h - 4 - (noiseHist[i] / mx) * (h - 8);
    i === 0 ? noiseCtx.moveTo(x,y) : noiseCtx.lineTo(x,y);
  }
  noiseCtx.stroke();
  // Fill under
  noiseCtx.lineTo(w, h); noiseCtx.lineTo(0, h); noiseCtx.closePath();
  noiseCtx.fillStyle = 'rgba(255,94,26,0.07)';
  noiseCtx.fill();
}

// ─── SCENE UI ──────────────────────────────────────────────────────────
let selObjIdx = -1;

function buildObjList() {
  const el = document.getElementById('obj-list');
  el.innerHTML = '';
  SCENE_OBJS.forEach((o, i) => {
    const div = document.createElement('div');
    div.className = 'obj-item' + (i === selObjIdx ? ' sel' : '');
    div.innerHTML = `<div class="obj-dot" style="background:${o.color}"></div>${o.name}<div class="obj-type">${o.type}</div>`;
    div.onclick = () => selectObj(i);
    el.appendChild(div);
  });
}

function selectObj(i) {
  selObjIdx = i;
  buildObjList();
  const m = MATS[SCENE_OBJS[i].mat];
  loadMatUI(m, SCENE_OBJS[i].mat);
}

function loadMatUI(m, idx) {
  const hex = '#' + m.color.map(c => Math.round(c * 255).toString(16).padStart(2,'0')).join('');
  document.getElementById('mat-color').value = hex;
  document.getElementById('sl-rough').value  = m.rough;
  document.getElementById('v-rough').textContent = m.rough.toFixed(2);
  document.getElementById('sl-ior').value   = m.ior;
  document.getElementById('v-ior').textContent  = m.ior.toFixed(2);
  document.getElementById('sl-emit').value  = m.emit;
  document.getElementById('v-emit').textContent = m.emit.toFixed(1);

  const types = ['mt-diffuse','mt-metal','mt-glass','mt-emit'];
  types.forEach((id, t) => document.getElementById(id).classList.toggle('on', m.type === t));

  document.getElementById('row-rough').style.display = m.type !== 3 ? 'flex' : 'none';
  document.getElementById('row-ior').style.display   = m.type === 2 ? 'flex' : 'none';
  document.getElementById('row-emit').style.display  = m.type === 3 ? 'flex' : 'none';

  updateMatPreview(m);
}

function updateMatPreview(m) {
  const hex = '#' + m.color.map(c => Math.round(c * 255).toString(16).padStart(2,'0')).join('');
  const el  = document.getElementById('mat-preview');
  if (m.type === 3) {
    el.style.background = `radial-gradient(circle, ${hex}, #111)`;
  } else if (m.type === 1) {
    el.style.background = `linear-gradient(135deg, ${shiftColor(hex,-30)} 0%, #fff 40%, ${hex} 100%)`;
  } else if (m.type === 2) {
    el.style.background = `linear-gradient(135deg, rgba(180,220,255,.15), white 50%, rgba(180,220,255,.1))`;
  } else {
    el.style.background = `radial-gradient(circle at 33% 33%, #fff 0%, ${hex} 40%, ${shiftColor(hex,-50)} 100%)`;
  }
}

function shiftColor(hex, v) {
  return '#' + [1,3,5].map(i => Math.max(0,Math.min(255,parseInt(hex.slice(i,i+2),16)+v)).toString(16).padStart(2,'0')).join('');
}

// ─── MATERIAL UI CALLBACKS ────────────────────────────────────────────
window.setMatType = function(t) {
  if (selObjIdx < 0) return;
  const matIdx = SCENE_OBJS[selObjIdx].mat;
  MATS[matIdx].type = t;
  loadMatUI(MATS[matIdx], matIdx);
  rebuildMatsBuffer();
  if (R.running) reset();
};

window.setMatColor = function(hex) {
  if (selObjIdx < 0) return;
  const matIdx = SCENE_OBJS[selObjIdx].mat;
  MATS[matIdx].color = [parseInt(hex.slice(1,3),16)/255, parseInt(hex.slice(3,5),16)/255, parseInt(hex.slice(5,7),16)/255];
  updateMatPreview(MATS[matIdx]);
  rebuildMatsBuffer();
  if (R.running) reset();
};

window.setMatVal = function(key, val, labelId, dec) {
  if (selObjIdx < 0) return;
  const matIdx = SCENE_OBJS[selObjIdx].mat;
  MATS[matIdx][key] = val;
  document.getElementById(labelId).textContent = val.toFixed(dec);
  rebuildMatsBuffer();
  if (R.running) reset();
};

// ─── RENDER CONTROLS ──────────────────────────────────────────────────
window.setEnv = function(e) {
  P.skyMode = {sky:0, sunset:1, studio:2, black:3}[e];
  ['sky','sunset','studio','black'].forEach(k => document.getElementById('env-'+k).classList.toggle('on', k===e));
  if (R.running) reset();
};

window.setToneMap = function(t) {
  P.toneMap = {aces:0, filmic:1, none:2}[t];
  ['aces','filmic','none'].forEach(k => document.getElementById('tm-'+k).classList.toggle('on', k===t));
  if (R.running) reset();
};

window.setPass = function(p) {
  P.pass = p;
  const names = ['beauty','albedo','normal','depth','shadow','emission'];
  names.forEach((n, i) => document.getElementById('pass-'+n).classList.toggle('on', i===p));
  document.getElementById('sb-status').textContent = (names[p]||'beauty').toUpperCase()+' PASS';
  if (R.running) reset();
};

// ─── MATH UTILS ──────────────────────────────────────────────────────
const v3norm = v => { const l = Math.sqrt(v[0]**2+v[1]**2+v[2]**2)||1; return v.map(x=>x/l); };
const v3cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const v3sub   = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const v3dot   = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];

// ─── BUILD SCENE BUFFERS ──────────────────────────────────────────────
function buildTriangles() {
  const tris = [];
  function tri(v0,v1,v2,mat) {
    const e1  = v3sub(v1,v0), e2 = v3sub(v2,v0);
    const n   = v3norm(v3cross(e1,e2));
    tris.push({v0,v1,v2,mat,n});
  }
  function quad(a,b,c,d,mat) { tri(a,b,c,mat); tri(a,c,d,mat); }

  // Cornell box walls in [-1, 1]
  // Floor
  quad([-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1], 8);
  // Ceiling
  quad([-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1], 9);
  // Back wall
  quad([-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1], 10);
  // Left wall (red)
  quad([-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1], 1);
  // Right wall (green)
  quad([1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1], 2);

  // Area light (top center)
  quad([-0.35,0.998,-0.15],[0.35,0.998,-0.15],[0.35,0.998,0.30],[-0.35,0.998,0.30], 3);

  // Tall box (left)
  function box(cx,cy,cz,sx,sy,sz,mat) {
    const x0=cx-sx/2,x1=cx+sx/2,y0=cy-sy/2,y1=cy+sy/2,z0=cz-sz/2,z1=cz+sz/2;
    quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],mat); // -z face
    quad([x0,y0,z1],[x0,y1,z1],[x1,y1,z1],[x1,y0,z1],mat); // +z face
    quad([x0,y0,z0],[x0,y0,z1],[x1,y0,z1],[x1,y0,z0],mat); // -y (bottom)
    quad([x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1],mat); // +y (top)
    quad([x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1],mat); // -x
    quad([x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0],mat); // +x
  }

  box(-0.42, -0.30, 0.18,  0.36, 1.42, 0.36, 11); // copper tall box
  return tris;
}

function buildGPUTriangleBuffer(device, tris) {
  // Struct layout (bytes): v0(12)+pad(4) v1(12)+pad(4) v2(12)+mat(4) n(12)+pad(4) = 64 bytes
  const stride = 16; // floats
  const buf    = new ArrayBuffer(tris.length * stride * 4);
  const f32    = new Float32Array(buf);
  const u32    = new Uint32Array(buf);
  tris.forEach((t, i) => {
    const o = i * stride;
    f32[o+0]=t.v0[0]; f32[o+1]=t.v0[1]; f32[o+2]=t.v0[2]; f32[o+3]=0;
    f32[o+4]=t.v1[0]; f32[o+5]=t.v1[1]; f32[o+6]=t.v1[2]; f32[o+7]=0;
    f32[o+8]=t.v2[0]; f32[o+9]=t.v2[1]; f32[o+10]=t.v2[2]; u32[o+11]=t.mat;
    f32[o+12]=t.n[0]; f32[o+13]=t.n[1]; f32[o+14]=t.n[2]; f32[o+15]=0;
  });
  const gpuBuf = device.createBuffer({ size: buf.byteLength, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(gpuBuf, 0, buf);
  return { gpuBuf, count: tris.length };
}

function buildGPUSphereBuffer(device) {
  // Struct: center(12)+radius(4) + mat(4)+pad(12) = 32 bytes = 8 floats
  const stride = 8;
  const buf    = new ArrayBuffer(SPHERES_DEF.length * stride * 4);
  const f32    = new Float32Array(buf);
  const u32    = new Uint32Array(buf);
  SPHERES_DEF.forEach((s, i) => {
    const o = i * stride;
    f32[o+0]=s[0]; f32[o+1]=s[1]; f32[o+2]=s[2]; f32[o+3]=s[3];
    u32[o+4]=s[4]; f32[o+5]=0; f32[o+6]=0; f32[o+7]=0;
  });
  const gpuBuf = device.createBuffer({ size: buf.byteLength, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(gpuBuf, 0, buf);
  return gpuBuf;
}

function buildGPUMatsBuffer(device) {
  // Struct: color(12)+rough(4) + mtype(4)+ior(4)+emit(4)+pad(4) = 32 bytes = 8 floats
  const stride = 8;
  const buf    = new ArrayBuffer(MATS.length * stride * 4);
  const f32    = new Float32Array(buf);
  const u32    = new Uint32Array(buf);
  MATS.forEach((m, i) => {
    const o = i * stride;
    f32[o+0]=m.color[0]; f32[o+1]=m.color[1]; f32[o+2]=m.color[2]; f32[o+3]=m.rough;
    u32[o+4]=m.type; f32[o+5]=m.ior; f32[o+6]=m.emit; f32[o+7]=0;
  });
  const gpuBuf = device.createBuffer({ size: buf.byteLength, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(gpuBuf, 0, buf);
  return gpuBuf;
}

function buildUniformBuffer(device) {
  // 256 bytes (64 floats), aligned to 256
  const gpuBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
  return gpuBuf;
}

function writeUniforms(device, gpuBuf, sample, W, H) {
  const rad = P.fov * Math.PI / 180;
  const fovTan = Math.tan(rad * 0.5);

  // Orbit camera
  const th = P.camTheta, ph = P.camPhi, r = P.camR;
  const cx = r * Math.sin(ph) * Math.cos(th) + P.panX;
  const cy = r * Math.cos(ph) + P.panY;
  const cz = r * Math.sin(ph) * Math.sin(th);
  const target = [P.panX, P.panY, 0];
  const camPos = [cx, cy, cz];
  const fwd    = v3norm(v3sub(target, camPos));
//  const worldUp= Math.abs(fwd[1]) > 0.99 ? [0,0,1] : [0,1,0];
  const worldUp= [0,0,1];
//  const right  = v3norm(v3cross(fwd, worldUp));
  const right  = [1,0,0];
  const up     = v3cross(right, fwd);

  // Build buffer (64 f32/u32 slots = 256 bytes)
  const ab  = new ArrayBuffer(256);
  const f32 = new Float32Array(ab);
  const u32 = new Uint32Array(ab);

  // Uniforms struct layout (matching WGSL struct):
  // camPos(3)+pad, camRight(3)+pad, camUp(3)+pad, camFwd(3)+pad  → 16 floats
  // fovTan, aperture, focalDist, sampleIdx                        →  4 floats
  // maxBounces, clampVal, pass, exposure                          →  4 floats
  // toneMap, doGamma, skyMode, skyStr                             →  4 floats
  // sunEl, resX, resY, pad4, pad5                                 →  5 floats
  let i = 0;
  f32[i++]=camPos[0]; f32[i++]=camPos[1]; f32[i++]=camPos[2]; f32[i++]=0;
  f32[i++]=right[0];  f32[i++]=right[1];  f32[i++]=right[2];  f32[i++]=0;
  f32[i++]=up[0];     f32[i++]=up[1];     f32[i++]=up[2];     f32[i++]=0;
  f32[i++]=fwd[0];    f32[i++]=fwd[1];    f32[i++]=fwd[2];    f32[i++]=0;
  f32[i++]=fovTan;
  f32[i++]=P.aperture;
  f32[i++]=P.focal;
  u32[i++]=sample;
  u32[i++]=P.bounces;
  f32[i++]=P.clamp;
  u32[i++]=P.pass;
  f32[i++]=P.exposure;
  u32[i++]=P.toneMap;
  u32[i++]=P.gamma ? 1 : 0;
  u32[i++]=P.skyMode;
  f32[i++]=P.skyStr;
  f32[i++]=P.sunEl * Math.PI / 180;
  u32[i++]=W;
  u32[i++]=H;

  device.queue.writeBuffer(gpuBuf, 0, ab);
}

// ─── WEBGPU STATE ─────────────────────────────────────────────────────
const R = {
  device: null,
  computePipeline: null,
  displayPipeline: null,
  computeBindGroup: null,
  displayBindGroup: null,
  uniformBuf: null,
  accumBuf: null,
  outTexture: null,
  matsBuf: null,
  trisBuf: null,
  sphBuf: null,
  triCount: 0,
  canvasCtx: null,
  canvasFormat: null,
  running: false,
  sample: 0,
  W: 0, H: 0,
  rafId: null,
  startTime: 0,
  sampleTimes: [],
};

async function initGPU() {
  if (!navigator.gpu) throw new Error('no-webgpu');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no-adapter');
  R.device = await adapter.requestDevice();
  R.device.lost.then(() => { R.running = false; });
}

function rebuildMatsBuffer() {
  if (!R.device) return;
  R.matsBuf = buildGPUMatsBuffer(R.device);
  // Rebuild bind groups since buffer changed
  rebuildComputeBindGroup();
}

function rebuildComputeBindGroup() {
  const d = R.device;
  R.computeBindGroup = d.createBindGroup({
    layout: R.computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: R.uniformBuf } },
      { binding: 1, resource: { buffer: R.trisBuf } },
      { binding: 2, resource: { buffer: R.sphBuf } },
      { binding: 3, resource: { buffer: R.matsBuf } },
      { binding: 4, resource: { buffer: R.accumBuf } },
      { binding: 5, resource: R.outTexture.createView() },
    ],
  });
}

async function setupRenderer() {
  const canvas = document.getElementById('gpu-canvas');
  const vp     = document.getElementById('viewport');
  const W = Math.floor(vp.clientWidth);
  const H = Math.floor(vp.clientHeight);
  R.W = W; R.H = H;
  canvas.width = W; canvas.height = H;

  document.getElementById('sb-res').textContent = `${W}×${H}`;

  R.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  R.canvasCtx    = canvas.getContext('webgpu');
  R.canvasCtx.configure({ device: R.device, format: R.canvasFormat, alphaMode: 'opaque' });

  const d = R.device;
  const mod = d.createShaderModule({ code: WGSL });

  // Check for compilation errors
  const info = await mod.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === 'error');
  if (errors.length > 0) {
    errors.forEach(e => console.error(`WGSL [${e.lineNum}:${e.linePos}] ${e.message}`));
    throw new Error('WGSL compilation failed');
  }

  // Compute pipeline
  R.computePipeline = d.createComputePipeline({
    layout: 'auto',
    compute: { module: mod, entryPoint: 'main' },
  });

  // Display pipeline
  R.displayPipeline = d.createRenderPipeline({
    layout: 'auto',
    vertex:   { module: mod, entryPoint: 'vs' },
    fragment: { module: mod, entryPoint: 'fs', targets: [{ format: R.canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });

  // Buffers
  R.uniformBuf = buildUniformBuffer(d);
  R.accumBuf   = d.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  R.outTexture = d.createTexture({
    size: [W, H], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const tris = buildTriangles();
  R.triCount  = tris.length;
  const { gpuBuf: triBuf } = buildGPUTriangleBuffer(d, tris);
  R.trisBuf = triBuf;
  R.sphBuf  = buildGPUSphereBuffer(d);
  R.matsBuf = buildGPUMatsBuffer(d);

  // Sampler for display pass
  const sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  R.displayBindGroup = d.createBindGroup({
    layout: R.displayPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: R.outTexture.createView() },
    ],
  });

  rebuildComputeBindGroup();

  document.getElementById('sb-tris').textContent    = R.triCount;
  document.getElementById('sb-spheres').textContent = SPHERES_DEF.length;
}

// ─── RENDER LOOP ──────────────────────────────────────────────────────
function reset() {
  if (!R.device) return;
  R.device.queue.writeBuffer(R.accumBuf, 0, new Float32Array(R.W * R.H * 4));
  R.sample    = 0;
  R.startTime = performance.now();
  R.sampleTimes = [];
}

function renderTick(ts) {
  if (!R.running) return;
  if (R.sample >= P.maxSpp) {
    finishRender();
    return;
  }

  const d   = R.device;
  const now = performance.now();
  writeUniforms(d, R.uniformBuf, R.sample, R.W, R.H);

  const enc = d.createCommandEncoder();

  // Path trace compute pass
  const cp = enc.beginComputePass();
  cp.setPipeline(R.computePipeline);
  cp.setBindGroup(0, R.computeBindGroup);
  cp.dispatchWorkgroups(Math.ceil(R.W / 8), Math.ceil(R.H / 8));
  cp.end();

  // Blit to screen
  const rp = enc.beginRenderPass({
    colorAttachments: [{
      view: R.canvasCtx.getCurrentTexture().createView(),
      clearValue: { r:0, g:0, b:0, a:1 },
      loadOp: 'clear', storeOp: 'store',
    }],
  });
  rp.setPipeline(R.displayPipeline);
  rp.setBindGroup(0, R.displayBindGroup);
  rp.draw(6);
  rp.end();

  d.queue.submit([enc.finish()]);

  R.sample++;
  R.sampleTimes.push(now);
  if (R.sampleTimes.length > 30) R.sampleTimes.shift();

  // Update stats
  const elapsed = (now - R.startTime) / 1000;
  const pct     = R.sample / P.maxSpp;
  const spps    = R.sampleTimes.length > 1
    ? (R.sampleTimes.length - 1) / ((R.sampleTimes[R.sampleTimes.length-1] - R.sampleTimes[0]) / 1000)
    : 0;
  const mrays   = spps * R.W * R.H / 1e6;
  const eta     = spps > 0 ? (P.maxSpp - R.sample) / spps : 0;
  const noise   = Math.max(0, 1 - Math.log(R.sample + 1) / Math.log(P.maxSpp + 1));

  document.getElementById('progress-bar').style.width = (pct * 100) + '%';
  document.getElementById('ring-prog').style.strokeDashoffset = (138 * (1 - pct)).toFixed(2);
  document.getElementById('ring-pct').textContent  = Math.round(pct * 100) + '%';
  document.getElementById('t-spp').textContent     = R.sample;
  document.getElementById('t-mrays').textContent   = mrays.toFixed(1);
  document.getElementById('t-fps').textContent     = spps.toFixed(1);
  document.getElementById('t-time').textContent    = elapsed.toFixed(1) + 's';
  document.getElementById('t-noise').textContent   = (noise * 100).toFixed(1) + '%';
  document.getElementById('ri-cur').textContent    = R.sample;
  document.getElementById('p-elapsed').textContent = elapsed.toFixed(1) + 's';
  document.getElementById('p-eta').textContent     = eta > 0 ? eta.toFixed(1) + 's' : '—';
  document.getElementById('p-spps').textContent    = spps.toFixed(2);
  document.getElementById('sb-status').textContent = 'RENDERING';

  drawNoiseChart(noise);
  R.rafId = requestAnimationFrame(renderTick);
}

function finishRender() {
  R.running = false;
  document.getElementById('btn-render').textContent = '▶  START RENDER';
  document.getElementById('sb-status').textContent  = 'CONVERGED ✓';
  document.getElementById('ring-prog').style.strokeDashoffset = '0';
  document.getElementById('ring-pct').textContent  = '100%';
  document.getElementById('progress-bar').style.width = '100%';
}

window.toggleRender = async function() {
  if (!R.device) {
    try {
      document.getElementById('sb-status').textContent = 'INIT GPU…';
      await initGPU();
      await setupRenderer();
    } catch(e) {
      console.error(e);
      if (e.message === 'no-webgpu') {
        document.getElementById('no-webgpu').classList.add('show');
      }
      document.getElementById('sb-status').textContent = 'ERROR';
      return;
    }
  }

  if (R.running) {
    R.running = false;
    if (R.rafId) cancelAnimationFrame(R.rafId);
    document.getElementById('btn-render').textContent = '▶  START RENDER';
    document.getElementById('sb-status').textContent = 'PAUSED';
  } else {
    reset();
    R.running = true;
    document.getElementById('btn-render').textContent = '■  STOP';
    R.rafId = requestAnimationFrame(renderTick);
  }
};

// Reset accumulation when camera moves
function cameraChanged() {
  if (R.device && R.accumBuf) {
    R.device.queue.writeBuffer(R.accumBuf, 0, new Float32Array(R.W * R.H * 4));
    R.sample = 0;
    R.startTime = performance.now();
    R.sampleTimes = [];
  }
}

// ─── CAMERA ORBIT ────────────────────────────────────────────────────
const canvas = document.getElementById('gpu-canvas');
let mDown = false, mButton = 0, mLast = [0,0];

P.camTheta = 0.28; P.camPhi = 1.08; P.camR = 4.5;
P.panX = 0; P.panY = 0;

canvas.addEventListener('mousedown', e => {
  mDown = true; mButton = e.button;
  mLast = [e.clientX, e.clientY];
  e.preventDefault();
});
window.addEventListener('mouseup', () => mDown = false);
window.addEventListener('mousemove', e => {
  if (!mDown) return;
  const dx = (e.clientX - mLast[0]) * 0.007;
  const dy = (e.clientY - mLast[1]) * 0.007;
  mLast = [e.clientX, e.clientY];
  if (mButton === 0) {
    P.camTheta -= dx;
    P.camPhi = Math.max(0.05, Math.min(Math.PI - 0.05, P.camPhi + dy));
  } else if (mButton === 2) {
    const th = P.camTheta, ph = P.camPhi;
    const r  = [-Math.sin(th), 0, Math.cos(th)];
    const u  = [-Math.cos(ph)*Math.cos(th), Math.sin(ph), -Math.cos(ph)*Math.sin(th)];
    P.panX -= r[0]*dx*1.5; P.panY -= u[1]*dy*1.5;
  }
  cameraChanged();
});
canvas.addEventListener('wheel', e => {
  P.camR = Math.max(1, Math.min(14, P.camR + e.deltaY * 0.005));
  cameraChanged();
  e.preventDefault();
}, { passive: false });
canvas.addEventListener('contextmenu', e => e.preventDefault());

// ─── INIT ─────────────────────────────────────────────────────────────
buildObjList();
document.getElementById('ri-max').textContent = P.maxSpp;

// Auto-start after short delay
window.addEventListener('load', () => setTimeout(() => {
  window.toggleRender();
}, 200));
