import { NormalizedLandmarkList } from '@mediapipe/face_mesh';

// Eye landmark indices for MediaPipe Face Mesh
const LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];

// Specific indices for EAR calculation
const LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380];

// Mouth landmark indices
const MOUTH_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

export function calculateEAR(eyeLandmarks: Array<{ x: number; y: number }>): number {
  if (eyeLandmarks.length < 6) return 0;

  // Calculate distances
  const p1 = eyeLandmarks[1]; // Top of eye
  const p2 = eyeLandmarks[5]; // Bottom of eye
  const p3 = eyeLandmarks[2]; // Top of eye (second point)
  const p4 = eyeLandmarks[4]; // Bottom of eye (second point)
  const p5 = eyeLandmarks[0]; // Left corner
  const p6 = eyeLandmarks[3]; // Right corner

  // Vertical distances
  const d1 = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
  const d2 = Math.sqrt((p3.x - p4.x) ** 2 + (p3.y - p4.y) ** 2);
  
  // Horizontal distance
  const d3 = Math.sqrt((p5.x - p6.x) ** 2 + (p5.y - p6.y) ** 2);

  // EAR formula
  return (d1 + d2) / (2.0 * d3);
}

export function calculateMouthRatio(mouthLandmarks: Array<{ x: number; y: number }>): number {
  // We expect the mouth landmark array to contain at least the set of indices
  // defined in MOUTH_INDICES (20 items). If not available, return 0.
  if (mouthLandmarks.length < 16) {
    // Not enough points to compute a reliable mouth ratio
    // Keep a lightweight log for diagnostics
    console.log(`❌ Insufficient mouth landmarks: ${mouthLandmarks.length}`);
    return 0;
  }

  // Use inner lip landmarks for vertical measurement and outer corners for width.
  // The ordering in MOUTH_INDICES is:
  // [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
  // where index 5 -> landmark 13 (upper inner lip) and index 15 -> landmark 14 (lower inner lip)
  const upperInner = mouthLandmarks[5];   // inner upper lip (landmark 13)
  const lowerInner = mouthLandmarks[15];  // inner lower lip (landmark 14)

  // Outer mouth corners for width estimation
  const leftCorner = mouthLandmarks[0];   // left corner (landmark 78)
  const rightCorner = mouthLandmarks[10]; // right corner (landmark 308)

  // Euclidean distances for height and width (more robust than single-axis diffs)
  const height = Math.hypot(upperInner.x - lowerInner.x, upperInner.y - lowerInner.y);
  const width = Math.hypot(rightCorner.x - leftCorner.x, rightCorner.y - leftCorner.y);

  if (width === 0 || Number.isNaN(width) || Number.isNaN(height)) return 0;

  const ratio = height / width;

  // Return the raw ratio (caller compares against configured threshold).
  // Keep the value reasonably rounded to avoid tiny floating noise.
  return Math.round(ratio * 10000) / 10000;
}

export function extractEyeLandmarks(landmarks: NormalizedLandmarkList) {
  const leftEye = LEFT_EYE_EAR_INDICES.map(index => ({
    x: landmarks[index].x,
    y: landmarks[index].y
  }));

  const rightEye = RIGHT_EYE_EAR_INDICES.map(index => ({
    x: landmarks[index].x,
    y: landmarks[index].y
  }));

  return { left: leftEye, right: rightEye };
}

export function extractMouthLandmarks(landmarks: NormalizedLandmarkList) {
  const mouth = MOUTH_INDICES.map(index => ({
    x: landmarks[index].x,
    y: landmarks[index].y
  }));

  return { points: mouth };
}