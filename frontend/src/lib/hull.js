/** Andrew's monotone-chain 2D convex hull. Points: [[x, y], ...] (n >= 3).
 * Returns hull vertices in counter-clockwise order. */
export function convexHull2D(points) {
  const sortedPoints = points.slice().sort((pointA, pointB) => pointA[0] - pointB[0] || pointA[1] - pointB[1]);
  if (sortedPoints.length < 3) return sortedPoints;

  const cross = (origin, pointA, pointB) =>
    (pointA[0] - origin[0]) * (pointB[1] - origin[1]) - (pointA[1] - origin[1]) * (pointB[0] - origin[0]);

  const lowerHull = [];
  for (const point of sortedPoints) {
    while (lowerHull.length >= 2 && cross(lowerHull[lowerHull.length - 2], lowerHull[lowerHull.length - 1], point) <= 0) {
      lowerHull.pop();
    }
    lowerHull.push(point);
  }
  const upperHull = [];
  for (let pointIdx = sortedPoints.length - 1; pointIdx >= 0; pointIdx--) {
    const point = sortedPoints[pointIdx];
    while (upperHull.length >= 2 && cross(upperHull[upperHull.length - 2], upperHull[upperHull.length - 1], point) <= 0) {
      upperHull.pop();
    }
    upperHull.push(point);
  }
  lowerHull.pop();
  upperHull.pop();
  return lowerHull.concat(upperHull);
}
