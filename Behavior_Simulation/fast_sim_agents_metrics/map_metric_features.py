# Copyright (c) 2024 Waymo LLC. All rights reserved.

# This is licensed under a BSD+Patent license.
# Please see LICENSE and PATENTS text files.
# ==============================================================================
"""Map-based metric features for sim agents."""

from typing import Optional, Sequence

import torch



# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Off-road threshold, i.e. smallest distance away from the road edge that is
# considered to be a off-road.
OFFROAD_DISTANCE_THRESHOLD = 0.0

# How close the start and end point of a map feature need to be for the feature
# to be considered cyclic, in m^2.
_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0
# Scaling factor for vertical distances used when finding the closest segment to
# a query point. This prevents wrong associations in cases with under- and
# over-passes.
_Z_STRETCH_FACTOR = 3.0


def dot_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Computes the dot product between two 2D vectors.

  Args:
    a: A tensor of shape (..., 2) containing the first 2D vector.
    b: A tensor of shape (..., 2) containing the second 2D vector.

  Returns:
    A tensor of shape (...) containing the dot product between the vectors.
  """
  return torch.sum(a * b, dim=-1)


def cross_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Computes the z-component of the cross product between two 2D vectors.

  Args:
    a: A tensor of shape (..., 2) containing the first 2D vector.
    b: A tensor of shape (..., 2) containing the second 2D vector.

  Returns:
    A tensor of shape (...) containing the z-component of the cross product
    between the vectors.
  """
  return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def compute_distance_to_road_edge(
    *,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    road_edge_polylines: Sequence[torch.Tensor]
) -> torch.Tensor:
  """Computes the distance to the road edge for each of the evaluated objects.

  Args:
    boxes: A float Tensor of shape (num_rollouts, num_objects, num_steps, [x,y,z,l,w,h,heading]) 
    valid: A boolean Tensor of shape (num_objects, num_steps) containing the
      validity of the objects over time.
    evaluated_object_mask: A boolean tensor of shape (num_objects), indicating
      whether each object should be considered part of the "evaluation set".
    road_edge_polylines: A sequence of polylines, each defined as a sequence of
      3d points with x, y, and z-coordinates. The polylines should be oriented
      such that port side is on-road and starboard side is off-road, a.k.a
      counterclockwise winding order.

  Returns:
    A tensor of shape (num_rollouts, num_evaluated_objects, num_steps), containing the
    distance to the road edge, for each timestep and for all the objects
    to be evaluated, as specified by `evaluated_object_mask`.

  Raises:
    ValueError: When the `road_edge_polylines` is empty, i.e. there is no map
      information in the Scenario.
  """
  if not road_edge_polylines:
    raise ValueError('Missing road edges.')

  num_rollouts, num_objects, num_steps, num_features = boxes.shape
  boxes = boxes.reshape(num_rollouts * num_objects * num_steps, num_features)
  # Compute box corners using `box_utils`, and take the xyz coords of the bottom
  # corners.
  box_corners = get_upright_3d_box_corners(boxes)[:, :4]
  box_corners = box_corners.reshape(num_rollouts, num_objects, num_steps, 4, 3)

  # Gather objects in the evaluation set
  # `eval_corners` shape: (num_rollouts, num_evaluated_objects, num_steps, 4, 3).
  eval_corners = box_corners[:, evaluated_object_mask]
  num_eval_objects = eval_corners.shape[1]

  # Flatten query points.
  # `flat_eval_corners` shape: (num_rollouts * num_evaluated_objects * num_steps * 4, 3).
  flat_eval_corners = eval_corners.reshape(-1, 3)

  # Tensorize road edges.
  polylines_tensor = _tensorize_polylines(road_edge_polylines)
  is_polyline_cyclic = _check_polyline_cycles(road_edge_polylines)

  # Compute distances for all query points.
  # `corner_distance_to_road_edge` shape: (num_rollouts * num_evaluated_objects * num_steps * 4).
  corner_distance_to_road_edge = _compute_signed_distance_to_polylines(
      xyzs=flat_eval_corners, polylines=polylines_tensor,
      is_polyline_cyclic=is_polyline_cyclic, z_stretch=_Z_STRETCH_FACTOR
  )
  # `corner_distance_to_road_edge` shape: (num_rollouts, num_evaluated_objects, num_steps, 4).
  corner_distance_to_road_edge = corner_distance_to_road_edge.reshape(
      num_rollouts, num_eval_objects, num_steps, 4
  )

  # Reduce to most off-road corner.
  # `signed_distances` shape: (num_rollouts, num_evaluated_objects, num_steps).
  signed_distances = torch.max(corner_distance_to_road_edge, dim=-1)[0]

  # Mask out invalid boxes.
  eval_validity = valid[evaluated_object_mask]
  eval_validity = eval_validity.unsqueeze(0).expand(num_rollouts, -1, -1)
  return torch.where(eval_validity, signed_distances, -EXTREMELY_LARGE_DISTANCE)


def _tensorize_polylines(polylines: Sequence[torch.Tensor]) -> torch.Tensor:
  """Stacks a sequence of polylines into a tensor.

  Args:
    polylines: A sequence of Polyline objects.

  Returns:
    A float tensor with shape (num_polylines, max_length, 4) containing xyz
      coordinates and a validity flag for all points in the polylines. Polylines
      are padded with zeros up to the length of the longest one.
  """
  max_length = max([len(polyline) for polyline in polylines])
  tensorize_polylines = []

  for i, polyline in enumerate(polylines):
    if len(polyline) < 2:
      continue
    tensorize_polyline = torch.zeros(max_length, 4, device=polyline.device)
    tensorize_polyline[:len(polyline),0:3] = polyline
    tensorize_polyline[:len(polyline),3] = 1.0
    tensorize_polylines.append(tensorize_polyline)
        
  return torch.stack(tensorize_polylines, dim=0)


def _check_polyline_cycles(polylines: Sequence[torch.Tensor]) -> torch.Tensor:
  """Checks if given polylines are cyclic and returns the result as a tensor.

  Args:
    polylines: A sequence of Polyline objects.

  Returns:
    A bool tensor with shape (num_polylines) indicating whether each polyline is
    cyclic.
  """
  cycles = []
  for polyline in polylines:
    # Skip degenerate polylines.
    if len(polyline) < 2:
      continue
    cycles.append(torch.sum(torch.square(polyline[0] - polyline[-1]), dim=-1)< _CYCLIC_MAP_FEATURE_TOLERANCE_M2)
  # shape: (num_polylines)
  return torch.stack(cycles, dim=0)


def _compute_signed_distance_to_polylines(
    xyzs: torch.Tensor,
    polylines: torch.Tensor,
    is_polyline_cyclic: Optional[torch.Tensor] = None,
    z_stretch: float = 1.0,
) -> torch.Tensor:
  """Computes the signed distance to the 2D boundary defined by polylines.

  Negative distances correspond to being inside the boundary (e.g. on the
  road), positive distances to being outside (e.g. off-road).

  The polylines should be oriented such that port side is inside the boundary
  and starboard is outside, a.k.a counterclockwise winding order.

  The altitudes i.e. the z-coordinates of query points and polyline segments
  are used to pair each query point with the most relevant segment, that is
  closest and at the right altitude. The distances returned are 2D distances in
  the xy plane.

  Note: degenerate segments (start == end) can cause undefined behaviour.

  Args:
    xyzs: A float Tensor of shape (num_points, 3) containing xyz coordinates of
      query points.
    polylines: Tensor with shape (num_polylines, num_segments+1, 4) containing
      sequences of xyz coordinates and validity, representing start and end
      points of consecutive segments.
    is_polyline_cyclic: A boolean Tensor with shape (num_polylines) indicating
      whether each polyline is cyclic. If None, all polylines are considered
      non-cyclic.
    z_stretch: Factor by which to scale distances over the z axis. This can be
      done to ensure edge points from the wrong level (e.g. overpasses) are not
      selected. Defaults to 1.0 (no stretching).

  Returns:
    A tensor of shape (num_points), containing the signed 2D distance from
      queried points to the nearest polyline.
  """
  num_points = xyzs.shape[0]
  num_polylines = polylines.shape[0]
  num_segments = polylines.shape[1] - 1

  # shape: (num_polylines, num_segments+1)
  is_point_valid = polylines[:, :, 3].bool()
  # shape: (num_polylines, num_segments)
  is_segment_valid = torch.logical_and(
      is_point_valid[:, :-1], is_point_valid[:, 1:]
  )

  if is_polyline_cyclic is None:
    is_polyline_cyclic = torch.zeros(num_polylines, dtype=torch.bool, device=polylines.device)

  # Get distance to each segment.
  # shape: (num_points, num_polylines, num_segments, 3)
  xyz_starts = polylines[None, :, :-1, :3]
  xyz_ends = polylines[None, :, 1:, :3]
  start_to_point = xyzs[:, None, None, :3] - xyz_starts
  start_to_end = xyz_ends - xyz_starts

  # Relative coordinate of point projection on segment.
  # shape: (num_points, num_polylines, num_segments)
  rel_t = torch.div(
      dot_product_2d(
          start_to_point[..., :2], start_to_end[..., :2]
      ),
      dot_product_2d(
          start_to_end[..., :2], start_to_end[..., :2]
      ).clamp(min=1e-10)  # Avoid division by zero
  )

  # Negative if point is on port side of segment, positive if point on
  # starboard side of segment.
  # shape: (num_points, num_polylines, num_segments)
  n = torch.sign(
      cross_product_2d(
          start_to_point[..., :2], start_to_end[..., :2]
      )
  )

  # Compute the absolute 3d distance to segment.
  # The vertical component is scaled by `z-stretch` to increase the separation
  # between different road altitudes.
  # shape: (num_points, num_polylines, num_segments, 3)
  segment_to_point = start_to_point - (
      start_to_end * torch.clamp(rel_t, 0.0, 1.0).unsqueeze(-1)
  )
  # shape: (3)
  stretch_vector = torch.tensor([1.0, 1.0, z_stretch], dtype=torch.float32, device=polylines.device)
  # shape: (num_points, num_polylines, num_segments)
  distance_to_segment_3d = torch.norm(
      segment_to_point * stretch_vector.view(1, 1, 1, 3),
      dim=-1,
  )
  # Absolute planar distance to segment.
  # shape: (num_points, num_polylines, num_segments)
  distance_to_segment_2d = torch.norm(
      segment_to_point[..., :2],
      dim=-1,
  )

  # There are 3 cases:
  #   - if the point projection on the line falls within the segment, the sign
  #       of the distance is `n`.
  #   - if the point projection on the segment falls before the segment start,
  #       the sign of the distance depends on the convexity of the prior and
  #       nearest segments.
  #   - if the point projection on the segment falls after the segment end, the
  #       sign of the distance depends on the convexity of the nearest and next
  #       segments.

  # shape: (num_points, num_polylines, num_segments+2, 2)
  start_to_end_padded = torch.cat(
      [
          start_to_end[:, :, -1:, :2],
          start_to_end[..., :2],
          start_to_end[:, :, :1, :2],
      ],
      dim=-2,
  )
  # shape: (num_points, num_polylines, num_segments+1)
  is_locally_convex = torch.greater(
      cross_product_2d(
          start_to_end_padded[:, :, :-1], start_to_end_padded[:, :, 1:]
      ),
      0.0,
  )

  # Get shifted versions of `n` and `is_segment_valid`. If the polyline is
  # cyclic, the tensors are rolled, else they are padded with their edge value.
  # shape: (num_points, num_polylines, num_segments)
  n_prior = torch.cat(
      [
          torch.where(
              is_polyline_cyclic.unsqueeze(0).unsqueeze(-1),
              n[:, :, -1:],
              n[:, :, :1],
          ),
          n[:, :, :-1],
      ],
      dim=-1,
  )
  n_next = torch.cat(
      [
          n[:, :, 1:],
          torch.where(
              is_polyline_cyclic.unsqueeze(0).unsqueeze(-1),
              n[:, :, :1],
              n[:, :, -1:],
          ),
      ],
      dim=-1,
  )
  # shape: (num_polylines, num_segments)
  is_prior_segment_valid = torch.cat(
      [
          torch.where(
              is_polyline_cyclic.unsqueeze(-1),
              is_segment_valid[:, -1:],
              is_segment_valid[:, :1],
          ),
          is_segment_valid[:, :-1],
      ],
      dim=-1,
  )
  is_next_segment_valid = torch.cat(
      [
          is_segment_valid[:, 1:],
          torch.where(
              is_polyline_cyclic.unsqueeze(-1),
              is_segment_valid[:, :1],
              is_segment_valid[:, -1:],
          ),
      ],
      dim=-1,
  )

  # shape: (num_points, num_polylines, num_segments)
  sign_if_before = torch.where(
      is_locally_convex[:, :, :-1],
      torch.maximum(n, n_prior),
      torch.minimum(n, n_prior),
  )
  sign_if_after = torch.where(
      is_locally_convex[:, :, 1:], torch.maximum(n, n_next), torch.minimum(n, n_next)
  )

  # shape: (num_points, num_polylines, num_segments)
  sign_to_segment = torch.where(
      (rel_t < 0.0) & is_prior_segment_valid.unsqueeze(0),
      sign_if_before,
      torch.where((rel_t > 1.0) & is_next_segment_valid.unsqueeze(0), sign_if_after, n)
  )

  # Flatten polylines together.
  # shape: (num_points, all_segments)
  distance_to_segment_3d = distance_to_segment_3d.reshape(
      num_points, num_polylines * num_segments
  )
  distance_to_segment_2d = distance_to_segment_2d.reshape(
      num_points, num_polylines * num_segments
  )
  sign_to_segment = sign_to_segment.reshape(
      num_points, num_polylines * num_segments
  )

  # Mask out invalid segments.
  # shape: (all_segments)
  is_segment_valid = is_segment_valid.reshape(
      num_polylines * num_segments
  )
  # shape: (num_points, all_segments)
  distance_to_segment_3d = torch.where(
      is_segment_valid.unsqueeze(0),
      distance_to_segment_3d,
      EXTREMELY_LARGE_DISTANCE,
  )
  distance_to_segment_2d = torch.where(
      is_segment_valid.unsqueeze(0),
      distance_to_segment_2d,
      EXTREMELY_LARGE_DISTANCE,
  )

  # Get closest segment according to absolute 3D distance and return the
  # corresponding signed 2D distance.
  # shape: (num_points)
  closest_segment_index = torch.argmin(distance_to_segment_3d, dim=-1)
  distance_sign = torch.gather(
      sign_to_segment, 1, closest_segment_index.unsqueeze(-1)
  ).squeeze(-1)
  distance_2d = torch.gather(
      distance_to_segment_2d, 1, closest_segment_index.unsqueeze(-1)
  ).squeeze(-1)
  return distance_sign * distance_2d


def _compute_winding_number(
    xyzs: torch.Tensor,
    polylines: torch.Tensor,
    closest_segment: torch.Tensor,
    t: torch.Tensor,
    is_polyline_cyclic: torch.Tensor,
) -> torch.Tensor:
  """Computes the winding number for each point with respect to each polyline.

  The winding number is used to determine if a point is inside or outside a
  polyline. A positive winding number indicates the point is inside the polyline,
  while a negative or zero winding number indicates it is outside.

  Args:
    xyzs: A float Tensor of shape (num_points, 1, 1, 3) containing xyz
      coordinates of query points.
    polylines: Tensor with shape (num_polylines, num_segments+1, 4) containing
      sequences of xyz coordinates and validity, representing start and end
      points of consecutive segments.
    closest_segment: A long Tensor of shape (num_points, num_polylines)
      containing the index of the closest segment for each point and polyline.
    t: A float Tensor of shape (num_points, num_polylines, num_segments)
      containing the projection of each point onto each segment.
    is_polyline_cyclic: A boolean Tensor with shape (num_polylines) indicating
      whether each polyline is cyclic.

  Returns:
    A float Tensor of shape (num_points, num_polylines) containing the winding
      number for each point with respect to each polyline.
  """
  num_points = xyzs.shape[0]
  num_polylines = polylines.shape[0]
  num_segments = polylines.shape[1] - 1

  # Get segment endpoints.
  # shape: (num_points, num_polylines, num_segments, 3)
  xyz_starts = polylines[None, :, :-1, :3]
  xyz_ends = polylines[None, :, 1:, :3]

  # Compute segment vectors.
  # shape: (num_points, num_polylines, num_segments, 3)
  segment_vectors = xyz_ends - xyz_starts

  # Compute point vectors.
  # shape: (num_points, num_polylines, num_segments, 3)
  point_vectors = xyzs - xyz_starts

  # Compute cross product between segment and point vectors.
  # shape: (num_points, num_polylines, num_segments, 3)
  cross_product = torch.cross(segment_vectors, point_vectors, dim=-1)

  # Compute winding number.
  # shape: (num_points, num_polylines)
  winding_number = torch.zeros(num_points, num_polylines, device=xyzs.device)

  # Add contribution from each segment.
  for i in range(num_segments):
    # Get next segment index.
    next_i = (i + 1) % num_segments

    # Compute angle between segments.
    # shape: (num_points, num_polylines)
    angle = torch.atan2(
        cross_product[:, :, i, 2],
        torch.sum(segment_vectors[:, :, i, :2] * point_vectors[:, :, i, :2], dim=-1)
    )

    # Add angle to winding number.
    winding_number += angle

    # If polyline is not cyclic, add contribution from last segment.
    if not is_polyline_cyclic[i]:
      # Compute angle between last segment and first segment.
      # shape: (num_points)
      last_angle = torch.atan2(
          cross_product[:, i, -1, 2],
          torch.sum(segment_vectors[:, i, -1, :2] * point_vectors[:, i, -1, :2], dim=-1)
      )
      winding_number[:, i] += last_angle

  return winding_number

def get_upright_3d_box_corners(boxes):
  """Given a set of upright boxes, return its 8 corners.

  Given a set of boxes, returns its 8 corners. The corners are ordered layers
  (bottom, top) first and then counter-clockwise within each layer.

  Args:
    boxes: torch Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
      height, heading].

  Returns:
    corners: torch Tensor [N, 8, 3].
  """
  center_x, center_y, center_z, length, width, height, heading = torch.unbind(
      boxes, dim=-1)

  # [N, 3, 3]
  rotation = get_yaw_rotation(heading)
  # [N, 3]
  translation = torch.stack([center_x, center_y, center_z], dim=-1)

  l2 = length * 0.5
  w2 = width * 0.5
  h2 = height * 0.5

  # [N, 8, 3]
  corners = torch.stack([
      l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
      -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
  ], dim=-1).reshape(-1, 8, 3)
  
  # [N, 8, 3]
  corners = torch.matmul(rotation, corners.transpose(-2, -1)).transpose(-2, -1) + translation.unsqueeze(-2)

  return corners


def get_yaw_rotation(heading):
  """Gets rotation matrix for yaw rotation.

  Args:
    heading: [N] tensor of heading angles in radians.

  Returns:
    [N, 3, 3] rotation matrix.
  """
  cos_h = torch.cos(heading)
  sin_h = torch.sin(heading)
  zeros = torch.zeros_like(cos_h)
  ones = torch.ones_like(cos_h)
  
  rotation = torch.stack([
      torch.stack([cos_h, -sin_h, zeros], dim=-1),
      torch.stack([sin_h, cos_h, zeros], dim=-1),
      torch.stack([zeros, zeros, ones], dim=-1)
  ], dim=-2)
  
  return rotation