import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, quaternion_to_axis_angle
import torch.nn.functional as F

def slerp_torch(q0, q1, t, DOT_THRESHOLD=0.9995):
    if q0.dim() == 1: q0 = q0.unsqueeze(0)
    if q1.dim() == 1: q1 = q1.unsqueeze(0)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    q1_flipped = q1.clone()
    neg_mask = dot < 0

    q1_flipped = torch.where(neg_mask, -q1_flipped, q1_flipped)
    dot[neg_mask] = -dot[neg_mask]

    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_omega = torch.sin(omega)

    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega

    q_slerp = w0 * q0 + w1 * q1_flipped

    q_lerp = (1.0 - t) * q0 + t * q1_flipped
    
    use_lerp_mask = dot > DOT_THRESHOLD
    q_out = torch.where(use_lerp_mask, q_lerp, q_slerp)

    return F.normalize(q_out, p=2, dim=-1)

def matrix_to_quaternion_torch(matrix):
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    
    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    trace = m00 + m11 + m22
    
    q = torch.zeros((matrix.size(0), 4), device=matrix.device, dtype=matrix.dtype)

    mask_pos_trace = trace > 0
    S = torch.sqrt(trace[mask_pos_trace] + 1.0) * 2
    q[mask_pos_trace, 0] = 0.25 * S
    q[mask_pos_trace, 1] = (m21[mask_pos_trace] - m12[mask_pos_trace]) / S
    q[mask_pos_trace, 2] = (m02[mask_pos_trace] - m20[mask_pos_trace]) / S
    q[mask_pos_trace, 3] = (m10[mask_pos_trace] - m01[mask_pos_trace]) / S

    mask_neg_trace = ~mask_pos_trace
    
    mask_m00 = mask_neg_trace & (m00 >= m11) & (m00 >= m22)
    S = torch.sqrt(1.0 + m00[mask_m00] - m11[mask_m00] - m22[mask_m00]) * 2
    q[mask_m00, 0] = (m21[mask_m00] - m12[mask_m00]) / S
    q[mask_m00, 1] = 0.25 * S
    q[mask_m00, 2] = (m01[mask_m00] + m10[mask_m00]) / S
    q[mask_m00, 3] = (m02[mask_m00] + m20[mask_m00]) / S

    mask_m11 = mask_neg_trace & (m11 > m00) & (m11 > m22)
    S = torch.sqrt(1.0 + m11[mask_m11] - m00[mask_m11] - m22[mask_m11]) * 2
    q[mask_m11, 0] = (m02[mask_m11] - m20[mask_m11]) / S
    q[mask_m11, 1] = (m01[mask_m11] + m10[mask_m11]) / S
    q[mask_m11, 2] = 0.25 * S
    q[mask_m11, 3] = (m12[mask_m11] + m21[mask_m11]) / S

    mask_m22 = mask_neg_trace & ~mask_m00 & ~mask_m11
    S = torch.sqrt(1.0 + m22[mask_m22] - m00[mask_m22] - m11[mask_m22]) * 2
    q[mask_m22, 0] = (m10[mask_m22] - m01[mask_m22]) / S
    q[mask_m22, 1] = (m02[mask_m22] + m20[mask_m22]) / S
    q[mask_m22, 2] = (m12[mask_m22] + m21[mask_m22]) / S
    q[mask_m22, 3] = 0.25 * S
    
    return q.squeeze()

def quaternion_to_matrix_torch(quaternions):
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)
    
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    N = quaternions.size(0)
    matrices = torch.zeros((N, 3, 3), device=quaternions.device, dtype=quaternions.dtype)
    
    xx, yy, zz = x * x, y * y, z * z
    xy, wz, xz, wy, yz, wx = x * y, w * z, x * z, w * y, y * z, w * x
    
    matrices[:, 0, 0] = 1 - 2 * (yy + zz)
    matrices[:, 0, 1] = 2 * (xy - wz)
    matrices[:, 0, 2] = 2 * (xz + wy)
    
    matrices[:, 1, 0] = 2 * (xy + wz)
    matrices[:, 1, 1] = 1 - 2 * (xx + zz)
    matrices[:, 1, 2] = 2 * (yz - wx)
    
    matrices[:, 2, 0] = 2 * (xz - wy)
    matrices[:, 2, 1] = 2 * (yz + wx)
    matrices[:, 2, 2] = 1 - 2 * (xx + yy)
    
    return matrices.squeeze()

def homogeneous_matrix_to_pose(H):
    """ Convert a 4x4 homogeneous transformation matrix to pose [tx, ty, tz, qx, qy, qz, qw].
        Assume H is a (4, 4) matrix.
    """
    if H.size(0) != 4 or H.size(1) != 4:
        raise ValueError("H must be a 4x4 matrix")

    # Extract the translation vector
    tx, ty, tz = H[0, 3], H[1, 3], H[2, 3]

    # Extract the 3x3 rotation matrix from the top-left corner
    R = H[:3, :3]

    # Compute the trace of the rotation matrix
    t = torch.trace(R)

    if t > 0:
        S = torch.sqrt(t + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    # Return the pose as [tx, ty, tz, qx, qy, qz, qw]
    return torch.tensor([tx, ty, tz, qx, qy, qz, qw])

def cam_pose_to_matrix(batch_poses):
    """
    Convert quaternion and translation to transformation matrix.
    Args:
        batch_poses: (B, 7) with [R, T] or [T, R]
    Returns:
        (B, 4, 4) transformation matrix
    """
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = quaternion_to_matrix(batch_poses[:,:4])
    c2w[:,:3,3] = batch_poses[:,4:]

    return c2w

# TODO: Identity would cause the problem...
def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3).expand(*batch_dims,3,3).to(data)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def at_to_transform_matrix(rot, trans):
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = axis_angle_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def qt_to_transform_matrix(rot, trans):
    """
    :param rot: axis-angle [bs, 3]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = quaternion_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T

def six_t_to_transform_matrix(rot, trans):
    """
    :param rot: 6d rotation [bs, 6]
    :param trans: translation vector[bs, 3]
    :return: transformation matrix [b, 4, 4]
    """
    bs = rot.shape[0]
    T = torch.eye(4).to(rot)[None, ...].repeat(bs, 1, 1)
    R = rotation_6d_to_matrix(rot)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return 