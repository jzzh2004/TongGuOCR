import copy
import itertools

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def findMiddle(corners, mask, points=[0.25, 0.5, 0.75]):
    num_middle_points = len(points)
    top = [np.array([])] * num_middle_points
    bottom = [np.array([])] * num_middle_points
    left = [np.array([])] * num_middle_points
    right = [np.array([])] * num_middle_points

    center_top = []
    center_bottom = []
    center_left = []
    center_right = []

    center = (int((corners[0][0][1] + corners[3][0][1]) / 2),
              int((corners[0][0][0] + corners[3][0][0]) / 2))
    for ratio in points:

        center_top.append(
            (center[0],
             int(corners[0][0][0] * (1 - ratio) + corners[1][0][0] * ratio)))

        center_bottom.append(
            (center[0],
             int(corners[2][0][0] * (1 - ratio) + corners[3][0][0] * ratio)))

        center_left.append(
            (int(corners[0][0][1] * (1 - ratio) + corners[2][0][1] * ratio),
             center[1]))

        center_right.append(
            (int(corners[1][0][1] * (1 - ratio) + corners[3][0][1] * ratio),
             center[1]))

    for i in range(0, center[0], 1):
        for j in range(num_middle_points):
            if top[j].size == 0:
                if mask[i, center_top[j][1]] == 255:
                    top[j] = np.asarray([center_top[j][1], i])
                    top[j] = top[j].reshape(1, 2)

    for i in range(mask.shape[0] - 1, center[0], -1):
        for j in range(num_middle_points):
            if bottom[j].size == 0:
                if mask[i, center_bottom[j][1]] == 255:
                    bottom[j] = np.asarray([center_bottom[j][1], i])
                    bottom[j] = bottom[j].reshape(1, 2)

    for i in range(mask.shape[1] - 1, center[1], -1):
        for j in range(num_middle_points):
            if right[j].size == 0:
                if mask[center_right[j][0], i] == 255:
                    right[j] = np.asarray([i, center_right[j][0]])
                    right[j] = right[j].reshape(1, 2)

    for i in range(0, center[1]):
        for j in range(num_middle_points):
            if left[j].size == 0:
                if mask[center_left[j][0], i] == 255:
                    left[j] = np.asarray([i, center_left[j][0]])
                    left[j] = left[j].reshape(1, 2)

    return np.asarray(top + bottom + left + right)


def findMiddlev2(corners, mask, points=[0.25, 0.5, 0.75]):

    num_middle_points = len(points)

    top = [np.array([])] * num_middle_points
    bottom = [np.array([])] * num_middle_points
    left = [np.array([])] * num_middle_points
    right = [np.array([])] * num_middle_points

    center_top = []
    center_bottom = []
    center_left = []
    center_right = []

    center = (int((corners[0][0][1] + corners[3][0][1]) / 2),
              int((corners[0][0][0] + corners[3][0][0]) / 2))
    for ratio in points:

        center_top.append(
            (center[0],
             int(corners[0][0][0] * (1 - ratio) + corners[1][0][0] * ratio)))

        center_bottom.append(
            (center[0],
             int(corners[2][0][0] * (1 - ratio) + corners[3][0][0] * ratio)))

        center_left.append(
            (int(corners[0][0][1] * (1 - ratio) + corners[2][0][1] * ratio),
             center[1]))

        center_right.append(
            (int(corners[1][0][1] * (1 - ratio) + corners[3][0][1] * ratio),
             center[1]))

    for j in range(num_middle_points):
        if top[j].size == 0:
            top_index = np.nonzero(mask[:, center_top[j][1]])[0].min()
            top[j] = np.asarray([center_top[j][1], top_index])
            top[j] = top[j].reshape(1, 2)

        if bottom[j].size == 0:
            bottom_index = np.nonzero(mask[:, center_bottom[j][1]])[0].max()
            bottom[j] = np.asarray([center_bottom[j][1], bottom_index])
            bottom[j] = bottom[j].reshape(1, 2)

        if right[j].size == 0:
            right_index = np.nonzero(mask[center_right[j][0], :])[0].max()
            right[j] = np.asarray([right_index, center_right[j][0]])
            right[j] = right[j].reshape(1, 2)

        if left[j].size == 0:
            left_index = np.nonzero(mask[center_left[j][0], :])[0].min()
            left[j] = np.asarray([left_index, center_left[j][0]])
            left[j] = left[j].reshape(1, 2)

    return np.asarray(top + bottom + left + right)


def DP_algorithmv1(contours):
    biggest = np.array([])
    max_area = 0
    step = 0.005
    count = 0
    # while biggest.size==0:
    while True:
        for i in contours:
            # print(i.shape)
            area = cv2.contourArea(i)
            # print(area,cv2.arcLength(i, True))
            if area > cv2.arcLength(i, True) * 10:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, (0.01 + step * count) * peri,
                                          True)
                if area > max_area and len(approx) == 4:
                    max_area = area
                    biggest_contours = i
                    biggest = approx
                    if abs(max_area -
                           cv2.contourArea(biggest)) / max_area > 0.3:
                        biggest = np.array([])
        count += 1
        if count > 200:
            break
    temp = biggest[0]
    return biggest, max_area, biggest_contours


def DP_algorithm(contours):
    biggest = np.array([])
    max_area = 0
    step = 0.005
    count = 0

    ### largest contours
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            biggest_contours = i
    peri = cv2.arcLength(biggest_contours, True)

    ### find four corners
    while True:
        approx = cv2.approxPolyDP(biggest_contours,
                                  (0.01 + step * count) * peri, True)
        if len(approx) == 4:
            biggest = approx
            break
            if abs(max_area - cv2.contourArea(biggest)) / max_area > 0.2:
                # if abs(max_area - cv2.contourArea(biggest))/max_area > 0.4:
                biggest = np.array([])
        count += 1
        if count > 200:
            break
    return biggest, max_area, biggest_contours


def drawRectangle(img, biggest, color, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    return img


def minAreaRect(contours, img):
    # biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            peri = cv2.arcLength(i, True)
            rect = cv2.minAreaRect(i)
            points = cv2.boxPoints(rect)
            max_area = area
    return points


def cropRectangle(img, biggest):
    # print(biggest)
    w = np.abs(biggest[0][0][0] - biggest[1][0][0])
    h = np.abs(biggest[0][0][1] - biggest[2][0][1])
    new_img = np.zeros((w, h, img.shape[-1]), dtype=np.uint8)
    new_img = img[biggest[0][0][1]:biggest[0][0][1] + h,
                  biggest[0][0][0]:biggest[0][0][0] + w]
    return new_img


def cvimg2torch(img, min=0, max=1):
    '''
    input:
        im -> ndarray uint8 HxWxC 
    return
        tensor -> torch.tensor BxCxHxW 
    '''
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)  # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img


def torch2cvimg(tensor, min=0, max=1):
    '''
    input:
        tensor -> torch.tensor BxCxHxW C can be 1,3
    return
        im -> ndarray uint8 HxWxC 
    '''
    im_list = []
    for i in range(tensor.shape[0]):
        im = tensor.detach().cpu().data.numpy()[i]
        im = im.transpose(1, 2, 0)
        im = np.clip(im, min, max)
        im = ((im - min) / (max - min) * 255).astype(np.uint8)
        im_list.append(im)
    return im_list


class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        '''
        target_control_points -> torch.tensor  num_pointx2 -1~1
        source_control_points -> torch.tensor  batch_size x num_point x 2 -1~1
        return:
            grid -> batch_size x hw x 2 -1~1
        '''
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(
            itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y],
                                      dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = self.compute_partial_repr(
            target_coordinate.to(target_control_points.device),
            target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr,
            torch.ones(HW, 1), target_coordinate
        ],
                                           dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([
            source_control_points,
            Variable(self.padding_matrix.expand(batch_size, 3, 2))
        ], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(
            Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate

    # phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(
            1, M, 2)
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :,
                                             0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix


class WarpTPS(nn.Module):
    '''
    Warps image using TPS
    
    Note)
    Thin plate spline surface height f is defined for every (x,y),
    f(x,y) = a[0] + a[1]*x + a[2]*y + \sum w_i*U(|P_i - (x,y)|_2)
    
    We construct two thin plates, dx plate and dy plate.
    Each plate contains dx values and dy values as height component.
    
    Args)
    normalize  		if true, normalize coordinates of control points
                to [-1, 1]
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, im, pts_before, pts_after):
        '''
        Deforms image according to movement of pts_before and pts_after
        Args)
        im		torch.Tensor object of size NxCxHxW
        pts_before	torch.Tensor object of size NxTx2 (T is # control pts)
        pts_after	torch.Tensor object of size NxTx2 (T is # control pts)
        '''
        # check input requirements
        assert (4 == im.dim())
        assert (3 == pts_after.dim())
        assert (3 == pts_before.dim())
        N = im.size()[0]
        assert (N == pts_after.size()[0] and N == pts_before.size()[0])
        assert (2 == pts_after.size()[2] and 2 == pts_before.size()[2])
        T = pts_after.size()[1]
        assert (T == pts_before.size()[1])
        H = im.size()[2]
        W = im.size()[3]

        if self.normalize:
            pts_after = pts_after.clone()
            pts_after[:, :, 0] /= 0.5 * W
            pts_after[:, :, 1] /= 0.5 * H
            pts_after -= 1
            pts_before = pts_before.clone()
            pts_before[:, :, 0] /= 0.5 * W
            pts_before[:, :, 1] /= 0.5 * H
            pts_before -= 1

        def construct_P():
            '''
            Consturcts matrix P of size NxTx3 where 
            P[n,i,0] := 1
            P[n,i,1:] := pts_after[n]
            '''
            # Create matrix P with same configuration as 'pts_after'
            P = pts_after.new_zeros((N, T, 3))
            P[:, :, 0] = 1
            P[:, :, 1:] = pts_after

            return P

        def calc_U(pt1, pt2):
            '''
            Calculate distance U between pt1 and pt2
            U(r) := r**2 * log(r)
            where
                r := |pt1 - pt2|_2
            Args)
            pt1 	torch.Tensor object, last dim is always 2
            pt2 	torch.Tensor object, last dim is always 2
            '''
            assert (2 == pt1.size()[-1])
            assert (2 == pt2.size()[-1])

            diff = pt1 - pt2
            sq_diff = diff**2
            sq_diff_sum = sq_diff.sum(-1)
            r = sq_diff_sum.sqrt()

            # Adds 1e-6 for numerical stability
            return (r**2) * torch.log(r + 1e-6)

        def construct_K():
            '''
            Consturcts matrix K of size NxTxT where 
            K[n,i,j] := U(|pts_after[n,i] - pts_after[n,j]|_2)
            '''

            # Assuming the number of control points are small enough,
            # We just use for-loop for easy-to-read code

            #   Create matrix K with same configuration as 'pts_after'
            K = pts_after.new_zeros((N, T, T))
            for i in range(T):
                for j in range(T):
                    K[:, i, j] = calc_U(pts_after[:, i, :], pts_after[:, j, :])

            return K

        def construct_L():
            '''
            Consturcts matrix L of size Nx(T+3)x(T+3) where 
            L[n] = [[ K[n]    P[n] ]]
                [[ P[n]^T    0  ]]
            '''
            P = construct_P()
            K = construct_K()

            # Create matrix L with same configuration as 'K'
            L = K.new_zeros((N, T + 3, T + 3))

            # Fill L matrix
            L[:, :T, :T] = K
            L[:, :T, T:(T + 3)] = P
            L[:, T:(T + 3), :T] = P.transpose(1, 2)

            return L

        def construct_uv_grid():
            '''
            Returns H x W x 2 tensor uv with UV coordinate as its elements
            uv[:,:,0] is H x W grid of x values
            uv[:,:,1] is H x W grid of y values
            '''
            u_range = torch.arange(
                start=-1.0, end=1.0, step=2.0 / W, device=im.device)
            assert (W == u_range.size()[0])
            u = u_range.new_zeros((H, W))
            u[:] = u_range

            v_range = torch.arange(
                start=-1.0, end=1.0, step=2.0 / H, device=im.device)
            assert (H == v_range.size()[0])
            vt = v_range.new_zeros((W, H))
            vt[:] = v_range
            v = vt.transpose(0, 1)

            return torch.stack([u, v], dim=2)

        L = construct_L()
        VT = pts_before.new_zeros((N, T + 3, 2))
        # Use delta x and delta y as known heights of the surface
        VT[:, :T, :] = pts_before - pts_after

        # Solve Lx = VT
        #   x is of shape (N, T+3, 2)
        #   x[:,:,0] represents surface parameters for dx surface
        #       (dx values as surface height (z))
        #   x[:,:,1] represents surface parameters for dy surface
        #       (dy values as surface height (z))
        # x, _ = torch.gesv(VT, L)
        x = torch.linalg.solve(L, VT)

        uv = construct_uv_grid()
        uv_batch = uv.repeat((N, 1, 1, 1))

        def calc_dxdy():
            '''
            Calculate surface height for each uv coordinate
            
            Returns NxHxWx2 tensor
            '''

            # control points of size NxTxHxWx2
            cp = uv.new_zeros((H, W, N, T, 2))
            cp[:, :, :] = pts_after
            cp = cp.permute([2, 3, 0, 1, 4])

            U = calc_U(uv, cp)  # U value matrix of size NxTxHxW
            w, a = x[:, :
                     T, :], x[:,
                              T:, :]  # w is of size NxTx2, a is of size Nx3x2
            w_x, w_y = w[:, :, 0], w[:, :, 1]  # NxT each
            a_x, a_y = a[:, :, 0], a[:, :, 1]  # Nx3 each
            dx = (a_x[:, 0].repeat((H, W, 1)).permute(2, 0, 1) +
                  torch.einsum('nhwd,nd->nhw', uv_batch, a_x[:, 1:]) +
                  torch.einsum('nthw,nt->nhw', U, w_x))  # dx values of NxHxW
            dy = (a_y[:, 0].repeat((H, W, 1)).permute(2, 0, 1) +
                  torch.einsum('nhwd,nd->nhw', uv_batch, a_y[:, 1:]) +
                  torch.einsum('nthw,nt->nhw', U, w_y))  # dy values of NxHxW

            return torch.stack([dx, dy], dim=3)

        dxdy = calc_dxdy()
        flow_field = uv + dxdy

        return flow_field
