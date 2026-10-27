import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TPS(nn.Module):

    def __init__(self, shortside_grid_length, eps=1e-9):
        super().__init__()
        self.shortside_grid_length = shortside_grid_length
        self.eps = eps

    def forward(self, src_batch, X, Y, w, h):
        device = X.device
        n, k = X.shape[:2]
        if w > h:
            grid_h = self.shortside_grid_length
            grid_w = int(w / h * self.shortside_grid_length)
        else:
            grid_w = self.shortside_grid_length
            grid_h = int(h / w * self.shortside_grid_length)
        grid = torch.ones(n, grid_h, grid_w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, grid_w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, grid_h)[..., None]
        grid = grid.view(-1, grid_h * grid_w, 2)
        Z = torch.zeros(n, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + self.eps)
        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)
        Q = torch.linalg.solve(L, Z)
        W, A = Q[:, :k], Q[:, k:]
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + self.eps)
        n, k = grid.shape[:2]
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        grid = grid.view(-1, grid_h, grid_w, 2)
        grid = F.interpolate(
            grid.permute(0, 3, 1, 2), (h, w),
            mode="bilinear",
            align_corners=True).permute(0, 2, 3, 1)
        dst_batch = F.grid_sample(
            src_batch, grid, padding_mode="zeros", align_corners=True)
        return dst_batch


class Connector(object):

    def __init__(self,
                 device,
                 side_point_num=8,
                 cort_seg_num=8,
                 no_rect_min_size=128,
                 seg_area_padding=16,
                 height=96,
                 min_width=64,
                 max_width=1536,
                 batch_size=16,
                 is_static=False):
        self.device = device
        assert side_point_num >= cort_seg_num
        self.side_point_num = side_point_num
        self.cort_seg_num = cort_seg_num
        self.no_rect_min_size = no_rect_min_size
        self.seg_area_padding = seg_area_padding
        self.height = height
        self.min_width = min_width
        self.max_width = max_width
        self.batch_size = batch_size
        self.is_static = is_static

        self.tps = TPS(shortside_grid_length=height)

    def __call__(self, img_raw, det_polys, det_segs):
        img = img_raw[:, :, [2, 1, 0]]
        img = self.img_numpy2tensor(img, self.device)
        h, w = img.shape[1:]
        leftside_points_list, rightside_points_list = self.multi_seg2bbox(
            det_polys, det_segs)
        textline_img_list = []
        points_list = []
        for leftside_points, rightside_points in zip(leftside_points_list,
                                                     rightside_points_list):
            src_pts, dst_pts, width, height = self.get_src_dst_pts(
                leftside_points, rightside_points)
            dst_img = self.tps_get_textline_img(img, src_pts, dst_pts, width,
                                                height)
            if dst_img is None:
                continue
            textline_img_list.append(dst_img)
            points_list.append(
                np.intp(
                    np.concatenate([
                        leftside_points[-1:], rightside_points,
                        leftside_points[:-1]
                    ],
                                   axis=0).reshape(-1)).tolist())
        all_textline_img_batch, all_textline_img_length_batch, all_points_batch = self.collate_batch(
            textline_img_list, points_list)
        return all_textline_img_batch, all_textline_img_length_batch, all_points_batch, w, h

    def img_numpy2tensor(self, img, device):
        if img.ndim == 2:
            img = img[:, :, None]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).type(torch.FloatTensor).to(device)
        img = img / 127.5 - 1.
        return img

    def multi_seg2bbox(self, polys, segs):
        leftside_points_list = []
        rightside_points_list = []
        for poly, seg in zip(polys, segs):
            leftside_points, rightside_points = self.seg2bbox(poly, seg)
            leftside_points_list.append(leftside_points)
            rightside_points_list.append(rightside_points)
        return leftside_points_list, rightside_points_list

    def tps_get_textline_img_batch(self, img, src_pts_list, dst_pts_list,
                                   width_list, height_list, minx_list,
                                   maxx_list, miny_list, maxy_list):
        ws = torch.tensor(maxx_list) - torch.tensor(minx_list)
        hs = torch.tensor(maxy_list) - torch.tensor(miny_list)
        maxw, maxh = torch.max(ws).item(), torch.max(hs).item()
        src_img_batch = torch.zeros(
            (len(src_pts_list), img.shape[0], int(maxh), int(maxw)),
            dtype=img.dtype,
            device=img.device)
        src_pts_list_new = []
        dst_pts_list_new = []
        for batch_i, (minx, maxx, miny, maxy, w, h, src_pts,
                      dst_pts) in enumerate(
                          zip(minx_list, maxx_list, miny_list, maxy_list, ws,
                              hs, src_pts_list, dst_pts_list)):
            w, h = w.item(), h.item()
            src_img_batch[
                batch_i, :, :int(h), :int(w)] = img[:,
                                                    int(miny):int(maxy),
                                                    int(minx):int(maxx)]
            src_pts[..., 0] = 2. * (src_pts[..., 0] - minx) / maxw - 1.
            src_pts[..., 1] = 2. * (src_pts[..., 1] - miny) / maxh - 1.
            src_pts_list_new.append(src_pts)
            dst_pts[..., 0] = 2. * dst_pts[..., 0] / maxw - 1.
            dst_pts[..., 1] = 2. * dst_pts[..., 1] / maxh - 1.
            dst_pts_list_new.append(dst_pts)
        src_pts_batch = torch.cat(src_pts_list_new, dim=0).to(img.device)
        dst_pts_batch = torch.cat(dst_pts_list_new, dim=0).to(img.device)
        dst_img_batch = self.tps(src_img_batch, dst_pts_batch, src_pts_batch,
                                 int(maxw), int(maxh))
        textline_img_list = []
        for dst_img, width, height in zip(dst_img_batch, width_list,
                                          height_list):
            textline_img_list.append(dst_img[:, :height, :width].cpu())
        return textline_img_list

    def is_clockwise(self, points):
        points_temp = np.concatenate([points[1:], points[:1]], axis=0)
        area = np.sum(points[:, 0] * points_temp[:, 1] -
                      points[:, 1] * points_temp[:, 0])
        return area > 0

    def seg2bbox(self, poly, seg):
        h, w = seg.shape
        poly = np.intp(poly.reshape(-1, 2))
        minx, maxx = np.min(poly[:, 0]) - self.seg_area_padding, np.max(
            poly[:, 0]) + self.seg_area_padding
        miny, maxy = np.min(poly[:, 1]) - self.seg_area_padding, np.max(
            poly[:, 1]) + self.seg_area_padding
        minx, miny = max(minx, 0), max(miny, 0)
        maxx, maxy = min(maxx, w), min(maxy, h)
        seg = seg[miny:maxy, minx:maxx].astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        max_area_index = 0
        max_area = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area
                max_area_index = i
        points = contours[max_area_index][:, 0, :]
        points[:, 0] += minx
        points[:, 1] += miny
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0]) + 1
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1]) + 1
        if maxx - minx < self.no_rect_min_size and maxy - miny < self.no_rect_min_size:
            return self.get_minarea_rect_bbox(points)
        grid_ys = np.intp(
            np.linspace(miny, maxy, self.side_point_num, endpoint=False))
        leftside_points = []
        rightside_points = []
        for grid_y in grid_ys:
            y_indices = np.where(points[:, 1] == grid_y)[0]
            y_xs = np.sort(points[y_indices, 0])
            if len(y_xs) < 2:
                continue
            else:
                leftside_points.append([y_xs[0], grid_y])
                rightside_points.append([y_xs[-1], grid_y])
        if len(leftside_points) == 0:
            return self.get_minarea_rect_bbox(points)
        # cut_num_a = 1
        # cut_num_b = 2
        cut_num_a = int(self.side_point_num / self.cort_seg_num * 1.)
        cut_num_b = int(self.side_point_num / self.cort_seg_num * 2.)
        leftside_points = np.array(leftside_points)
        rightside_points = np.array(rightside_points)
        top_vector_left = leftside_points[cut_num_b] - leftside_points[
            cut_num_a]
        top_vector_right = rightside_points[cut_num_b] - rightside_points[
            cut_num_a]
        top_vector = top_vector_left / np.linalg.norm(
            top_vector_left) + top_vector_right / np.linalg.norm(
                top_vector_right)
        if (top_vector == 0).any():
            lt_point = np.array([[leftside_points[cut_num_a, 0], miny]])
            rt_point = np.array([[rightside_points[cut_num_a, 0], miny]])
        else:
            top_slope = top_vector[1] / top_vector[0]
            if top_slope == 0:
                lt_point = np.array([[minx, leftside_points[cut_num_a, 1]]])
                rt_point = np.array([[minx, rightside_points[cut_num_a, 1]]])
            else:
                top_left_y, top_right_y = self.get_top_vertex_y(
                    miny, top_slope, rightside_points[cut_num_a, 0] -
                    leftside_points[cut_num_a, 0])
                lt_point = np.array([[
                    (top_left_y - leftside_points[cut_num_a, 1]) / top_slope +
                    leftside_points[cut_num_a, 0], top_left_y
                ]])
                rt_point = np.array([[
                    (top_right_y - rightside_points[cut_num_a, 1]) / top_slope
                    + rightside_points[cut_num_a, 0], top_right_y
                ]])
        bottom_vector_left = leftside_points[-cut_num_b] - leftside_points[
            -cut_num_a]
        bottom_vector_right = rightside_points[-cut_num_b] - rightside_points[
            -cut_num_a]
        bottom_vector = bottom_vector_left / np.linalg.norm(
            bottom_vector_left) + bottom_vector_right / np.linalg.norm(
                bottom_vector_right)
        if (bottom_vector == 0).any():
            lb_point = np.array([[leftside_points[-cut_num_a, 0], maxy]])
            rb_point = np.array([[rightside_points[-cut_num_a, 0], maxy]])
        else:
            bottom_slope = bottom_vector[1] / bottom_vector[0]
            if bottom_slope == 0:
                lb_point = np.array([[maxx, leftside_points[-cut_num_a, 1]]])
                rb_point = np.array([[maxx, rightside_points[-cut_num_a, 1]]])
            else:
                bottom_left_y, bottom_right_y = self.get_top_vertex_y(
                    maxy, bottom_slope, rightside_points[-cut_num_a, 0] -
                    leftside_points[-cut_num_a, 0])
                lb_point = np.array([[
                    (bottom_left_y - leftside_points[-cut_num_a, 1]) /
                    bottom_slope + leftside_points[-cut_num_a, 0],
                    bottom_left_y
                ]])
                rb_point = np.array([[
                    (bottom_right_y - rightside_points[-cut_num_a, 1]) /
                    bottom_slope + rightside_points[-cut_num_a, 0],
                    bottom_right_y
                ]])
        leftside_points_interped = leftside_points[cut_num_a:-cut_num_a]
        leftside_points_interped = leftside_points_interped[np.intp(
            np.linspace(
                0, leftside_points_interped.shape[0] - 1,
                min(self.side_point_num - 2,
                    leftside_points_interped.shape[0])))]
        rightside_points_interped = rightside_points[cut_num_a:-cut_num_a]
        rightside_points_interped = rightside_points_interped[np.intp(
            np.linspace(
                0, rightside_points_interped.shape[0] - 1,
                min(self.side_point_num - 2,
                    rightside_points_interped.shape[0])))]
        leftside_points_new = np.intp(
            np.concatenate(
                [lb_point, leftside_points_interped[::-1], lt_point], axis=0))
        rightside_points_new = np.intp(
            np.concatenate([rt_point, rightside_points_interped, rb_point],
                           axis=0))
        leftside_points_new[:, 0] = np.clip(
            leftside_points_new[:, 0], a_min=0, a_max=w - 1)
        leftside_points_new[:, 1] = np.clip(
            leftside_points_new[:, 1], a_min=0, a_max=h - 1)
        rightside_points_new[:, 0] = np.clip(
            rightside_points_new[:, 0], a_min=0, a_max=w - 1)
        rightside_points_new[:, 1] = np.clip(
            rightside_points_new[:, 1], a_min=0, a_max=h - 1)
        return leftside_points_new, rightside_points_new

    def get_src_dst_pts(self, leftside_points, rightside_points):
        leftside_pts = torch.from_numpy(leftside_points).type(
            torch.FloatTensor)
        rightside_pts = torch.from_numpy(rightside_points).type(
            torch.FloatTensor)
        raw_width = max(
            torch.linalg.norm(rightside_pts[0] - leftside_pts[-1]).item(),
            torch.linalg.norm(rightside_pts[-1] - leftside_pts[0]).item())
        rightside_vectors = torch.linalg.norm(
            rightside_pts[1:] - rightside_pts[:-1], dim=1)
        leftside_vectors = torch.linalg.norm(
            leftside_pts[1:] - leftside_pts[:-1], dim=1)
        raw_height = max(
            torch.sum(rightside_vectors).item(),
            torch.sum(leftside_vectors).item())
        width = self.height
        height = min(
            max(int(raw_height / raw_width * width), self.min_width),
            self.max_width)
        rightside_cumsum = torch.cat(
            [torch.tensor([0.]),
             torch.cumsum(rightside_vectors, dim=0)])
        rightside_cumsum /= rightside_cumsum[-1].item()
        rightside_dst_pts = torch.cat([
            rightside_cumsum[:, None] * height,
            torch.zeros((rightside_pts.shape[0], 1)),
        ],
                                      dim=1)
        leftside_cumsum = torch.cat(
            [torch.tensor([0.]),
             torch.cumsum(leftside_vectors, dim=0)])
        leftside_cumsum /= leftside_cumsum[-1].item()
        leftside_dst_pts = torch.cat([
            height - leftside_cumsum[:, None] * height,
            torch.ones((leftside_pts.shape[0], 1)) * width
        ],
                                     dim=1)
        src_pts = torch.cat([rightside_pts, leftside_pts], dim=0)
        dst_pts = torch.cat([rightside_dst_pts, leftside_dst_pts], dim=0)
        return src_pts, dst_pts, height, width

    def tps_get_textline_img(self, img, src_pts, dst_pts, width, height):
        if width <= 0 or height <= 0:
            return None
        minx, maxx, miny, maxy = int(torch.min(
            src_pts[:,
                    0]).item()), int(torch.max(src_pts[:, 0]).item()) + 1, int(
                        torch.min(src_pts[:, 1]).item()), int(
                            torch.max(src_pts[:, 1]).item()) + 1
        w, h = max(maxx - minx, width), max(maxy - miny, height)
        if w <= 0 or h <= 0:
            return None
        src_img = torch.zeros((img.shape[0], h, w),
                              dtype=img.dtype,
                              device=img.device)
        try:
            src_img[:, :maxy - miny, :maxx - minx] = img[:, miny:maxy,
                                                         minx:maxx]
        except Exception as e:
            print(e.__class__.__name__, e)
            return None
        src_pts[:, 0] = 2. * (src_pts[:, 0] - minx) / w - 1.
        src_pts[:, 1] = 2. * (src_pts[:, 1] - miny) / h - 1.
        dst_pts[:, 0] = 2. * dst_pts[:, 0] / w - 1.
        dst_pts[:, 1] = 2. * dst_pts[:, 1] / h - 1.
        src_pts = src_pts.to(img.device)
        dst_pts = dst_pts.to(img.device)
        try:
            dst_img = self.tps(src_img[None, ...], dst_pts[None, ...],
                               src_pts[None, ...], w, h)[0, :, :height, :width]
        except Exception as e:
            print(e.__class__.__name__, e, w, h, width, height)
            return None
        return dst_img

    def collate_batch(self, textline_img_list, points_list):
        textline_img_list_temp = []
        textline_img_length_list = []
        for textline_img in textline_img_list:
            c, h, w = textline_img.shape
            img_padding = torch.ones((1, c, h, self.max_width),
                                     dtype=textline_img.dtype,
                                     device=textline_img.device) * -1.
            img_padding[0, :, :, :w] = textline_img
            textline_img_list_temp.append(img_padding)
            textline_img_length_list.append(w / self.max_width)
        batch_num = math.ceil(len(textline_img_list) / self.batch_size)
        all_textline_img_batch = [
            torch.cat(textline_img_list_temp[i::batch_num], dim=0)
            for i in range(batch_num)
        ]
        all_textline_img_length_batch = [
            textline_img_length_list[i::batch_num] for i in range(batch_num)
        ]
        all_points_batch = [
            points_list[i::batch_num] for i in range(batch_num)
        ]
        if not self.is_static:
            for pindex in range(len(all_textline_img_batch)):
                max_length = int(
                    max(all_textline_img_length_batch[pindex]) *
                    self.max_width)
                all_textline_img_batch[pindex] = all_textline_img_batch[
                    pindex][..., :max_length]
                all_textline_img_length_batch[pindex] = (np.array(all_textline_img_length_batch[pindex]) *
                    self.max_width / max_length).tolist()
        return all_textline_img_batch, all_textline_img_length_batch, all_points_batch

    def get_minarea_rect_bbox(self, points):
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])
        rect = cv2.minAreaRect(points)
        box = np.intp(cv2.boxPoints(rect))
        lt_point_index = np.argmin(
            np.linalg.norm(box - np.array([[minx, miny]]), axis=1))
        points = np.concatenate([box[lt_point_index:], box[:lt_point_index]],
                                axis=0)
        leftside_points = np.array([points[3], points[0]])
        rightside_points = np.array([points[1], points[2]])
        return leftside_points, rightside_points

    def get_top_vertex_y(self, boundary_y, slope, x_width):
        theta = np.arctan(slope)
        if theta < 0:
            theta = -theta
        right_y = boundary_y
        left_y = boundary_y + x_width * np.cos(theta) * np.sin(theta)
        if slope < 0:
            left_y, right_y = right_y, left_y
        return left_y, right_y
