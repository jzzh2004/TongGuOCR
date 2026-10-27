import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .MBD_utils import (DP_algorithm, WarpTPS, cvimg2torch, findMiddlev2,
                        reorder, torch2cvimg)


class Dewarper(object):

    def __init__(self):
        pass

    def dewarp(self, im):
        org_h, org_w = im.shape[:2]

        is_filter, mask = self.filtering(im)

        grid = None
        if is_filter:
            dewarped = im
        else:
            try:
                dewarped, grid = self.mask_base_dewarper(im, mask)
            except:
                dewarped = im
        return dewarped, grid, org_w, org_h

    def poly_recovery(self, old_points_batch, grid, org_w, org_h):
        if grid is not None:
            new_points_batch = []
            for old_points in old_points_batch:
                old_points = np.array(old_points).reshape(-1, 2)
                new_points = self.point_transform_batch_accelorate_cuda_fm(
                    old_points, grid, org_h, org_w)
                new_points = new_points.reshape(-1).tolist()
                new_points_batch.append(new_points)
            return new_points_batch
        else:
            return old_points_batch

    def filtering(self, im):
        h, w = im.shape[:2]
        size = 512
        half_size = int(size / 2)
        im = cv2.resize(im, (size, size))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        ## get value mask
        value_mask = np.where(im_gray == 0, 0, 1)
        binary_value_mask = cv2.copyMakeBorder(
            value_mask,
            half_size,
            half_size,
            half_size,
            half_size,
            value=0,
            borderType=cv2.BORDER_CONSTANT)
        value_mask = (binary_value_mask * 255).astype(np.uint8)

        ## noise remove
        kernel = np.ones((3, 3))
        value_mask = cv2.erode(value_mask, kernel, iterations=4)
        value_mask = cv2.dilate(value_mask, kernel, iterations=4)
        value_mask = cv2.dilate(value_mask, kernel, iterations=4)
        value_mask = cv2.erode(value_mask, kernel, iterations=4)

        ## filter based on value element num
        binary_value_mask = np.where(value_mask > 100, 1, 0)
        total_element_num = size * size
        value_element_num = np.sum(binary_value_mask)
        if value_element_num > total_element_num * 0.95:
            return True, value_mask

        ## filter based on contours num
        binary_value_mask = np.where(value_mask > 100, 255, 0)
        # _, contours, hierarchy = cv2.findContours(binary_value_mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)  ## cv2.__version__ == 3.x
        contours, hierarchy = cv2.findContours(
            binary_value_mask.astype(np.uint8), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE)  ## cv2.__version__ == 4.x
        binary_value_mask_3_channel = np.stack(
            (binary_value_mask, binary_value_mask, binary_value_mask), -1)
        show_contours = cv2.drawContours(
            binary_value_mask_3_channel.astype(np.uint8), contours, -1,
            (0, 0, 255), 3)
        if len(contours) > 1:
            return True, show_contours

        # binary_value_mask_3_channel = np.stack((binary_value_mask,binary_value_mask,binary_value_mask),-1)
        # show_contours = cv2.drawContours(binary_value_mask_3_channel.astype(np.uint8),contours,-1,(0,0,255),3)
        # cv2.imshow('value_mask',value_mask)
        # cv2.imshow('im',im)
        # cv2.imshow('show_contours',show_contours)
        # cv2.waitKey(0)

        value_mask = value_mask[half_size:-half_size, half_size:-half_size]
        value_mask = cv2.resize(value_mask, (w, h))
        value_mask = np.where(value_mask > 100, 255, 0).astype(np.uint8)
        return False, value_mask

    def point_transform_batch_accelorate_cuda_fm(self, points_list,
                                                 new_coordinate, org_h, org_w):
        size = 512
        '''
        forward dewarping 
        input:
            points_list -> [(w0,h0),(w1,h1)...], a list consists of absolute coordinates
            new_coordinate -> torch.Tensor, (1,h,w,2)
        outputL
            new_points -> [(w0,h0),(w1,h1)...], a list consists of new absolute coordinates
        '''
        points = np.asarray(points_list).reshape(-1, 2)
        points = points / np.asarray((org_w, org_h)) * np.asarray((size, size))
        points = points.astype(int)
        points_num = points.shape[0]
        points = torch.from_numpy(points).cuda()

        new_coordinate = F.interpolate(
            new_coordinate.permute(0, 3, 1, 2), (size, size))
        new_coordinate = new_coordinate.permute(0, 2, 3, 1)[0]
        new_coordinate = (new_coordinate + 1) / 2 * (
            torch.Tensor((org_w, org_h)).view(-1, 2).cuda())

        new_points_array = new_coordinate[points[:, 1], points[:, 0]]
        new_points_array = new_points_array.cpu().numpy().astype(int)
        return new_points_array

    def mask_base_dewarper(self, image, mask):
        size = 1024
        org_h, org_w = image.shape[:2]
        image_org = image.copy()
        image = cv2.resize(image, (size, size))
        mask = cv2.resize(mask, (size, size))
        '''
        input:
            image -> ndarray HxWx3 uint8
            mask -> ndarray HxW uint8
        return
            dewarped -> ndarray HxWx3 uint8
            grid (optional) -> ndarray HxWx2 -1~1
        '''

        ## get contours
        # _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  ## cv2.__version__ == 3.x
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)  ## cv2.__version__ == 4.x

        ## get biggest contours and four corners based on Douglas-Peucker algorithm
        four_corners, maxArea, contour = DP_algorithm(contours)
        four_corners = reorder(four_corners)

        ## reserve biggest contours and remove other noisy contours
        new_mask = np.zeros_like(mask)
        new_mask = cv2.drawContours(new_mask, [contour], -1, 255, cv2.FILLED)

        ## obtain middle points
        # ratios = [0.25,0.5,0.75]  # ratios = [0.125,0.25,0.375,0.5,0.625,0.75,0.875]
        ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        middle = findMiddlev2(
            corners=four_corners, mask=new_mask, points=ratios)

        ## all points
        source_points = np.concatenate(
            (four_corners, middle), axis=0
        )  ## all_point = four_corners(topleft,topright,bottom)+top+bottom+left+right

        ## target points
        h, w = image.shape[:2]
        padding = 0
        target_points = [[padding, padding], [w - padding, padding],
                         [padding, h - padding], [w - padding, h - padding]]
        for ratio in ratios:
            target_points.append(
                [int((w - 2 * padding) * ratio) + padding, padding])
        for ratio in ratios:
            target_points.append(
                [int((w - 2 * padding) * ratio) + padding, h - padding])
        for ratio in ratios:
            target_points.append(
                [padding, int((h - 2 * padding) * ratio) + padding])
        for ratio in ratios:
            target_points.append(
                [w - padding,
                 int((h - 2 * padding) * ratio) + padding])

        ## dewarp base on generated grid
        source_points = np.asarray(source_points).reshape(-1, 2) / np.array(
            [image.shape[:2][::-1]]).reshape(1, 2)
        source_points = torch.from_numpy(source_points).float().cuda()
        source_points = source_points.unsqueeze(0)
        source_points = (source_points - 0.5) * 2
        target_points = np.asarray(target_points).reshape(-1, 2) / np.array(
            [image.shape[:2][::-1]]).reshape(1, 2)
        target_points = torch.from_numpy(target_points).float().cuda()
        target_points = (target_points - 0.5) * 2
        target_points = target_points.unsqueeze(0)
        model = WarpTPS()
        grid = model(
            cvimg2torch(cv2.resize(image, (256, 256))).cuda(), source_points,
            target_points)
        grid = F.interpolate(
            grid.permute(0, 3, 1, 2), (org_h, org_w),
            mode='bilinear').permute(0, 2, 3, 1)
        dewarped = torch2cvimg(
            F.grid_sample(cvimg2torch(image_org).cuda(), grid))[0].copy()

        # return dewarped,grid[0].cpu().numpy()
        return dewarped, grid

    def mask_base_cropper(self, image, mask):
        '''
        input:
            image -> ndarray HxWx3 uint8
            mask -> ndarray HxW uint8
        return
            dewarped -> ndarray HxWx3 uint8
            grid (optional) -> ndarray HxWx2 -1~1
        '''

        ## get contours
        _, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)  ## cv2.__version__ == 3.x
        # contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)  ## cv2.__version__ == 4.x

        ## get biggest contours and four corners based on Douglas-Peucker algorithm
        four_corners, maxArea, contour = DP_algorithm(contours)
        four_corners = reorder(four_corners)

        ## reserve biggest contours and remove other noisy contours
        new_mask = np.zeros_like(mask)
        new_mask = cv2.drawContours(new_mask, [contour], -1, 255, cv2.FILLED)

        ## 最小外接矩形
        rect = cv2.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        box = cv2.boxPoints(
            rect)  # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点坐标
        box = np.intp(box)
        box = box.reshape((4, 1, 2))
