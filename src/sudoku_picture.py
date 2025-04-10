import numpy as np
import cv2
import os
import torch.nn as nn
import scipy.interpolate

from .entry_detection.entry_detection import EntryDetector
from .entry_detection.resnet import ResNet
from .dlx import Sudoku
import numba as nb

@nb.njit   
def uncrop(picture, unwarped, offset_x, offset_y):
        
    for i in range(unwarped.shape[0]):
        for j in range(unwarped.shape[1]):
            if unwarped[i, j] <= 150:
                picture[i + offset_y, j + offset_x, 0] = 0
                picture[i + offset_y, j + offset_x, 1] = 150
                picture[i + offset_y, j + offset_x, 2] = 0
    
    return picture

def find_intersection(a: np.ndarray, index_a: int, b: np.ndarray, index_b: int, offset: int=200):
    # Speed up computation of intersection of vertical and horizontal line since we roughly know where the lines are (we ranked them)    
    result = np.bitwise_and(a[0 if index_a - offset < 0 else index_a - offset: a.shape[0] if index_a + offset + 1 > a.shape[0] else index_a + offset, 0 if index_b - offset < 0 else index_b - offset: b.shape[0] if index_b + offset + 1 > b.shape[0] else index_b + offset], b[0 if index_a - offset < 0 else index_a - offset: a.shape[0] if index_a + offset + 1 > a.shape[0] else index_a + offset, 0 if index_b - offset < 0 else index_b - offset: b.shape[0] if index_b + offset + 1 > b.shape[0] else index_b + offset])
    moments = cv2.moments(result)
    if moments["m00"] != 0:
        return int(moments["m10"] / moments["m00"]) + (0 if index_b - offset < 0 else index_b - offset), int(moments["m01"] / moments["m00"]) + (0 if index_a - offset < 0 else index_a - offset)
    return -1, -1
            

class SudokuPicture():
    
    def __init__(self) -> None:
        
        self.entries = {i: cv2.cvtColor(cv2.imread(f"./data/entries/{i}.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2GRAY) for i in "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        
        self.characters = "123456789ABCDEFG"
            
        self.path = os.getcwd()
        
        channels = [64, 64, 128, 256, 512]
        layers = [3, 4, 6, 4]
        expansion = 4
        first_layer = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)]
        padding = 1
        stride = 2
        bottleneck_patch_size = 4
        h_bottleneck = 512
        loss = nn.CrossEntropyLoss()
        h = 16
        checkpoint = f'./tb_logs/sudoku/resnet_box_{h}/version_0/checkpoints/'
        for file in os.listdir(checkpoint):
            if file.startswith("epoch"):
                checkpoint += file
        model = ResNet.load_from_checkpoint(checkpoint_path=checkpoint, channels=channels, layers=layers, expansion=expansion, first_layer=first_layer, padding=padding, stride=stride, bottleneck_patch_size=bottleneck_patch_size, h_bottleneck=h_bottleneck, h=h, loss=loss)
        
        self.entry_detector = EntryDetector(model=model, characters=self.characters, width=28, height=28)
    
    def solve(self, picture):
        
        crop = self._crop_from_contour(picture)
        
        if crop is None:
            return None
        
        warped = self._detect_intersections_and_warp(crop)
        
        if warped is None:
            return None
        
        self.state = self._get_entries(warped)
        solver = Sudoku(self.state)
        result = solver.solve()
        solution = None
        for solution in result:
            break
        if solution is None:
            return "".join(self.state)
            
        filled = self._fill(warped, "".join(solution))
        unwarped = self._unwarp(filled)
        picture = uncrop(picture, unwarped, self.offset_x, self.offset_y)
        
        return picture
        
    def _crop_from_contour(self, picture):
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

        c, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(c, key=cv2.contourArea, reverse=True)[0]

        peri = cv2.arcLength(c, True)
        if peri < 2 * np.sum(gray.shape) / 8:
            return None
        
        self.safety_x = int(np.round(0.05 * gray.shape[1]))
        self.safety_y = int(np.round(0.05 * gray.shape[0]))
        self.offset_x = np.max([0, np.min(c[:, 0, 0]) - self.safety_x])
        self.offset_y = np.max([0, np.min(c[:, 0, 1]) - self.safety_y])
        offset_x2 = np.min([gray.shape[1], np.max(c[:, 0, 0]) + self.safety_x + 1])
        offset_y2 = np.min([gray.shape[0], np.max(c[:, 0, 1]) + self.safety_x + 1])
        
        mask = np.zeros(picture.shape[: 2], dtype=np.uint8)
        cv2.drawContours(mask, [c], 0, 255, cv2.FILLED)
        mask = mask[self.offset_y: offset_y2, self.offset_x: offset_x2]
        pic = np.copy(picture)[self.offset_y: offset_y2, self.offset_x: offset_x2, :]
        pic[mask == 0] = 255 * np.ones((3,))
        
        self.center, _, self.angle = cv2.minAreaRect(c)
        if self.angle >= 45:
            self.angle -= 90
        elif self.angle <= -45:
            self.angle += 90

        M = cv2.getRotationMatrix2D((self.center[0] - self.offset_y, self.center[1] - self.offset_x), self.angle, 1)        
        pic = cv2.warpAffine(pic, M, (pic.shape[1], pic.shape[0]), borderValue=(255, 255, 255))
        self.crop_shape = pic.shape
        return pic
    
    def _detect_vlines(self, gray):
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        height = gray.shape[1]
        
        thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(np.round(0.03 * height)))), iterations=1)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for c in cnts:
            if cv2.arcLength(c, True) > 1.5 * height:
                lines.append(c)
        
        lines.sort(key=lambda x: np.min(x[:, 0, 0]))
        
        min_x = [np.min(line[:, 0, 0]) for line in lines]
        
        for i in range(len(lines)):
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [lines[i]], 0, 255, cv2.FILLED)
            lines[i] = mask
            
        return lines, min_x
    
    def _detect_missing_lines(self, array, maximum):
        array = np.array(array, dtype=np.float32)
        array -= array[0]
        array /= array[-1]
        array *= maximum - 1
        result = np.zeros(array.shape[0], dtype=np.int32)
        array = np.diff(array)
        for i in range(1, array.shape[0] + 1):
            result[i] = int(np.round(result[i - 1] + array[i - 1]))
        return result
    
    def _detect_intersections_and_warp(self, picture):
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        vlines, min_v = self._detect_vlines(gray)
        hlines, min_h = self._detect_vlines(gray.T)
        
        if len(vlines) > 10 and len(hlines) > 10:
            self.number_of_rows = 16
            self.number_of_columns = 16
        else:
            self.number_of_rows = 9
            self.number_of_columns = 9
            
        if len(vlines) <= self.number_of_rows - 2 and len(hlines) <= self.number_of_columns - 2:
            return None
        
        vlines_numbers = self._detect_missing_lines(min_v, self.number_of_columns + 1)
        hlines_numbers = self._detect_missing_lines(min_h, self.number_of_rows + 1)
        
        pixels_x = int(np.round(picture.shape[0] / self.number_of_rows))
        pixels_y = int(np.round(picture.shape[1] / self.number_of_columns))
        
        intersections = []
        anchors = []
        for i, vline, mv in zip(vlines_numbers, vlines, min_v):
            for j, hline, mh in zip(hlines_numbers, hlines, min_h):
                x, y = find_intersection(hline.T, mh, vline, mv)
                if x != -1 and y != -1:
                    intersections.append((x, y))
                    anchors.append((pixels_x * i, pixels_y * j))
        
        self.anchors = np.array(anchors)
        self.intersections = np.array(intersections)
        
        picture = cv2.adaptiveThreshold(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 20)
        
        grid_x, grid_y = np.meshgrid(np.arange(self.number_of_columns * pixels_x + 1), np.arange(self.number_of_rows * pixels_y + 1))
        mapping_x = scipy.interpolate.SmoothBivariateSpline(self.anchors[:, 0], self.anchors[:, 1], self.intersections[:, 0])(grid_x, grid_y, grid=False)
        mapping_y = scipy.interpolate.SmoothBivariateSpline(self.anchors[:, 0], self.anchors[:, 1], self.intersections[:, 1])(grid_x, grid_y, grid=False)

        return cv2.remap(picture, np.float32(mapping_x), np.float32(mapping_y), cv2.INTER_LINEAR)
    
    def _compute_reverse_mapping(self):
        grid_x, grid_y = np.meshgrid(np.arange(self.crop_shape[1]), np.arange(self.crop_shape[0]))
        self.reverse_mapping_x = scipy.interpolate.SmoothBivariateSpline(self.intersections[:, 0], self.intersections[:, 1], self.anchors[:, 0])(grid_x, grid_y, grid=False)
        self.reverse_mapping_y = scipy.interpolate.SmoothBivariateSpline(self.intersections[:, 0], self.intersections[:, 1], self.anchors[:, 1])(grid_x, grid_y, grid=False)
    
    def _unwarp(self, filled):
        
        self._compute_reverse_mapping()

        pic = cv2.remap(filled, np.float32(self.reverse_mapping_x), np.float32(self.reverse_mapping_y), cv2.INTER_LINEAR, borderValue=(255,))
        
        M = cv2.getRotationMatrix2D((self.center[0] - self.offset_y, self.center[1] - self.offset_x), - self.angle, 1)
        return cv2.warpAffine(pic, M, (pic.shape[1], pic.shape[0]), borderValue=(255,))
        
    def _uncrop(self, picture, unwarped):
        
        colour = np.array([0, 150, 0])
        for i in range(unwarped.shape[0]):
            for j in range(unwarped.shape[1]):
                if not all(unwarped[i, j, :] > 150):
                    picture[i + self.offset_y, j + self.offset_x, :] = colour
        
        return picture
        
    def _get_entries(self, image, threshold=15):
        if self.number_of_columns == 16:
            offsets_x = [int(np.round(0 * image.shape[0])), int(np.round(0.0625 * image.shape[0])), int(np.round(0.125 * image.shape[0])), int(np.round(0.1875 * image.shape[0])), int(np.round(0.25 * image.shape[0])), int(np.round(0.3125 * image.shape[0])), int(np.round(0.375 * image.shape[0])), int(np.round(0.4375 * image.shape[0])), int(np.round(0.5 * image.shape[0])), int(np.round(0.5625 * image.shape[0])), int(np.round(0.625 * image.shape[0])), int(np.round(0.6875 * image.shape[0])), int(np.round(0.75 * image.shape[0])), int(np.round(0.8125 * image.shape[0])), int(np.round(0.875 * image.shape[0])), int(np.round(0.9375 * image.shape[0]))]
            offsets_y = [0, int(np.round(0.0625 * image.shape[1])), int(np.round(0.125 * image.shape[1])), int(np.round(0.1875 * image.shape[1]))]
            offsets_y_3_by_3 = [int(np.round(0 * image.shape[1])), int(np.round(0.25 * image.shape[1])), int(np.round(0.50 * image.shape[1])), int(np.round(0.75 * image.shape[1]))]
            
            size_y = int(np.round(0.057 * image.shape[1]))
            size_x = int(np.round(0.057 * image.shape[0]))
        else:
            offsets_x = [int(np.round(0.001 * image.shape[0])), int(np.round(0.113 * image.shape[0])), int(np.round(0.223 * image.shape[0])), int(np.round(0.335 * image.shape[0])), int(np.round(0.445 * image.shape[0])), int(np.round(0.555 * image.shape[0])), int(np.round(0.668 * image.shape[0])), int(np.round(0.780 * image.shape[0])), int(np.round(0.887 * image.shape[0]))]
            offsets_y = [0, int(np.round(0.112 * image.shape[1])), int(np.round(0.22 * image.shape[1]))]
            offsets_y_3_by_3 = [int(np.round(0 * image.shape[1])), int(np.round(0.333 * image.shape[1])), int(np.round(0.666 * image.shape[1]))]
            
            size_y = int(np.round(0.1 * image.shape[1]))
            size_x = int(np.round(0.1 * image.shape[0]))
        
        offset_box_xx = int(0.05 * size_x)
        offset_box_yy = int(0.05 * size_y)
            
        # boxes = np.empty((self.number_of_rows * self.number_of_rows, 1, self.entry_detector.width, self.entry_detector.height), dtype=np.float32)
        boxes = []
        entries = np.array(["#" for _ in range(self.number_of_columns * self.number_of_rows)])
        current = 0
        for i in range(self.number_of_rows):
            offset_x = offsets_x[i]
            for j in range(int(np.round(np.sqrt(self.number_of_columns)))):
                for k in range(int(np.round(np.sqrt(self.number_of_columns)))):
                    offset_y = offsets_y_3_by_3[j] + offsets_y[k]
                    box = image[offset_x: offset_x + size_x, offset_y: offset_y + size_y]
                    if np.mean(box[offset_box_xx: -offset_box_xx, offset_box_yy: -offset_box_yy]) > threshold:
                        boxes.append(np.float32(cv2.resize(box, (self.entry_detector.width, self.entry_detector.height))))
                    else:
                        entries[current] = "0"
                    current += 1
        
        detected_entries = self.entry_detector.get_entries(boxes, self.number_of_rows)
        current = 0
        for i in range(entries.shape[0]):
            if entries[i] == "#":
                entries[i] = detected_entries[current]
                current += 1
        return "".join(entries)
        
    def _fill(self, image, state):
    
        result = 255 * np.ones((image.shape[:2]), dtype=image.dtype)

        if self.number_of_columns == 16:
            offsets_x = [int(np.round(0 * image.shape[0])), int(np.round(0.0625 * image.shape[0])), int(np.round(0.125 * image.shape[0])), int(np.round(0.1875 * image.shape[0])), int(np.round(0.25 * image.shape[0])), int(np.round(0.3125 * image.shape[0])), int(np.round(0.375 * image.shape[0])), int(np.round(0.4375 * image.shape[0])), int(np.round(0.5 * image.shape[0])), int(np.round(0.5625 * image.shape[0])), int(np.round(0.625 * image.shape[0])), int(np.round(0.6875 * image.shape[0])), int(np.round(0.75 * image.shape[0])), int(np.round(0.8125 * image.shape[0])), int(np.round(0.875 * image.shape[0])), int(np.round(0.9375 * image.shape[0]))]
            offsets_y = [0, int(np.round(0.0625 * image.shape[1])), int(np.round(0.125 * image.shape[1])), int(np.round(0.1875 * image.shape[1]))]
            offsets_y_3_by_3 = [int(np.round(0 * image.shape[1])), int(np.round(0.25 * image.shape[1])), int(np.round(0.50 * image.shape[1])), int(np.round(0.75 * image.shape[1]))]
            
            size_y = int(np.round(0.057 * image.shape[1]))
            size_x = int(np.round(0.057 * image.shape[0]))
        else:
            offsets_x = [int(np.round(0.001 * image.shape[0])), int(np.round(0.113 * image.shape[0])), int(np.round(0.223 * image.shape[0])), int(np.round(0.335 * image.shape[0])), int(np.round(0.445 * image.shape[0])), int(np.round(0.555 * image.shape[0])), int(np.round(0.668 * image.shape[0])), int(np.round(0.780 * image.shape[0])), int(np.round(0.887 * image.shape[0]))]
            offsets_y = [0, int(np.round(0.112 * image.shape[1])), int(np.round(0.22 * image.shape[1]))]
            offsets_y_3_by_3 = [int(np.round(0 * image.shape[1])), int(np.round(0.333 * image.shape[1])), int(np.round(0.666 * image.shape[1]))]
            
            size_y = int(np.round(0.1 * image.shape[1]))
            size_x = int(np.round(0.1 * image.shape[0]))
        
        l = 0
        for i in range(int(np.round(self.number_of_rows))):
            offset_x = offsets_x[i]
            for j in range(int(np.round(np.sqrt(self.number_of_columns)))):
                for k in range(int(np.round(np.sqrt(self.number_of_columns)))):
                    if self.state[l] == "0":
                        offset_y = offsets_y_3_by_3[j] + offsets_y[k]
                        if state[l] == "0":
                            result[offset_x: offset_x + size_x, offset_y: offset_y + size_y] = cv2.resize(self.entries["O"], (size_y, size_x))
                        else:
                            result[offset_x: offset_x + size_x, offset_y: offset_y + size_y] = cv2.resize(self.entries[state[l]], (size_y, size_x))
                    
                    l += 1
         
        return result
    
