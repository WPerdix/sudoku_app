import cv2
import numpy as np
import os
import scipy


class BoxSaver():
    
    def __init__(self, save_path: str, path: str, file: str, ext: str) -> None:
            
        self.path = os.getcwd()
        self.save_path = save_path
        self.file = file
        self.path = path
        
        self.number_of_rows = 9
        self.number_of_columns = 9
        self.width = 28
        self.height = self.width
        
        image = cv2.imread(f'{self.path}/{self.file}{ext}')
        cv2.imwrite(f'{self.save_path}/{self.file}/original.jpg', np.uint8(image))
        self.solve(image)
    
    def solve(self, picture):
        
        crop = self._crop_from_contour(picture)
        
        warped = self._detect_intersections_and_warp(crop)
        
        self.state = self._get_entries(warped)
        
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
        for i, vline in zip(vlines_numbers, vlines):
            for j, hline in zip(hlines_numbers, hlines):
                mask = cv2.bitwise_and(hline.T, vline)
                moments = cv2.moments(mask)
                if moments["m00"] != 0:
                    x = int(moments["m10"] / moments["m00"])
                    y = int(moments["m01"] / moments["m00"])
                
                    intersections.append((x, y))
                    anchors.append((pixels_x * i, pixels_y * j))
        
        self.anchors = np.array(anchors)
        self.intersections = np.array(intersections)
        
        picture = cv2.adaptiveThreshold(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 20)
        
        grid_x, grid_y = np.meshgrid(np.arange(self.number_of_columns * pixels_x + 1), np.arange(self.number_of_rows * pixels_y + 1))
        mapping_x = scipy.interpolate.SmoothBivariateSpline(self.anchors[:, 0], self.anchors[:, 1], self.intersections[:, 0])(grid_x, grid_y, grid=False)
        mapping_y = scipy.interpolate.SmoothBivariateSpline(self.anchors[:, 0], self.anchors[:, 1], self.intersections[:, 1])(grid_x, grid_y, grid=False)

        return cv2.remap(picture, np.float32(mapping_x), np.float32(mapping_y), cv2.INTER_LINEAR)
        
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
        
        if not os.path.exists(f'{self.save_path}/{self.file}/'):
            os.makedirs(f'{self.save_path}/{self.file}/')
            os.makedirs(f'{self.save_path}/{self.file}/boxes/')
        
        cv2.imwrite(f'{self.save_path}/{self.file}/registered.jpg', np.uint8(image))
        
        with open(f'{self.path}/{self.file}.dat', "r") as dat_file:
            dat_file_data = dat_file.read()
            
        with open(f'{self.save_path}/{self.file}/digits.dat', "w") as dat_file:
            dat_file.write(dat_file_data)
        
        offset_box_xx = int(0.05 * size_x)
        offset_box_yy = int(0.05 * size_y)
        
        if self.number_of_columns > 9:
            pass
        for i in range(self.number_of_rows):
            offset_x = offsets_x[i]
            for j in range(int(np.round(np.sqrt(self.number_of_columns)))):
                for k in range(int(np.round(np.sqrt(self.number_of_columns)))):
                    offset_y = offsets_y_3_by_3[j] + offsets_y[k]
                    box = image[offset_x: offset_x + size_x, offset_y: offset_y + size_y]
                    box = cv2.resize(box, (self.width, self.height))
                    cv2.imwrite(f'{self.save_path}/{self.file}/boxes/{i + 1}_{j * int(np.round(np.sqrt(self.number_of_columns))) + k + 1}.jpg', box)
                    
                  
def get_box_dataset(path):
    
    X = []
    y = []
    
    for directory in os.listdir(path):
        for sudoku in os.listdir(f'{path}/{directory}'):
            digits = []
            with open(f'{path}/{directory}/{sudoku}/digits.dat', "r") as dat_file:
                dat_file_data = dat_file.read().split(" ")
                for d in dat_file_data:
                    if len(d) == 1:
                        digits.append(int(d))
                    elif len(d) >= 2:
                        s = d.split("\n")
                        for c in s:
                            digits.append(int(c))
            
            boxes = []
            for box in sorted(os.listdir(f'{path}/{directory}/{sudoku}/boxes/')):
                b = cv2.imread(f'{path}/{directory}/{sudoku}/boxes/{box}')[:, :, 0]
                # b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
                boxes.append(b.reshape(*b.shape, 1))
            
            X += boxes
            y += digits

    X = np.array(X)
    y = np.array(y)
    yh = np.array(yh)

    new_X = []
    new_y = []
    new_yh = []
    for s in np.unique(y):
        if s == 0:
            temp = X[y == s]
            random_permutation = np.random.permutation(temp.shape[0])
            new_X.append(temp[random_permutation[: int(np.round(len(y[y != s]) / 9))]])
            temp = y[y == s]
            new_y.append(temp[random_permutation[: int(np.round(len(y[y != s]) / 9))]])
            temp = yh[y == s]
            new_yh.append(temp[random_permutation[: int(np.round(len(y[y != s]) / 9))]])
        else:
            new_X.append(X[y == s])
            new_y.append(y[y == s])
            new_yh.append(yh[y == s])
    
    return np.vstack(new_X), np.concatenate(new_y), np.concatenate(new_yh)
                    
                    
if __name__ == "__main__":
    
    from tqdm import tqdm
    
    ext = '.jpg'
    save_path = f'{os.getcwd()}/data/sudokus'
    path = f'{os.getcwd()}/data'
            
    folders = ['handwritten', 'empty', 'generated']
    for folder in folders:
        if not os.path.exists(f'{save_path}/{folder}'):
            os.makedirs(f'{save_path}/{folder}')
        for file in tqdm(os.listdir(f'{os.getcwd()}/data/{folder}/')):
            if file.endswith(ext):
                try:
                    BoxSaver(save_path=f'{save_path}/{folder}', path=f'{path}/{folder}', file=file.split(ext)[0], ext=ext)
                except:
                    print(f'{folder}/{file}', "Something went wrong!")
    
    X, y, yh = get_box_dataset(save_path)
    print(len(X), len(y), len(yh))
    


