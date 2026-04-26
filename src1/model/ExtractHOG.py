import cv2
import numpy as np


class HOG:
    def __init__(self,img, pixels=(144, 256), cell_size=8, block_size=2,step_size=8, nbins=9):
        self.img=img
        self.cell_size  = cell_size
        self.block_size = block_size
        self.nbins      = nbins
        self.pixels     = pixels
        self.num_block  = (
            (pixels[0] - block_size * cell_size) // step_size + 1,
            (pixels[1] - block_size * cell_size) // step_size + 1
        )
        self.step_size = step_size

    def computeHOG(self):
        if len(self.img.shape) == 3:
            if self.img.shape[2] == 4:
                img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
            elif self.img.shape[2] == 3:
                img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = self.img
        else:
            img_gray = self.img
            
        hog_feature = []

        blk_px = self.block_size * self.cell_size
        for i in range(0, self.num_block[1] * self.step_size, self.step_size):
            for j in range(0, self.num_block[0] * self.step_size, self.step_size):
                img_block = img_gray[i:i + blk_px, j:j + blk_px]
                if img_block.shape[0] < blk_px or img_block.shape[1] < blk_px:
                    continue
                block_feat = self.compute_block(img_block)
                hog_feature.append(block_feat)

        hog_feature = np.concatenate(hog_feature)
        norm = np.linalg.norm(hog_feature)
        return hog_feature / (norm + 1e-6)

    def compute_block(self, img_block):
        hist_block = []
        for i in range(0, self.block_size * self.cell_size, self.cell_size):
            for j in range(0, self.block_size * self.cell_size, self.cell_size):
                img_cell = img_block[i:i + self.cell_size, j:j + self.cell_size]
                hist_block.append(self.compute_cell(img_cell))

        block_vec = np.concatenate(hist_block)

        # Chuẩn hoá L2 theo block
        norm = np.linalg.norm(block_vec)
        return block_vec / (norm + 1e-6)

    def compute_cell(self, img_cell):
        Ix, Iy = self.daoham_numpy(img_cell)
        mag, direction = self.calculate_gradient_magnitude_numpy(Ix, Iy)

        hist     = np.zeros(self.nbins)
        bin_width = 180.0 / self.nbins

        for i in range(img_cell.shape[0]):
            for j in range(img_cell.shape[1]):
                angle     = direction[i, j]
                magnitude = mag[i, j]

                if magnitude == 0:
                    continue

                bin_index = angle / bin_width
                idx1 = int(bin_index) % self.nbins
                idx2 = (idx1 + 1) % self.nbins

                dist_to_left_bin = angle - (int(bin_index) * bin_width)
                ratio_right = dist_to_left_bin / bin_width
                ratio_left  = 1.0 - ratio_right

                hist[idx1] += magnitude * ratio_left
                hist[idx2] += magnitude * ratio_right

        return hist

    def daoham_numpy(self, img):
        img = np.array(img, dtype=np.float64)
        Ix  = np.zeros_like(img)
        Iy  = np.zeros_like(img)

        Ix[:, 1:-1] = img[:, 2:] - img[:, :-2]
        Iy[1:-1, :] = img[2:, :] - img[:-2, :]

        Ix[:, 0]  = img[:, 1]
        Ix[:, -1] = -img[:, -2]
        Iy[0, :]  = img[1, :]
        Iy[-1, :] = -img[-2, :]

        return Ix, Iy

    def calculate_gradient_magnitude_numpy(self, Ix, Iy):
        Ix = np.array(Ix, dtype=np.float64)
        Iy = np.array(Iy, dtype=np.float64)

        magnitude     = np.sqrt(Ix**2 + Iy**2)
        direction_rad = np.arctan2(Iy, Ix)
        direction_deg = np.degrees(direction_rad)
        direction     = np.where(direction_deg < 0, direction_deg + 180, direction_deg)

        return magnitude, direction

    def gaussian(self, img, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)