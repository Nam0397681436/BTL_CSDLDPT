import numpy as np

class Gaussian:
    
    def __init__(self, size=3, sigma=1):
        self.size=size
        self.sigma=sigma

    def create_gausian_kernal(self):
        center= self.size//2
        x, y= np.mgrid[-center:center+1,-center:center+1]

        g=(1/(2*np.pi*self.sigma**2))*np.exp(-(x**2+y**2)/(2*self.sigma**2))
        return g / np.sum(g)

    def gaussian(self, img : np.ndarray):
        gau_kernal=self.create_gausian_kernal()
        img_gau=np.zeros_like(img)
        pad= self.size//2
        
        padded_img=np.pad(img,pad,mode='constant',constant_values=0)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                roi=padded_img[i:i+self.size,j:j+self.size]
                img_gau[i,j]=np.sum(roi*gau_kernal)
        
        return img_gau
