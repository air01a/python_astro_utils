import numpy as np
from collections import deque
from astropy.io import fits
from astropy.stats import sigma_clip
import astroalign as aa
import matplotlib.pyplot as plt
import cv2

class LiveStacker:
    def __init__(self, max_images=20):
        self.max_images = max_images
        self.image_queue = deque(maxlen=max_images)
        self.quality_weights = deque(maxlen=max_images)
        self.stack = None
        self.reference_image = None

    def compute_noise_level(self, image):
        # Évalue le bruit sur fond de ciel par sigma-clipping
        clipped = sigma_clip(image, sigma=3)
        return np.std(clipped)

    def align_image(self, image):
        if self.reference_image is None:
            self.reference_image = image
            return image
        try:
            aligned, _ = aa.register(image, self.reference_image)
            return aligned
        except Exception as e:
            print(f"[Erreur d'alignement] {e}")
            return None

    def add_frame(self, image):
        image = image.astype(np.float32)

        # 1. ALIGNEMENT (toujours sur la luminance pour cohérence)
        if image.ndim == 3 and image.shape[2] == 3:
            # Convertir RGB → Gray pour l'alignement
            gray_for_align = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_for_align = image

        aligned_gray = self.align_image(gray_for_align)
        if aligned_gray is None:
            return self.stack

        if image.ndim == 3 and image.shape[2] == 3:
            # Appliquer la même transformation à l’image RGB complète
            aligned_rgb, _ = aa.register(image, self.reference_image)
            aligned = aligned_rgb
        else:
            aligned = aligned_gray

        # 2. BRUIT GLOBAL & PONDÉRATION
        noise = self.compute_noise_level(aligned_gray)
        weight = 1.0 / (noise ** 2 + 1e-8)

        self.image_queue.append(aligned)
        self.quality_weights.append(weight)

        # 3. STACKING
        weights_array = np.array(self.quality_weights)
        stack_array = np.array(self.image_queue)

        if aligned.ndim == 2:
            # N&B
            clipped = sigma_clip(stack_array, sigma=2.5, axis=0)
            mean = np.ma.average(clipped, axis=0, weights=weights_array)
            self.stack = mean.filled(0)

        elif aligned.ndim == 3 and aligned.shape[2] == 3:
            # Couleur RGB, traitement canal par canal
            clipped_r = sigma_clip(stack_array[:, :, :, 0], sigma=2.5, axis=0)
            clipped_g = sigma_clip(stack_array[:, :, :, 1], sigma=2.5, axis=0)
            clipped_b = sigma_clip(stack_array[:, :, :, 2], sigma=2.5, axis=0)

            mean_r = np.ma.average(clipped_r, axis=0, weights=weights_array)
            mean_g = np.ma.average(clipped_g, axis=0, weights=weights_array)
            mean_b = np.ma.average(clipped_b, axis=0, weights=weights_array)

            self.stack = np.stack([
                mean_r.filled(0),
                mean_g.filled(0),
                mean_b.filled(0)
            ], axis=-1)

        return self.stack


def load_fits_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)

# Exemple d'utilisation
if __name__ == "__main__":
    from glob import glob
    from fitsprocessor import FitsImageManager

    fits_files = sorted(glob("../../utils/01-observation-m16/01-images-initial/*.fits"))  # Répertoire avec images FITS
    stacker = LiveStacker(max_images=20)
    fits = FitsImageManager()
    for path in fits_files:
        print(f"Empile {path}")
        fits.open_fits(path)
        fits.debayer()
        img = fits.processed_data
        result = stacker.add_frame(img)
    fits.processed_data=stacker.stack
    fits.save_as_image("test.jpg")
    fits.save_fits("test.fits")


