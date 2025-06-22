import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import griddata
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
import logging

class AstroImageProcessor:
    def __init__(self, bit_depth=16):
        """
        Processeur d'images astronomiques pour export JPG
        
        Args:
            bit_depth: Profondeur de bits pour les calculs internes (8 ou 16)
        """
        self.bit_depth = bit_depth
        self.max_value = 2**bit_depth - 1
        
        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_data):
        """
        Charge une image (array numpy ou fichier FITS)
        
        Args:
            image_data: array numpy ou chemin vers fichier FITS
        """
        if isinstance(image_data, (str, Path)):
            try:
                with fits.open(image_data) as hdul:
                    data = hdul[0].data.astype(np.float64)
                    self.logger.info(f"Image FITS chargée: {data.shape}")
                    return data
            except Exception as e:
                self.logger.error(f"Erreur chargement FITS: {e}")
                return None
        else:
            return image_data.astype(np.float64)
    
    def remove_background_gradient(self, image, method='mesh', mesh_size=64):
        """
        Supprime le gradient de fond
        
        Args:
            image: Image d'entrée
            method: Méthode ('mesh', 'polynomial', 'median')
            mesh_size: Taille de la grille pour la méthode mesh
        """
        is_color = image.ndim == 3
        
        if is_color:
            processed = np.zeros_like(image)
            for channel in range(image.shape[0]):
                processed[channel] = self._remove_gradient_single(
                    image[channel], method, mesh_size
                )
            return processed
        else:
            return self._remove_gradient_single(image, method, mesh_size)
    
    def _remove_gradient_single(self, image, method, mesh_size):
        """Supprime le gradient sur une image monochrome"""
        h, w = image.shape
        
        if method == 'mesh':
            # Méthode par grille d'échantillonnage
            y_coords = np.arange(0, h, mesh_size)
            x_coords = np.arange(0, w, mesh_size)
            
            # Échantillonne le fond sur une grille
            background_samples = []
            sample_coords = []
            
            for y in y_coords:
                for x in x_coords:
                    y_end = min(y + mesh_size, h)
                    x_end = min(x + mesh_size, w)
                    patch = image[y:y_end, x:x_end]
                    
                    # Utilise sigma clipping pour avoir une estimation robuste
                    mean, _, _ = sigma_clipped_stats(patch, sigma=2.0)
                    background_samples.append(mean)
                    sample_coords.append((y + mesh_size//2, x + mesh_size//2))
            
            # Interpole pour créer la carte de fond
            sample_coords = np.array(sample_coords)
            background_samples = np.array(background_samples)
            
            yi, xi = np.mgrid[0:h, 0:w]
            background_map = griddata(
                sample_coords, background_samples, (yi, xi), 
                method='cubic', fill_value=np.median(background_samples)
            )
            
            return image - background_map
            
        elif method == 'polynomial':
            # Ajustement polynomial 2D
            y, x = np.mgrid[0:h, 0:w]
            y_flat = y.flatten()
            x_flat = x.flatten()
            image_flat = image.flatten()
            
            # Matrice des coefficients pour polynôme de degré 2
            A = np.column_stack([
                np.ones_like(x_flat),
                x_flat, y_flat,
                x_flat**2, x_flat*y_flat, y_flat**2
            ])
            
            # Résolution par moindres carrés
            coeffs = np.linalg.lstsq(A, image_flat, rcond=None)[0]
            background_map = (coeffs[0] + coeffs[1]*x + coeffs[2]*y + 
                            coeffs[3]*x**2 + coeffs[4]*x*y + coeffs[5]*y**2)
            
            return image - background_map
            
        elif method == 'median':
            # Filtrage médian large puis soustraction
            kernel_size = min(h, w) // 20  # 5% de la taille d'image
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            background_map = median_filter(image, size=kernel_size)
            return image - background_map
        
        else:
            self.logger.warning(f"Méthode inconnue: {method}")
            return image
    
    def histogram_stretch(self, image, method='midtones', 
                         shadow_clip=0.0, highlight_clip=1.0, 
                         midtone=0.5, gamma=1.0):
        """
        Étire l'histogramme
        
        Args:
            image: Image d'entrée
            method: 'linear', 'midtones', 'asinh', 'sqrt'
            shadow_clip: Point de coupure des ombres (percentile 0-1)
            highlight_clip: Point de coupure des hautes lumières (percentile 0-1)
            midtone: Position des tons moyens (0-1)
            gamma: Correction gamma
        """
        is_color = image.ndim == 3
        
        if is_color:
            stretched = np.zeros_like(image)
            for channel in range(image.shape[0]):
                stretched[channel] = self._stretch_single_channel(
                    image[channel], method, shadow_clip, highlight_clip, midtone, gamma
                )
            return stretched
        else:
            return self._stretch_single_channel(
                image, method, shadow_clip, highlight_clip, midtone, gamma
            )
    
    def _stretch_single_channel(self, image, method, shadow_clip, highlight_clip, midtone, gamma):
        """Étire l'histogramme d'un canal"""
        # Normalisation initiale [0,1]
        img_min, img_max = np.min(image), np.max(image)
        if img_max <= img_min:
            return np.zeros_like(image)
        
        normalized = (image - img_min) / (img_max - img_min)
        
        # Points de coupure basés sur les percentiles
        shadow_val = np.percentile(normalized, shadow_clip * 100)
        highlight_val = np.percentile(normalized, highlight_clip * 100)
        
        # Clamping
        clipped = np.clip(normalized, shadow_val, highlight_val)
        
        # Renormalisation après clipping
        if highlight_val > shadow_val:
            clipped = (clipped - shadow_val) / (highlight_val - shadow_val)
        
        # Application de l'étirage selon la méthode
        if method == 'linear':
            stretched = clipped
            
        elif method == 'midtones':
            # Étirage des tons moyens (courbe en S)
            stretched = self._midtone_stretch(clipped, midtone)
            
        elif method == 'asinh':
            # Étirement asinh (bon pour les données à grande dynamique)
            stretched = np.arcsinh(clipped * 10) / np.arcsinh(10)
            
        elif method == 'sqrt':
            # Étirement racine carrée
            stretched = np.sqrt(clipped)
            
        else:
            stretched = clipped
        
        # Correction gamma
        if gamma != 1.0:
            stretched = np.power(stretched, 1.0/gamma)
        
        return np.clip(stretched, 0, 1)
    
    def _midtone_stretch(self, image, midtone):
        """Applique un étirement des tons moyens"""
        # Transformation basée sur la fonction de transfert Photoshop
        if midtone <= 0 or midtone >= 1:
            return image
        
        # Calcul du paramètre de la courbe
        gamma = np.log(0.5) / np.log(midtone)
        return np.power(image, gamma)
    
    def enhance_contrast(self, image, method='clahe', clip_limit=2.0, tile_grid_size=8):
        """
        Améliore le contraste
        
        Args:
            image: Image d'entrée [0,1]
            method: 'clahe', 'unsharp_mask', 'local_contrast'
            clip_limit: Limite de clipping pour CLAHE
            tile_grid_size: Taille de grille pour CLAHE
        """
        is_color = image.ndim == 3
        
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                   tileGridSize=(tile_grid_size, tile_grid_size))
            
            if is_color:
                enhanced = np.zeros_like(image)
                for channel in range(image.shape[0]):
                    img_8bit = (image[channel] * 255).astype(np.uint8)
                    enhanced_8bit = clahe.apply(img_8bit)
                    enhanced[channel] = enhanced_8bit.astype(np.float64) / 255.0
                return enhanced
            else:
                img_8bit = (image * 255).astype(np.uint8)
                enhanced_8bit = clahe.apply(img_8bit)
                return enhanced_8bit.astype(np.float64) / 255.0
                
        elif method == 'unsharp_mask':
            return self._unsharp_mask(image)
            
        elif method == 'local_contrast':
            return self._local_contrast_enhancement(image)
        
        return image
    
    def _unsharp_mask(self, image, radius=2.0, strength=0.5):
        """Applique un masque flou pour augmenter la netteté"""
        is_color = image.ndim == 3
        
        if is_color:
            enhanced = np.zeros_like(image)
            for channel in range(image.shape[0]):
                blurred = gaussian_filter(image[channel], sigma=radius)
                mask = image[channel] - blurred
                enhanced[channel] = image[channel] + strength * mask
            return np.clip(enhanced, 0, 1)
        else:
            blurred = gaussian_filter(image, sigma=radius)
            mask = image - blurred
            return np.clip(image + strength * mask, 0, 1)
    
    def _local_contrast_enhancement(self, image, kernel_size=15):
        """Amélioration du contraste local"""
        is_color = image.ndim == 3
        
        if is_color:
            enhanced = np.zeros_like(image)
            for channel in range(image.shape[0]):
                local_mean = gaussian_filter(image[channel], sigma=kernel_size/3)
                enhanced[channel] = image[channel] + 0.3 * (image[channel] - local_mean)
            return np.clip(enhanced, 0, 1)
        else:
            local_mean = gaussian_filter(image, sigma=kernel_size/3)
            return np.clip(image + 0.3 * (image - local_mean), 0, 1)
    
    def reduce_noise(self, image, method='bilateral', strength=1.0):
        """
        Réduit le bruit
        
        Args:
            image: Image d'entrée [0,1]
            method: 'bilateral', 'gaussian', 'median'
            strength: Intensité du débruitage
        """
        is_color = image.ndim == 3
        
        if method == 'bilateral':
            if is_color:
                denoised = np.zeros_like(image)
                for channel in range(image.shape[0]):
                    img_8bit = (image[channel] * 255).astype(np.uint8)
                    denoised_8bit = cv2.bilateralFilter(
                        img_8bit, -1, 
                        sigmaColor=25*strength, 
                        sigmaSpace=25*strength
                    )
                    denoised[channel] = denoised_8bit.astype(np.float64) / 255.0
                return denoised
            else:
                img_8bit = (image * 255).astype(np.uint8)
                denoised_8bit = cv2.bilateralFilter(
                    img_8bit, -1, 
                    sigmaColor=25*strength, 
                    sigmaSpace=25*strength
                )
                return denoised_8bit.astype(np.float64) / 255.0
                
        elif method == 'gaussian':
            if is_color:
                denoised = np.zeros_like(image)
                for channel in range(image.shape[0]):
                    denoised[channel] = gaussian_filter(image[channel], sigma=strength)
                return denoised
            else:
                return gaussian_filter(image, sigma=strength)
                
        elif method == 'median':
            kernel_size = int(2 * strength + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            if is_color:
                denoised = np.zeros_like(image)
                for channel in range(image.shape[0]):
                    denoised[channel] = median_filter(image[channel], size=kernel_size)
                return denoised
            else:
                return median_filter(image, size=kernel_size)
        
        return image
    
    def color_balance(self, image, method='gray_world'):
        """
        Équilibrage des couleurs (pour images couleur uniquement)
        
        Args:
            image: Image couleur [0,1]
            method: 'gray_world', 'white_patch'
        """
        if image.ndim != 3:
            return image
        
        if method == 'gray_world':
            # Suppose que la moyenne de l'image doit être grise
            means = [np.mean(image[c]) for c in range(3)]
            target_mean = np.mean(means)
            
            balanced = np.zeros_like(image)
            for channel in range(3):
                if means[channel] > 0:
                    factor = target_mean / means[channel]
                    balanced[channel] = image[channel] * factor
                else:
                    balanced[channel] = image[channel]
            
            return np.clip(balanced, 0, 1)
            
        elif method == 'white_patch':
            # Suppose que le pixel le plus brillant doit être blanc
            maxes = [np.max(image[c]) for c in range(3)]
            target_max = max(maxes)
            
            balanced = np.zeros_like(image)
            for channel in range(3):
                if maxes[channel] > 0:
                    factor = target_max / maxes[channel]
                    balanced[channel] = image[channel] * factor
                else:
                    balanced[channel] = image[channel]
            
            return np.clip(balanced, 0, 1)
        
        return image
    
    def process_image(self, image_data, 
                     remove_gradient=False, gradient_method='mesh',
                     stretch_method='midtones', shadow_clip=0.0001, highlight_clip=0.9999,
                     midtone=0.3, gamma=1.0,
                     enhance_contrast=True, contrast_method='clahe',
                     reduce_noise_flag=True, noise_method='bilateral', noise_strength=0.5,
                     color_balance_flag=True, balance_method='gray_world',
                     saturation_boost=1.2):
        """
        Pipeline complet de traitement
        
        Args:
            image_data: Données d'image (array ou chemin FITS)
            remove_gradient: Supprime le gradient de fond
            gradient_method: Méthode de suppression du gradient
            stretch_method: Méthode d'étirement d'histogramme
            shadow_clip/highlight_clip: Points de coupure (percentiles)
            midtone: Position des tons moyens
            gamma: Correction gamma
            enhance_contrast: Active l'amélioration du contraste
            contrast_method: Méthode d'amélioration du contraste
            reduce_noise_flag: Active la réduction de bruit
            noise_method: Méthode de réduction de bruit
            noise_strength: Force de la réduction de bruit
            color_balance_flag: Active l'équilibrage des couleurs
            balance_method: Méthode d'équilibrage des couleurs
            saturation_boost: Facteur de boost de saturation
        """
        # Chargement
        image = self.load_image(image_data)
        if image is None:
            return None
        
        self.logger.info("Début du traitement d'image")
        
        # 1. Suppression du gradient de fond
        if remove_gradient:
            self.logger.info("Suppression du gradient de fond")
            image = self.remove_background_gradient(image, method=gradient_method)
        
        # 2. Étirement d'histogramme
        self.logger.info(f"Étirement d'histogramme ({stretch_method})")
        image = self.histogram_stretch(
            image, method=stretch_method,
            shadow_clip=shadow_clip, highlight_clip=highlight_clip,
            midtone=midtone, gamma=gamma
        )
        
        # 3. Amélioration du contraste
        if enhance_contrast:
            self.logger.info(f"Amélioration du contraste ({contrast_method})")
            image = self.enhance_contrast(image, method=contrast_method)
        
        # 4. Réduction de bruit
        if reduce_noise_flag:
            self.logger.info(f"Réduction de bruit ({noise_method})")
            image = self.reduce_noise(image, method=noise_method, strength=noise_strength)
        
        # 5. Équilibrage des couleurs (images couleur uniquement)
        if color_balance_flag and image.ndim == 3:
            self.logger.info(f"Équilibrage des couleurs ({balance_method})")
            image = self.color_balance(image, method=balance_method)
        
        # 6. Boost de saturation (images couleur uniquement)
        if saturation_boost != 1.0 and image.ndim == 3:
            self.logger.info(f"Boost de saturation (x{saturation_boost})")
            image = self._boost_saturation(image, saturation_boost)
        
        self.logger.info("Traitement terminé")
        return image
    
    def _boost_saturation(self, image, factor):
        """Augmente la saturation d'une image couleur"""
        # Conversion RGB vers HSV
        rgb_image = np.transpose(image, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        hsv_image = cv2.cvtColor(rgb_image.astype(np.float32), cv2.COLOR_RGB2HSV)
        
        # Boost de saturation
        hsv_image[:, :, 1] *= factor
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 1)
        
        # Conversion HSV vers RGB
        rgb_boosted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return np.transpose(rgb_boosted, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    
    def save_jpg(self, image, filepath, quality=95, resize_factor=1.0):
        """
        Sauvegarde l'image en JPG
        
        Args:
            image: Image traitée [0,1]
            filepath: Chemin de sauvegarde
            quality: Qualité JPG (1-100)
            resize_factor: Facteur de redimensionnement
        """
        try:
            # Conversion en format d'affichage
            if image.ndim == 3:
                # Image couleur (C,H,W) -> (H,W,C)
                display_image = np.transpose(image, (1, 2, 0))
            else:
                # Image N&B -> RGB
                display_image = np.stack([image, image, image], axis=2)
            
            # Conversion en 8-bit
            image_8bit = (np.clip(display_image, 0, 1) * 255).astype(np.uint8)
            
            # Redimensionnement si nécessaire
            if resize_factor != 1.0:
                h, w = image_8bit.shape[:2]
                new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                image_8bit = cv2.resize(image_8bit, (new_w, new_h), 
                                      interpolation=cv2.INTER_LANCZOS4)
            
            # Sauvegarde avec PIL pour contrôler la qualité
            pil_image = Image.fromarray(image_8bit)
            pil_image.save(filepath, 'JPEG', quality=quality, optimize=True)
            
            self.logger.info(f"Image JPG sauvegardée: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde JPG: {e}")
            return False

# Exemple d'utilisation
if __name__ == "__main__":
    processor = AstroImageProcessor()
    
    # Exemple de traitement d'une image FITS
    input_file = "live_stack_00002.fit"
    output_file = "processed_image.jpg"
    
    if Path(input_file).exists():
        # Traitement avec paramètres par défaut
        processed = processor.process_image(
            input_file,
            stretch_method='asinh',
            midtone=1,
            shadow_clip=0.3,
            highlight_clip=0.9995,
            enhance_contrast=True,
            contrast_method='clahe',
            remove_gradient=True,
            gradient_method='polynomial', 
        )
        
        if processed is not None:
            # Sauvegarde en JPG
            processor.save_jpg(processed, output_file, quality=95)
            print(f"Image traitée sauvée: {output_file}")
        else:
            print("Erreur lors du traitement")
    else:
        print(f"Fichier {input_file} non trouvé")