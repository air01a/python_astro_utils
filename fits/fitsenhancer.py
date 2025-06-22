import numpy as np
from typing import Optional, Tuple, Union, Callable
import warnings
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure, restoration
from skimage.filters import unsharp_mask
import cv2

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets non disponible. Les fonctions de wavelets ne seront pas disponibles.")


class AstroImageEnhancer:
    """
    Librairie d'amélioration d'images astronomiques.
    Compatible avec FitsImageManager pour le traitement d'images FITS.
    
    Fonctionnalités:
    - Étirement d'histogramme (linéaire, asinh, midtones)
    - Ajustements de contraste avancés
    - Netteté par wavelets et masque flou
    - Débruitage (gaussien, médian, wavelets)
    - Normalisation et calibration
    """
    
    def __init__(self):
        self.history = []  # Historique des opérations
    
    def _add_to_history(self, operation: str, params: dict = None):
        """Ajoute une opération à l'historique."""
        self.history.append({
            'operation': operation,
            'parameters': params or {}
        })
    
    def _ensure_float(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Convertit l'image en float64 pour les calculs et retourne les infos de conversion.
        """
        original_dtype = image.dtype
        original_range = (image.min(), image.max())
        
        if image.dtype.kind in ['u', 'i']:  # unsigned ou signed integer
            if image.dtype == np.uint8:
                image_float = image.astype(np.float64) / 255.0
            elif image.dtype == np.uint16:
                image_float = image.astype(np.float64) / 65535.0
            else:
                image_float = image.astype(np.float64) / np.iinfo(image.dtype).max
        else:
            image_float = image.astype(np.float64)
            if original_range[1] > 1.0:  # Probablement pas normalisé
                image_float = image_float / original_range[1]
        
        conversion_info = {
            'original_dtype': original_dtype,
            'original_range': original_range,
            'was_normalized': original_range[1] <= 1.0
        }
        
        return image_float, conversion_info
    
    def _restore_dtype(self, image: np.ndarray, conversion_info: dict) -> np.ndarray:
        """Restaure le type de données original."""
        original_dtype = conversion_info['original_dtype']
        original_range = conversion_info['original_range']
        
        if original_dtype.kind in ['u', 'i']:
            if original_dtype == np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                image = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
            else:
                image = np.clip(image * np.iinfo(original_dtype).max, 
                               np.iinfo(original_dtype).min, 
                               np.iinfo(original_dtype).max).astype(original_dtype)
        else:
            if not conversion_info['was_normalized']:
                image = image * original_range[1]
            image = image.astype(original_dtype)
        
        return image
    
    # ========== ÉTIREMENT D'HISTOGRAMME ==========
    
    def linear_stretch(self, image: np.ndarray, 
                      low_percentile: float = 1.0, 
                      high_percentile: float = 99.0) -> np.ndarray:
        """
        Étirement linéaire de l'histogramme.
        
        Args:
            image: Image d'entrée
            low_percentile: Percentile bas pour le clipping
            high_percentile: Percentile haut pour le clipping
            
        Returns:
            Image avec étirement linéaire appliqué
        """
        image_float, conv_info = self._ensure_float(image)
        
        # Calculer les percentiles
        p_low = np.percentile(image_float, low_percentile)
        p_high = np.percentile(image_float, high_percentile)
        
        # Étirement linéaire
        stretched = np.clip((image_float - p_low) / (p_high - p_low), 0, 1)
        
        result = self._restore_dtype(stretched, conv_info)
        self._add_to_history('linear_stretch', {
            'low_percentile': low_percentile,
            'high_percentile': high_percentile
        })
        
        return result
    
    def asinh_stretch(self, image: np.ndarray, 
                     stretch_factor: float = 0.1,
                     black_point: float = 0.0) -> np.ndarray:
        """
        Étirement asinh (inverse sinus hyperbolique) - idéal pour l'astronomie.
        
        Args:
            image: Image d'entrée
            stretch_factor: Facteur d'étirement (plus petit = plus d'étirement)
            black_point: Point noir (soustrait avant l'étirement)
            
        Returns:
            Image avec étirement asinh appliqué
        """
        image_float, conv_info = self._ensure_float(image)
        
        # Appliquer le point noir
        image_float = np.maximum(image_float - black_point, 0)
        
        # Étirement asinh
        stretched = np.arcsinh(image_float / stretch_factor) / np.arcsinh(1.0 / stretch_factor)
        stretched = np.clip(stretched, 0, 1)
        
        result = self._restore_dtype(stretched, conv_info)
        self._add_to_history('asinh_stretch', {
            'stretch_factor': stretch_factor,
            'black_point': black_point
        })
        
        return result
    
    def midtone_stretch(self, image: np.ndarray, 
                       midtone: float = 0.5,
                       shadows_clip: float = 0.0,
                       highlights_clip: float = 1.0) -> np.ndarray:
        """
        Étirement des tons moyens (similaire à l'outil Levels de Photoshop).
        
        Args:
            image: Image d'entrée
            midtone: Valeur gamma pour les tons moyens (0.1-10.0)
            shadows_clip: Point de clipping des ombres
            highlights_clip: Point de clipping des hautes lumières
            
        Returns:
            Image avec étirement des tons moyens
        """
        image_float, conv_info = self._ensure_float(image)
        
        # Clipping des ombres et hautes lumières
        image_float = np.clip((image_float - shadows_clip) / (highlights_clip - shadows_clip), 0, 1)
        
        # Correction gamma pour les tons moyens
        stretched = np.power(image_float, 1.0 / midtone)
        
        result = self._restore_dtype(stretched, conv_info)
        self._add_to_history('midtone_stretch', {
            'midtone': midtone,
            'shadows_clip': shadows_clip,
            'highlights_clip': highlights_clip
        })
        
        return result
    
    def histogram_equalization(self, image: np.ndarray, adaptive: bool = False) -> np.ndarray:
        """
        Égalisation d'histogramme (globale ou adaptative).
        
        Args:
            image: Image d'entrée
            adaptive: Si True, utilise CLAHE (Contrast Limited Adaptive Histogram Equalization)
            
        Returns:
            Image avec égalisation d'histogramme
        """
        if len(image.shape) == 3:
            # Image couleur - traiter chaque canal séparément
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                if adaptive:
                    # Convertir en uint8 pour CLAHE
                    channel = exposure.rescale_intensity(image[:, :, i], out_range=(0, 255)).astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    result[:, :, i] = clahe.apply(channel).astype(image.dtype)
                else:
                    result[:, :, i] = exposure.equalize_hist(image[:, :, i])
        else:
            # Image en niveaux de gris
            if adaptive:
                # CLAHE
                image_uint8 = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(image_uint8).astype(image.dtype)
            else:
                result = exposure.equalize_hist(image)
        
        self._add_to_history('histogram_equalization', {'adaptive': adaptive})
        return result
    
    # ========== CONTRASTE ==========
    
    def local_contrast_enhancement(self, image: np.ndarray, 
                                  radius: float = 50.0,
                                  amount: float = 1.0) -> np.ndarray:
        """
        Amélioration du contraste local (unsharp masking).
        
        Args:
            image: Image d'entrée
            radius: Rayon du flou gaussien
            amount: Force de l'amélioration
            
        Returns:
            Image avec contraste local amélioré
        """
        enhanced = unsharp_mask(image, radius=radius, amount=amount, preserve_range=True)
        
        self._add_to_history('local_contrast_enhancement', {
            'radius': radius,
            'amount': amount
        })
        
        return enhanced.astype(image.dtype)
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Correction gamma.
        
        Args:
            image: Image d'entrée
            gamma: Valeur gamma (>1 assombrit, <1 éclaircit)
            
        Returns:
            Image avec correction gamma
        """
        image_float, conv_info = self._ensure_float(image)
        corrected = np.power(image_float, 1.0 / gamma)
        result = self._restore_dtype(corrected, conv_info)
        
        self._add_to_history('gamma_correction', {'gamma': gamma})
        return result
    
    # ========== WAVELETS ==========
    
    def wavelet_sharpen(self, image: np.ndarray, 
                       wavelet: str = 'db4',
                       levels: int = 4,
                       sharpen_factor: float = 1.5) -> np.ndarray:
        """
        Netteté par wavelets - préserve mieux les détails fins.
        
        Args:
            image: Image d'entrée
            wavelet: Type de wavelet ('db4', 'db8', 'haar', 'bior2.2')
            levels: Nombre de niveaux de décomposition
            sharpen_factor: Facteur d'amélioration de la netteté
            
        Returns:
            Image avec netteté améliorée par wavelets
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets n'est pas installé. Utilisez: pip install PyWavelets")
        
        if len(image.shape) == 3:
            # Image couleur - traiter chaque canal
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = self._wavelet_sharpen_channel(
                    image[:, :, i], wavelet, levels, sharpen_factor
                )
        else:
            result = self._wavelet_sharpen_channel(image, wavelet, levels, sharpen_factor)
        
        self._add_to_history('wavelet_sharpen', {
            'wavelet': wavelet,
            'levels': levels,
            'sharpen_factor': sharpen_factor
        })
        
        return result
    
    def _wavelet_sharpen_channel(self, channel: np.ndarray, 
                               wavelet: str, levels: int, 
                               sharpen_factor: float) -> np.ndarray:
        """Applique la netteté par wavelets sur un canal."""
        # Décomposition wavelet
        coeffs = pywt.wavedec2(channel, wavelet, level=levels)
        
        # Améliorer les détails (coefficients haute fréquence)
        coeffs_enhanced = [coeffs[0]]  # Approximation (basse fréquence)
        
        for i in range(1, len(coeffs)):
            # Améliorer les détails horizontaux, verticaux et diagonaux
            cH, cV, cD = coeffs[i]
            cH_enhanced = cH * sharpen_factor
            cV_enhanced = cV * sharpen_factor
            cD_enhanced = cD * sharpen_factor
            coeffs_enhanced.append((cH_enhanced, cV_enhanced, cD_enhanced))
        
        # Reconstruction
        enhanced = pywt.waverec2(coeffs_enhanced, wavelet)
        
        # Préserver le type et la gamme de valeurs
        return np.clip(enhanced, channel.min(), channel.max()).astype(channel.dtype)
    
    def wavelet_denoise(self, image: np.ndarray, 
                       wavelet: str = 'db8',
                       sigma: Optional[float] = None,
                       mode: str = 'soft') -> np.ndarray:
        """
        Débruitage par wavelets (BayesShrink).
        
        Args:
            image: Image d'entrée
            wavelet: Type de wavelet
            sigma: Écart-type du bruit (auto-estimé si None)
            mode: Type de seuillage ('soft' ou 'hard')
            
        Returns:
            Image débruitée
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets n'est pas installé. Utilisez: pip install PyWavelets")
        
        # Estimer le bruit si non fourni
        if sigma is None:
            sigma = restoration.estimate_sigma(image, average_sigmas=True, channel_axis=-1 if len(image.shape) == 3 else None)
        
        # Débruitage
        denoised = restoration.denoise_wavelet(
            image, method='BayesShrink', mode=mode, 
            wavelet=wavelet, rescale_sigma=True,
            channel_axis=-1 if len(image.shape) == 3 else None
        )
        
        self._add_to_history('wavelet_denoise', {
            'wavelet': wavelet,
            'sigma': sigma,
            'mode': mode
        })
        
        return denoised.astype(image.dtype)
    
    # ========== DÉBRUITAGE ==========
    
    def gaussian_denoise(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Débruitage par filtre gaussien.
        
        Args:
            image: Image d'entrée
            sigma: Écart-type du filtre gaussien
            
        Returns:
            Image débruitée
        """
        denoised = gaussian_filter(image.astype(np.float64), sigma=sigma)
        result = denoised.astype(image.dtype)
        
        self._add_to_history('gaussian_denoise', {'sigma': sigma})
        return result
    
    def median_denoise(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Débruitage par filtre médian (efficace contre le bruit impulsionnel).
        
        Args:
            image: Image d'entrée
            size: Taille du noyau du filtre médian
            
        Returns:
            Image débruitée
        """
        if len(image.shape) == 3:
            # Image couleur
            denoised = np.zeros_like(image)
            for i in range(image.shape[2]):
                denoised[:, :, i] = median_filter(image[:, :, i], size=size)
        else:
            denoised = median_filter(image, size=size)
        
        self._add_to_history('median_denoise', {'size': size})
        return denoised
    
    def bilateral_denoise(self, image: np.ndarray, 
                         d: int = 9, 
                         sigma_color: float = 75.0, 
                         sigma_space: float = 75.0) -> np.ndarray:
        """
        Débruitage par filtre bilatéral (préserve les contours).
        
        Args:
            image: Image d'entrée
            d: Diamètre du voisinage des pixels
            sigma_color: Filtre sigma dans l'espace couleur
            sigma_space: Filtre sigma dans l'espace des coordonnées
            
        Returns:
            Image débruitée
        """
        # Convertir en uint8 pour OpenCV si nécessaire
        if image.dtype != np.uint8:
            image_normalized = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
            denoised = cv2.bilateralFilter(image_normalized, d, sigma_color, sigma_space)
            # Remettre à l'échelle originale
            denoised = exposure.rescale_intensity(denoised.astype(np.float64), 
                                                 out_range=(image.min(), image.max())).astype(image.dtype)
        else:
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        self._add_to_history('bilateral_denoise', {
            'd': d, 
            'sigma_color': sigma_color, 
            'sigma_space': sigma_space
        })
        
        return denoised
    
    # ========== NORMALISATION ET CALIBRATION ==========
    
    def normalize_background(self, image: np.ndarray, 
                           background_percentile: float = 10.0) -> np.ndarray:
        """
        Normalise le fond de l'image (soustraction du fond).
        
        Args:
            image: Image d'entrée
            background_percentile: Percentile pour estimer le fond
            
        Returns:
            Image avec fond normalisé
        """
        background_level = np.percentile(image, background_percentile)
        normalized = np.maximum(image.astype(np.float64) - background_level, 0)
        result = normalized.astype(image.dtype)
        
        self._add_to_history('normalize_background', {
            'background_percentile': background_percentile
        })
        
        return result
    
    def remove_gradient(self, image: np.ndarray, 
                       polynomial_degree: int = 2) -> np.ndarray:
        """
        Supprime les gradients de l'image (correction de vignettage).
        
        Args:
            image: Image d'entrée
            polynomial_degree: Degré du polynôme pour l'ajustement
            
        Returns:
            Image avec gradient supprimé
        """
        if len(image.shape) == 3:
            # Image couleur - traiter chaque canal
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = self._remove_gradient_channel(
                    image[:, :, i], polynomial_degree
                )
        else:
            result = self._remove_gradient_channel(image, polynomial_degree)
        
        self._add_to_history('remove_gradient', {
            'polynomial_degree': polynomial_degree
        })
        
        return result
    
    def _remove_gradient_channel(self, channel: np.ndarray, degree: int) -> np.ndarray:
        """Supprime le gradient sur un canal."""
        h, w = channel.shape
        
        # Créer les coordonnées
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_norm = (x - w/2) / (w/2)
        y_norm = (y - h/2) / (h/2)
        
        # Créer la matrice des termes polynomiaux
        terms = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                terms.append((x_norm ** i * y_norm ** j).flatten())
        
        A = np.column_stack(terms)
        b = channel.flatten()
        
        # Résolution par moindres carrés
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Calculer le gradient
        gradient = np.sum([coeffs[k] * terms[k].reshape(h, w) for k in range(len(coeffs))], axis=0)
        
        # Soustraire le gradient
        corrected = channel.astype(np.float64) - gradient
        corrected = np.maximum(corrected, 0)  # Éviter les valeurs négatives
        
        return corrected.astype(channel.dtype)
    
    # ========== UTILITÉ ==========
    
    def get_history(self) -> list:
        """Retourne l'historique des opérations effectuées."""
        return self.history.copy()
    
    def clear_history(self):
        """Efface l'historique des opérations."""
        self.history.clear()
    
    def apply_to_fits_manager(self, fits_manager, operation_func: Callable, *args, **kwargs):
        """
        Applique une opération à un FitsImageManager.
        
        Args:
            fits_manager: Instance de FitsImageManager
            operation_func: Fonction d'amélioration à appliquer
            *args, **kwargs: Arguments pour la fonction
            
        Returns:
            Image traitée
        """
        if fits_manager.processed_data is None:
            raise ValueError("Aucune image chargée dans le FitsImageManager")
        
        # Appliquer l'opération
        enhanced_data = operation_func(fits_manager.processed_data, *args, **kwargs)
        
        # Mettre à jour les données du gestionnaire FITS
        fits_manager.processed_data = enhanced_data
        
        return enhanced_data


# Exemple d'utilisation avec FitsImageManager
if __name__ == "__main__":
    from fitsprocessor import FitsImageManager
    
    # Créer les instances
    fits_manager = FitsImageManager()
    enhancer = AstroImageEnhancer()
    
    try:
        # Ouvrir une image FITS
        fits_manager.open_fits("../../utils/01-observation-m16/01-images-initial/TargetSet.M27.8.00.LIGHT.329.2023-10-01_21-39-23.fits.fits")
            
        # Appliquer des améliorations
        print("Application d'améliorations...")
        
        # Méthode 1: Directement avec apply_to_fits_manager
        enhancer.apply_to_fits_manager(fits_manager, enhancer.asinh_stretch, 
                                     stretch_factor=0.05)
        
        # Méthode 2: Manuellement
        #fits_manager.processed_data = enhancer.wavelet_sharpen(
        #    fits_manager.processed_data, sharpen_factor=1.2
        #)
        
        #fits_manager.processed_data = enhancer.wavelet_denoise(
        #    fits_manager.processed_data
        #)
        
        # Afficher l'historique
        print("\nOpérations effectuées:")
        for op in enhancer.get_history():
            print(f"- {op['operation']}: {op['parameters']}")
        
        # Sauvegarder le résultat
        fits_manager.save_fits("enhanced_image.fits")
        
    except Exception as e:
        print(f"Erreur: {e}")