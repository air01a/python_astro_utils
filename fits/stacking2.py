import numpy as np
from collections import deque
import astroalign as aa
from scipy import ndimage
from typing import Optional, Tuple, List
import warnings


class LiveStacker:
    """
    Live stacker pour astrophotographie avec sigma clipping et pondération intelligente.
    Utilise une fenêtre glissante pour optimiser l'usage mémoire.
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 sigma_low: float = 2.0,
                 sigma_high: float = 3.0,
                 min_images_for_clipping: int = 5):
        """
        Initialise le live stacker.
        
        Args:
            window_size: Taille de la fenêtre glissante pour le sigma clipping
            sigma_low: Seuil sigma inférieur pour le clipping
            sigma_high: Seuil sigma supérieur pour le clipping
            min_images_for_clipping: Nombre minimum d'images avant d'activer le sigma clipping
        """
        self.window_size = window_size
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.min_images_for_clipping = min_images_for_clipping
        
        # Buffers circulaires pour la fenêtre glissante
        self.image_buffer = deque(maxlen=window_size)
        self.weight_buffer = deque(maxlen=window_size)
        
        # Stack cumulatif et poids
        self.cumulative_stack = None
        self.cumulative_weights = None
        self.total_weight = 0.0
        
        # Image de référence pour l'alignement
        self.reference_image = None
        self.reference_sources = None
        
        # Statistiques
        self.n_images_processed = 0
        self.n_pixels_clipped = 0
        
    def _detect_sources(self, image: np.ndarray, 
                       detection_sigma: float = 5.0) -> Optional[np.ndarray]:
        """
        Détecte les sources dans l'image pour l'alignement.
        
        Args:
            image: Image d'entrée
            detection_sigma: Seuil de détection des sources
            
        Returns:
            Array des positions des sources (x, y)
        """
        try:
            sources = aa.find_sources(image, detection_sigma=detection_sigma)
            return sources
        except Exception as e:
            warnings.warn(f"Échec de détection des sources: {e}")
            return None
    
    def _align_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Aligne l'image sur la référence.
        
        Args:
            image: Image à aligner
            
        Returns:
            Tuple (image_alignée, succès)
        """
        if self.reference_image is None:
            # Première image = référence
            self.reference_image = image.copy()
            self.reference_sources = self._detect_sources(image)
            return image, True
        
        try:
            # Détection des sources dans l'image courante
            sources = self._detect_sources(image)
            if sources is None or len(sources) < 10:
                return None, False
            
            # Alignement avec astroalign
            aligned_image, footprint = aa.apply_transform(
                aa.find_transform(sources, self.reference_sources)[0],
                image,
                self.reference_image
            )
            
            return aligned_image, True
            
        except Exception as e:
            warnings.warn(f"Échec d'alignement: {e}")
            return None, False
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """
        Calcule la qualité de l'image basée sur le bruit.
        
        Args:
            image: Image à évaluer
            
        Returns:
            Poids de qualité (plus élevé = meilleure qualité)
        """
        # Méthode robuste pour estimer le bruit de fond
        # Utilise les percentiles pour éviter les étoiles
        background_mask = image < np.percentile(image, 70)
        background_pixels = image[background_mask]
        
        if len(background_pixels) == 0:
            return 1.0
        
        # Estimation du bruit par MAD (Median Absolute Deviation)
        median_bg = np.median(background_pixels)
        mad = np.median(np.abs(background_pixels - median_bg))
        noise_estimate = 1.4826 * mad  # Conversion MAD -> sigma pour distribution normale
        
        # Poids inversement proportionnel au bruit
        # Ajouter une constante pour éviter division par zéro
        weight = 1.0 / (noise_estimate + 1e-10)
        
        # Normaliser le poids (optionnel, dépend de vos préférences)
        return weight
    
    def _sigma_clip_stack(self, images: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectue le sigma clipping sur un stack d'images.
        
        Args:
            images: Stack d'images (N, H, W)
            weights: Poids correspondants (N,)
            
        Returns:
            Tuple (stack_clippé, masque_valide)
        """
        if len(images) < self.min_images_for_clipping:
            # Pas assez d'images pour le clipping, moyenne pondérée simple
            weighted_images = images * weights[:, np.newaxis, np.newaxis]
            return (np.sum(weighted_images, axis=0) / np.sum(weights),
                    np.ones_like(images, dtype=bool))
        
        # Calcul de la moyenne et écart-type pondérés
        weights_norm = weights / np.sum(weights)
        weighted_mean = np.average(images, axis=0, weights=weights)
        
        # Calcul de l'écart-type pondéré
        variance = np.average((images - weighted_mean[np.newaxis, :, :])**2, 
                             axis=0, weights=weights)
        std = np.sqrt(variance)
        
        # Masque de sigma clipping
        deviations = np.abs(images - weighted_mean[np.newaxis, :, :])
        sigma_mask = ((deviations <= self.sigma_high * std[np.newaxis, :, :]) & 
                     (deviations >= -self.sigma_low * std[np.newaxis, :, :]))
        
        # Compter les pixels clippés
        self.n_pixels_clipped += np.sum(~sigma_mask)
        
        # Recalcul avec masque
        masked_images = np.where(sigma_mask, images, np.nan)
        weights_expanded = np.broadcast_to(weights[:, np.newaxis, np.newaxis], 
                                         masked_images.shape)
        
        # Moyenne pondérée avec gestion des NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            clipped_stack = np.nansum(masked_images * weights_expanded, axis=0) / \
                           np.nansum(np.where(~np.isnan(masked_images), weights_expanded, 0), axis=0)
        
        # Remplacer les NaN par la moyenne simple si tous les pixels sont clippés
        nan_mask = np.isnan(clipped_stack)
        if np.any(nan_mask):
            clipped_stack[nan_mask] = weighted_mean[nan_mask]
        
        return clipped_stack, sigma_mask
    
    def add_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Ajoute une nouvelle image au stack.
        
        Args:
            image: Nouvelle image à ajouter
            
        Returns:
            Stack mis à jour ou None si échec d'alignement
        """
        # Alignement de l'image
        aligned_image, success = self._align_image(image)
        if not success:
            warnings.warn("Échec d'alignement, image ignorée")
            return None
        
        # Calcul du poids de qualité
        weight = self._calculate_image_quality(aligned_image)
        
        # Ajout aux buffers
        self.image_buffer.append(aligned_image.astype(np.float64))
        self.weight_buffer.append(weight)
        
        # Conversion en arrays numpy pour traitement vectorisé
        current_images = np.array(list(self.image_buffer))
        current_weights = np.array(list(self.weight_buffer))
        
        # Sigma clipping sur la fenêtre courante
        windowed_stack, valid_mask = self._sigma_clip_stack(current_images, current_weights)
        
        # Mise à jour du stack cumulatif
        if self.cumulative_stack is None:
            # Première image
            self.cumulative_stack = windowed_stack.copy()
            self.cumulative_weights = np.sum(current_weights)
            self.total_weight = self.cumulative_weights
        else:
            # Intégration avec le stack existant
            # Pondération entre ancien stack et nouveau
            old_weight = self.total_weight
            new_weight = np.sum(current_weights)
            
            # Mise à jour pondérée
            total_new_weight = old_weight + new_weight
            
            self.cumulative_stack = (
                (self.cumulative_stack * old_weight + windowed_stack * new_weight) / 
                total_new_weight
            )
            
            self.total_weight = total_new_weight
        
        self.n_images_processed += 1
        
        return self.cumulative_stack.copy()
    
    def get_current_stack(self) -> Optional[np.ndarray]:
        """
        Retourne le stack actuel.
        
        Returns:
            Stack actuel ou None si aucune image traitée
        """
        return self.cumulative_stack.copy() if self.cumulative_stack is not None else None
    
    def get_statistics(self) -> dict:
        """
        Retourne les statistiques du processus de stacking.
        
        Returns:
            Dictionnaire des statistiques
        """
        return {
            'n_images_processed': self.n_images_processed,
            'n_pixels_clipped': self.n_pixels_clipped,
            'total_weight': self.total_weight,
            'buffer_size': len(self.image_buffer),
            'clipping_ratio': self.n_pixels_clipped / (self.n_images_processed * 
                            (self.cumulative_stack.size if self.cumulative_stack is not None else 1))
        }
    
    def reset(self):
        """Remet à zéro le stacker."""
        self.image_buffer.clear()
        self.weight_buffer.clear()
        self.cumulative_stack = None
        self.cumulative_weights = None
        self.total_weight = 0.0
        self.reference_image = None
        self.reference_sources = None
        self.n_images_processed = 0
        self.n_pixels_clipped = 0


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation du stacker
    from glob import glob
    from fitsprocessor import FitsImageManager
    stacker = LiveStacker(window_size=20, sigma_low=2.0, sigma_high=3.0)
    
    # Simulation d'ajout d'images
    print("Exemple d'utilisation du LiveStacker:")
    print("=" * 50)

    # Images simulées (remplacez par vos vraies images FITS)
    fits_files = sorted(glob("../../utils/01-observation-m16/01-images-initial/*.fits")) 
        
    fits = FitsImageManager()
    for path in fits_files:
        print(f"Empile {path}")
        fits.open_fits(path)
        fits.debayer()
        img = fits.processed_data
        stacker.add_image(img)
    fits.processed_data=stacker.get_current_stack()
    fits.save_as_image("test2.jpg")
    fits.save_fits("test2.fits")
    # Statistiques finales
    final_stats = stacker.get_statistics()
    print("\nStatistiques finales:")
    print(f"Images traitées: {final_stats['n_images_processed']}")
    print(f"Pixels clippés: {final_stats['n_pixels_clipped']}")
    print(f"Ratio de clipping: {final_stats['clipping_ratio']:.4f}")
    print(f"Poids total: {final_stats['total_weight']:.2f}")