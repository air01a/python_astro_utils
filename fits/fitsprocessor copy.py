import numpy as np
from astropy.io import fits
from astropy.time import Time
import os
from typing import Optional, Tuple, Union, Dict, Any
import warnings

try:
    from colour_demosaicing import demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_Malvar2004
    DEBAYER_AVAILABLE = True
except ImportError:
    DEBAYER_AVAILABLE = False
    warnings.warn("colour_demosaicing non disponible. Le debayering ne sera pas possible.")


class FitsImageManager:

    
    # Patterns Bayer supportés
    BAYER_PATTERNS = {
        'RGGB': 'RGGB',
        'BGGR': 'BGGR', 
        'GRBG': 'GRBG',
        'GBRG': 'GBRG'
    }
    
    def __init__(self):
        self.filename = None
        self.original_data = None
        self.processed_data = None
        self.header = None
        self.is_color = False
        self.bayer_pattern = None
        self.original_shape = None
        self.is_debayered = False
        
    def open_fits(self, filename: str, hdu_index: int = 0) -> None:
        """
        Ouvre un fichier FITS et charge les données.
        
        Args:
            filename: Chemin vers le fichier FITS
            hdu_index: Index de l'HDU à charger (défaut: 0)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Fichier non trouvé: {filename}")
            
        self.filename = filename
        
        with fits.open(filename) as hdul:
            # Vérifier que l'HDU existe
            if hdu_index >= len(hdul):
                raise IndexError(f"HDU index {hdu_index} non valide. Le fichier a {len(hdul)} HDU(s)")
            
            hdu = hdul[hdu_index]
            self.original_data = hdu.data.copy()
            self.header = hdu.header.copy()
            self.original_shape = self.original_data.shape
            
        # Détecter le pattern Bayer
        self._detect_bayer_pattern()
        
        # Initialiser les données traitées
        self.processed_data = self.original_data.copy()
        self.is_debayered = False
        
        print(f"Fichier ouvert: {filename}")
        print(f"Dimensions: {self.original_shape}")
        print(f"Type de données: {self.original_data.dtype}")
        if self.bayer_pattern:
            print(f"Pattern Bayer détecté: {self.bayer_pattern}")
        else:
            print("Aucun pattern Bayer détecté (image couleur ou monochrome)")
    
    def _detect_bayer_pattern(self) -> None:
        """Détecte le pattern Bayer à partir des métadonnées du header."""
        self.bayer_pattern = None
        self.is_color = False
        
        # Rechercher dans différents champs possibles
        bayer_keys = ['BAYERPAT', 'COLORTYP', 'XBAYROFF', 'YBAYROFF']
        
        # Méthode 1: Pattern Bayer explicite
        if 'BAYERPAT' in self.header:
            pattern = self.header['BAYERPAT'].upper()
            if pattern in self.BAYER_PATTERNS:
                self.bayer_pattern = pattern
                return
        
        # Méthode 2: Offsets Bayer (XBAYROFF, YBAYROFF)
        if 'XBAYROFF' in self.header and 'YBAYROFF' in self.header:
            x_offset = self.header['XBAYROFF']
            y_offset = self.header['YBAYROFF']
            
            # Déduire le pattern à partir des offsets
            if x_offset == 0 and y_offset == 0:
                self.bayer_pattern = 'RGGB'
            elif x_offset == 1 and y_offset == 0:
                self.bayer_pattern = 'GRBG'
            elif x_offset == 0 and y_offset == 1:
                self.bayer_pattern = 'GBRG'
            elif x_offset == 1 and y_offset == 1:
                self.bayer_pattern = 'BGGR'
        
        # Méthode 3: Détection basée sur la géométrie
        if self.bayer_pattern is None and len(self.original_shape) == 2:
            # Si c'est une image 2D, c'est probablement une image Bayer
            # Pattern par défaut (peut être modifié manuellement)
            self.bayer_pattern = 'RGGB'
        
        # Vérifier si c'est déjà une image couleur
        if len(self.original_shape) == 3 and self.original_shape[2] == 3:
            self.is_color = True
            self.bayer_pattern = None
    
    def set_bayer_pattern(self, pattern: str) -> None:
        """
        Définit manuellement le pattern Bayer.
        
        Args:
            pattern: Pattern Bayer ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        """
        pattern = pattern.upper()
        if pattern not in self.BAYER_PATTERNS:
            raise ValueError(f"Pattern non supporté: {pattern}. Utilisez: {list(self.BAYER_PATTERNS.keys())}")
        
        self.bayer_pattern = pattern
        print(f"Pattern Bayer défini: {pattern}")
    
    def debayer(self, algorithm: str = 'bilinear') -> np.ndarray:
        """
        Effectue le debayering de l'image.
        
        Args:
            algorithm: Algorithme de debayering ('bilinear' ou 'malvar')
            
        Returns:
            Image debayerisée (H, W, 3)
        """
        if not DEBAYER_AVAILABLE:
            raise ImportError("Le module colour_demosaicing n'est pas installé")
        
        if self.bayer_pattern is None:
            raise ValueError("Aucun pattern Bayer défini. Utilisez set_bayer_pattern() ou vérifiez que l'image est bien une image Bayer")
        
        if self.is_color:
            print("L'image est déjà en couleur")
            return self.processed_data
        
        # Normaliser les données pour le debayering (0-1)
        data_normalized = self.processed_data.astype(np.float64)
        data_min, data_max = data_normalized.min(), data_normalized.max()
        if data_max > data_min:
            data_normalized = (data_normalized - data_min) / (data_max - data_min)
        
        # Effectuer le debayering
        if algorithm == 'bilinear':
            debayered = demosaicing_CFA_Bayer_bilinear(data_normalized, self.bayer_pattern)
        elif algorithm == 'malvar':
            debayered = demosaicing_CFA_Bayer_Malvar2004(data_normalized, self.bayer_pattern)
        else:
            raise ValueError("Algorithme non supporté. Utilisez 'bilinear' ou 'malvar'")
        
        # Remettre à l'échelle originale
        debayered = debayered * (data_max - data_min) + data_min
        debayered = debayered.astype(self.processed_data.dtype)
        
        self.processed_data = debayered
        self.is_debayered = True
        self.is_color = True
        
        print(f"Debayering effectué avec l'algorithme: {algorithm}")
        print(f"Nouvelles dimensions: {self.processed_data.shape}")
        
        return self.processed_data
    
    def apply_operation(self, operation, *args, **kwargs) -> np.ndarray:
        """
        Applique une opération sur les données de l'image.
        
        Args:
            operation: Fonction à appliquer sur les données
            *args, **kwargs: Arguments pour la fonction
            
        Returns:
            Données modifiées
        """
        if self.processed_data is None:
            raise ValueError("Aucune donnée chargée")
        
        self.processed_data = operation(self.processed_data, *args, **kwargs)
        return self.processed_data
    
    def adjust_brightness(self, factor: float) -> np.ndarray:
        """Ajuste la luminosité de l'image."""
        return self.apply_operation(lambda data: np.clip(data * factor, 
                                                        np.iinfo(data.dtype).min if data.dtype.kind in 'ui' else data.min(),
                                                        np.iinfo(data.dtype).max if data.dtype.kind in 'ui' else data.max()))
    
    def adjust_contrast(self, factor: float) -> np.ndarray:
        """Ajuste le contraste de l'image."""
        def contrast_op(data):
            mean_val = np.mean(data)
            return np.clip((data - mean_val) * factor + mean_val,
                          np.iinfo(data.dtype).min if data.dtype.kind in 'ui' else data.min(),
                          np.iinfo(data.dtype).max if data.dtype.kind in 'ui' else data.max())
        return self.apply_operation(contrast_op)
    
    def crop(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Recadre l'image."""
        if len(self.processed_data.shape) == 3:
            self.processed_data = self.processed_data[y1:y2, x1:x2, :]
        else:
            self.processed_data = self.processed_data[y1:y2, x1:x2]
        return self.processed_data
    
    def reset_to_original(self) -> None:
        """Remet les données à leur état original."""
        self.processed_data = self.original_data.copy()
        self.is_debayered = False
        self.is_color = len(self.original_shape) == 3 and self.original_shape[2] == 3
        print("Données remises à l'état original")
    
    def save_fits(self, output_filename: str, preserve_original_format: bool = True) -> None:
        """
        Sauvegarde l'image traitée au format FITS.
        
        Args:
            output_filename: Nom du fichier de sortie
            preserve_original_format: Si True, convertit l'image couleur en Bayer si l'original était Bayer
        """
        if self.processed_data is None:
            raise ValueError("Aucune donnée à sauvegarder")
        
        # Préparer les données à sauvegarder
        data_to_save = self.processed_data.copy()
        header_to_save = self.header.copy()
        
        # Si on veut préserver le format original et que l'image était Bayer
        if preserve_original_format and self.bayer_pattern and self.is_debayered:
            data_to_save = self._convert_to_bayer(data_to_save)
            print(f"Image convertie au format Bayer original: {self.bayer_pattern}")
        
        # Mettre à jour le header
        header_to_save['HISTORY'] = f'Processed with FitsImageManager on {Time.now().iso}'
        
        # Créer le HDU et sauvegarder
        hdu = fits.PrimaryHDU(data=data_to_save, header=header_to_save)
        hdu.writeto(output_filename, overwrite=True)
        
        print(f"Fichier sauvegardé: {output_filename}")
        print(f"Dimensions: {data_to_save.shape}")
    
    def _convert_to_bayer(self, color_image: np.ndarray) -> np.ndarray:
        """
        Convertit une image couleur RGB vers le format Bayer original.
        Cette fonction simule un capteur Bayer en échantillonnant les canaux appropriés.
        """
        if len(color_image.shape) != 3 or color_image.shape[2] != 3:
            return color_image
        
        h, w, _ = color_image.shape
        bayer_image = np.zeros((h, w), dtype=color_image.dtype)
        
        # Extraire les canaux RGB
        r_channel = color_image[:, :, 0]
        g_channel = color_image[:, :, 1] 
        b_channel = color_image[:, :, 2]
        
        # Appliquer le pattern Bayer
        if self.bayer_pattern == 'RGGB':
            bayer_image[0::2, 0::2] = r_channel[0::2, 0::2]  # R
            bayer_image[0::2, 1::2] = g_channel[0::2, 1::2]  # G
            bayer_image[1::2, 0::2] = g_channel[1::2, 0::2]  # G
            bayer_image[1::2, 1::2] = b_channel[1::2, 1::2]  # B
        elif self.bayer_pattern == 'BGGR':
            bayer_image[0::2, 0::2] = b_channel[0::2, 0::2]  # B
            bayer_image[0::2, 1::2] = g_channel[0::2, 1::2]  # G
            bayer_image[1::2, 0::2] = g_channel[1::2, 0::2]  # G
            bayer_image[1::2, 1::2] = r_channel[1::2, 1::2]  # R
        elif self.bayer_pattern == 'GRBG':
            bayer_image[0::2, 0::2] = g_channel[0::2, 0::2]  # G
            bayer_image[0::2, 1::2] = r_channel[0::2, 1::2]  # R
            bayer_image[1::2, 0::2] = b_channel[1::2, 0::2]  # B
            bayer_image[1::2, 1::2] = g_channel[1::2, 1::2]  # G
        elif self.bayer_pattern == 'GBRG':
            bayer_image[0::2, 0::2] = g_channel[0::2, 0::2]  # G
            bayer_image[0::2, 1::2] = b_channel[0::2, 1::2]  # B
            bayer_image[1::2, 0::2] = r_channel[1::2, 0::2]  # R
            bayer_image[1::2, 1::2] = g_channel[1::2, 1::2]  # G
        
        return bayer_image
    
    def get_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'image chargée."""
        if self.processed_data is None:
            return {"status": "Aucune image chargée"}
        
        return {
            "filename": self.filename,
            "original_shape": self.original_shape,
            "current_shape": self.processed_data.shape,
            "dtype": str(self.processed_data.dtype),
            "bayer_pattern": self.bayer_pattern,
            "is_color": self.is_color,
            "is_debayered": self.is_debayered,
            "data_range": (self.processed_data.min(), self.processed_data.max())
        }
    
    def save_as_image(self, output_filename: str, format: str = 'auto', 
                     quality: int = 95, stretch: bool = True,
                     stretch_percentiles: Tuple[float, float] = (1.0, 99.0)) -> None:
        """
        Sauvegarde l'image traitée au format PNG ou JPEG.
        
        Args:
            output_filename: Nom du fichier de sortie
            format: Format de sortie ('png', 'jpg', 'jpeg', 'auto')
            quality: Qualité JPEG (1-100, ignoré pour PNG)
            stretch: Appliquer un étirement automatique pour l'affichage
            stretch_percentiles: Percentiles pour l'étirement (bas, haut)
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow n'est pas installé. Utilisez: pip install Pillow")
        
        if self.processed_data is None:
            raise ValueError("Aucune donnée à sauvegarder")
        
        # Déterminer le format automatiquement si nécessaire
        if format == 'auto':
            ext = os.path.splitext(output_filename)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                format = 'jpeg'
            elif ext == '.png':
                format = 'png'
            elif ext == '.tif':
                format = 'tif'
            else:
                format = 'png'  # Par défaut
                if not output_filename.endswith('.png'):
                    output_filename += '.png'
        
        # Préparer les données pour l'export
        data_to_export = self.processed_data.copy()
        
        # Gestion des images couleur vs niveaux de gris
        if len(data_to_export.shape) == 3:
            # Image couleur (H, W, 3)
            if data_to_export.shape[2] != 3:
                raise ValueError(f"Format couleur non supporté: {data_to_export.shape}")
        elif len(data_to_export.shape) == 2:
            # Image en niveaux de gris - OK
            pass
        else:
            raise ValueError(f"Dimensions d'image non supportées: {data_to_export.shape}")
        
        # Conversion en float pour les calculs
        if data_to_export.dtype.kind in ['u', 'i']:
            # Entiers - normaliser selon le type
            if data_to_export.dtype == np.uint8:
                data_float = data_to_export.astype(np.float64) / 255.0
            elif data_to_export.dtype == np.uint16:
                data_float = data_to_export.astype(np.float64) / 65535.0
            else:
                data_float = data_to_export.astype(np.float64) / np.iinfo(data_to_export.dtype).max
        else:
            # Float - normaliser par la valeur max si > 1
            data_float = data_to_export.astype(np.float64)
            if data_float.max() > 1.0:
                data_float = data_float / data_float.max()
        
        # Appliquer un étirement pour améliorer l'affichage
        if stretch:
            if len(data_float.shape) == 3:
                # Image couleur - étirer chaque canal
                for i in range(3):
                    channel = data_float[:, :, i]
                    p_low = np.percentile(channel, stretch_percentiles[0])
                    p_high = np.percentile(channel, stretch_percentiles[1])
                    if p_high > p_low:
                        data_float[:, :, i] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            else:
                # Image en niveaux de gris
                p_low = np.percentile(data_float, stretch_percentiles[0])
                p_high = np.percentile(data_float, stretch_percentiles[1])
                if p_high > p_low:
                    data_float = np.clip((data_float - p_low) / (p_high - p_low), 0, 1)
        
        # Convertir en uint8 pour PIL
        
        data_uint8 = (np.clip(data_float, 0, 1) * 255).astype(np.uint8)
        
        # Créer l'image PIL
        if len(data_uint8.shape) == 3:
            # Image couleur RGB
            pil_image = Image.fromarray(data_uint8, mode='RGB')
        else:
            # Image en niveaux de gris
            pil_image = Image.fromarray(data_uint8, mode='L')
        
        # Sauvegarder selon le format
        if format.lower() in ['jpg', 'jpeg']:
            # Pour JPEG, convertir en RGB si nécessaire (pas de transparence)
            if pil_image.mode in ['RGBA', 'LA']:
                pil_image = pil_image.convert('RGB')
            pil_image.save(output_filename, format='JPEG', quality=quality, optimize=True)
        else:
            # PNG par défaut
            pil_image.save(output_filename, format='PNG', optimize=True)
        
        print(f"Image sauvegardée: {output_filename}")
        print(f"Format: {format.upper()}")
        print(f"Dimensions: {data_uint8.shape}")
        if format.lower() in ['jpg', 'jpeg']:
            print(f"Qualité: {quality}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer une instance du gestionnaire
    fits_manager = FitsImageManager()
    file='../../utils/01-observation-m16/01-images-initial/TargetSet.M27.8.00.LIGHT.329.2023-10-01_21-39-23.fits.fits'

    try:
        # Ouvrir un fichier FITS
        fits_manager.open_fits(file)
        
        # Afficher les informations
        print("\nInformations sur l'image:")
        info = fits_manager.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Définir le pattern Bayer si nécessaire
        # fits_manager.set_bayer_pattern('RGGB')
        
        # Debayeriser l'image si c'est une image Bayer
        if fits_manager.bayer_pattern:
            fits_manager.debayer(algorithm='bilinear')
        
        # Appliquer des modifications
        fits_manager.adjust_brightness(1.2)
        fits_manager.adjust_contrast(1.1)
        
        # Sauvegarder le résultat
        fits_manager.save_fits("resultat.fits", preserve_original_format=True)
        
    except Exception as e:
        print(f"Erreur: {e}")