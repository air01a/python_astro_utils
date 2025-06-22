import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astroalign as aa
from pathlib import Path
import logging
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
HOT_PIXEL_RATIO = 2    
def debug_display(aligned_image, footprint):
    plt.subplot(1,2,1)
    plt.imshow(aligned_image[0] if aligned_image.ndim == 3 else aligned_image, cmap='gray')
    plt.title("Aligned image")
    plt.subplot(1,2,2)
    plt.imshow(footprint, cmap='gray')
    plt.title("Footprint")
    plt.show()


class LiveStacker:
    def __init__(self, sigma_clip=3.0, bootstrap_frames=20, detection_sigma=5.0, max_control_points=50):
        """
        Initialiseur du live stacker avec astroalign
        
        Args:
            sigma_clip: Seuil de sigma clipping (défaut: 3.0)
            bootstrap_frames: Nombre d'images pour établir la référence initiale
            detection_sigma: Seuil de détection d'étoiles pour astroalign (défaut: 5.0)
            max_control_points: Nombre max de points de contrôle pour astroalign
        """
        self.sigma_clip = sigma_clip
        self.bootstrap_frames = bootstrap_frames
        self.detection_sigma = detection_sigma
        self.max_control_points = max_control_points
        
        # Données de stacking
        self.reference_image = None
        self.stacked_image = None
        self.weight_map = None  # Carte des poids (nombre d'images par pixel)
        self.noise_map = None   # Carte du bruit de fond
        
        # Historique pour le bootstrap
        self.bootstrap_images = []
        self.frame_count = 0
        self.is_color = False  # Déterminé automatiquement
        self.bad_pixel_mask = None
        # Configuration logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def _neighbors_average(self,data):
        """
        returns an array containing the means of all original array's pixels' neighbors
        :param data: the image to compute means for
        :return: an array containing the means of all original array's pixels' neighbors
        :rtype: numpy.Array
        """

        kernel = np.ones((3, 3))
        kernel[1, 1] = 0

        neighbor_sum = convolve2d(data, kernel, mode='same', boundary='fill', fillvalue=0)
        num_neighbor = convolve2d(np.ones(data.shape), kernel, mode='same', boundary='fill', fillvalue=0)

        return (neighbor_sum / num_neighbor).astype(data.dtype)


    def hot_pixel_remover(self, image):

        # the idea is to check every pixel value against its 8 neighbors
        # if its value is more than _HOT_RATIO times the mean of its neighbors' values
        # me replace its value with that mean

        # this can only work on B&W or non-debayered color images


        if not self.is_color:
            means = self._neighbors_average(image)
            try:
                image = np.where(image / (means+0.00001) > HOT_PIXEL_RATIO, means, image)
            except Exception as exc:
                self.logger.error("Error during hotpixelremover :( %s"%str(exc))
        else:
            print("Hot Pixel Remover cannot work on debayered color images.")
        return image

    def debayer_image(self, raw_data, pattern='RGGB', white_balance=(0.81, 0.7154, 0.7721)):
        raw_uint16 = raw_data.astype(np.uint16)

        code = {
            'RGGB': cv2.COLOR_BAYER_RG2RGB,
            'BGGR': cv2.COLOR_BAYER_BG2RGB,
            'GRBG': cv2.COLOR_BAYER_GR2RGB,
            'GBRG': cv2.COLOR_BAYER_GB2RGB
        }[pattern]

        # Débayerisation en uint16, puis conversion en float pour traitement
        rgb = cv2.cvtColor(raw_uint16, code)
        rgb = rgb.astype(np.float64)

        # Appliquer la balance des blancs
        r_mul, g_mul, b_mul = white_balance
        rgb[:, :, 0] *= r_mul  # R
        rgb[:, :, 1] *= g_mul  # G
        rgb[:, :, 2] *= b_mul  # B

        # Clipping pour rester dans la plage 16 bits
        rgb = np.clip(rgb, 0, 65535)

        # Format (3, H, W) pour stacking
        return rgb.transpose(2, 0, 1)
    

    def load_fits_image(self, filepath):
        """Charge une image FITS et retourne les données, avec débayerisation si nécessaire"""
        try:
            self.is_color=False
            with fits.open(filepath) as hdul:
                data = self.hot_pixel_remover(hdul[0].data.astype(np.float64))

                #data = hdul[0].data.astype(np.float64)
                #if 'BZERO' in hdul[0].header:
                #    data -= hdul[0].header['BZERO'] 
                if data.ndim == 2:
                    header = hdul[0].header
                    if 'BAYERPAT' in header:

                        self.is_color = True
                        pattern = header['BAYERPAT']

                        self.logger.error(f"{pattern}")

                        data = self.debayer_image(data, pattern=pattern)
                        self.logger.info(f"Image débayerisée (pattern {pattern}) : {data.shape}")
                    else:
                        self.is_color = False
                        self.logger.info(f"Image N&B détectée : {data.shape}")
                elif data.ndim == 3:
                    self.is_color = True
                    self.logger.info(f"Image couleur détectée : {data.shape}")
                else:
                    raise ValueError(f"Format d'image non supporté : {data.ndim} dimensions")
 
                return data
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {filepath}: {e}")
            return None
    
    def get_luminance(self, image):
        """Extrait la luminance d'une image (pour l'alignement)"""
        if self.is_color:
            if image.ndim == 3:
                # Conversion RGB vers luminance (formule standard)
                return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                raise ValueError("Image couleur attendue mais format incorrect")
        else:
            return image
    
    def estimate_background_noise(self, image):
        """Estime le bruit de fond de l'image"""
        if self.is_color:
            # Pour les images couleur, calcule le bruit sur chaque canal
            noise_per_channel = []
            for channel in range(image.shape[0]):
                _, _, std = sigma_clipped_stats(image[channel], sigma=self.sigma_clip)
                noise_per_channel.append(std)
            return np.array(noise_per_channel)
        else:
            # Pour les images N&B
            _, _, std = sigma_clipped_stats(image, sigma=self.sigma_clip)
            return std
    
    def align_image_astroalign(self, image, reference):
        """Aligne l'image avec la référence en utilisant astroalign"""
        try:
            # Utilise la luminance pour l'alignement
            source_lum = self.get_luminance(image)
            target_lum = self.get_luminance(reference)
            
            # Alignement avec astroalign
            aligned_lum, footprint = aa.register(
                source_lum, 
                target_lum,
                detection_sigma=self.detection_sigma,
                max_control_points=self.max_control_points
            )
            
            # Si c'est une image couleur, applique la même transformation aux 3 canaux
            if self.is_color:
                # Récupère la transformation
                transform,_ = aa.find_transform(source_lum, target_lum)
                aligned_image = np.zeros_like(image)
                
                for channel in range(image.shape[0]):
                    aligned_image[channel] = aa.apply_transform(
                        transform, image[channel], target_lum
                    )[0]
                
                return aligned_image, footprint
            else:
                return aligned_lum, footprint
                
        except aa.MaxIterError:
            self.logger.warning("Astroalign: Nombre maximum d'itérations atteint")
            return image, np.ones_like(self.get_luminance(image))
        except Exception as e:
            self.logger.warning(f"Erreur d'alignement astroalign: {e}")
            return image, np.ones_like(self.get_luminance(image))
    
    def sigma_clip_pixel(self, pixel_values, weights=None):
        """Applique le sigma clipping sur un ensemble de valeurs de pixels"""
        if len(pixel_values) < 3:
            return np.mean(pixel_values), np.sum(weights) if weights is not None else len(pixel_values)
        
        # Calcul des statistiques pondérées si nécessaire
        if weights is not None:
            mean = np.average(pixel_values, weights=weights)
            variance = np.average((pixel_values - mean)**2, weights=weights)
            std = np.sqrt(variance)
        else:
            mean = np.mean(pixel_values)
            std = np.std(pixel_values)
        
        # Masque de sigma clipping
        mask = np.abs(pixel_values - mean) <= self.sigma_clip * std
        
        if np.sum(mask) == 0:
            return mean, np.sum(weights) if weights is not None else len(pixel_values)
        
        clipped_values = pixel_values[mask]
        clipped_weights = weights[mask] if weights is not None else None
        
        if clipped_weights is not None:
            final_value = np.average(clipped_values, weights=clipped_weights)
            total_weight = np.sum(clipped_weights)
        else:
            final_value = np.mean(clipped_values)
            total_weight = len(clipped_values)
        
        return final_value, total_weight
    
    def bootstrap_reference(self):
        """Crée l'image de référence à partir des premières images (optimisé NumPy)"""
        if len(self.bootstrap_images) < self.bootstrap_frames:
            return False

        self.logger.info(f"Création de la référence avec {len(self.bootstrap_images)} images")

        reference_candidate = self.bootstrap_images[0]
        aligned_images = [reference_candidate]
        valid_footprints = [np.ones_like(self.get_luminance(reference_candidate))]

        for i, img in enumerate(self.bootstrap_images[0:-1], 1):
            aligned_img, footprint = self.align_image_astroalign(img, reference_candidate)
            aligned_images.append(aligned_img)
            valid_footprints.append(footprint)

        if self.is_color:
            channels = []
            weights = []
            for c in range(3):  # On suppose RGB en (3, H, W)
                channel_stack = np.stack([img[c] for img in aligned_images])  # (N, H, W)
                footprint_stack = np.stack(valid_footprints)  # (N, H, W)
                masked = np.where(footprint_stack > 0, channel_stack, np.nan)

                mean = np.nanmean(masked, axis=0)
                std = np.nanstd(masked, axis=0)
                mask = np.abs(masked - mean[None, :, :]) <= self.sigma_clip * std[None, :, :]

                clipped = np.where(mask, masked, np.nan)
                final = np.nanmean(clipped, axis=0)
                weight = np.sum(~np.isnan(clipped), axis=0)

                channels.append(final)
                weights.append(weight)

            self.stacked_image = np.stack(channels)
            self.weight_map = np.stack(weights)
        else:
            image_stack = np.stack(aligned_images)  # (N, H, W)
            footprint_stack = np.stack(valid_footprints)  # (N, H, W)
            masked = np.where(footprint_stack > 0, image_stack, np.nan)

            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0)
            mask = np.abs(masked - mean[None, :, :]) <= self.sigma_clip * std[None, :, :]

            clipped = np.where(mask, masked, np.nan)
            self.stacked_image = np.nanmean(clipped, axis=0)
            self.weight_map = np.sum(~np.isnan(clipped), axis=0)

        self.reference_image = reference_candidate
        if self.is_color:
            # Convertir les 3 sigma par canal en carte bruit homogène
            noise_per_channel = self.estimate_background_noise(self.reference_image)
            self.noise_map = np.zeros_like(self.reference_image)
            for c in range(3):
                self.noise_map[c] = noise_per_channel[c]
        else:
            sigma = self.estimate_background_noise(self.reference_image)
            self.noise_map = np.full_like(self.reference_image, sigma)

        self.logger.info("Référence créée avec succès")
        return True

    
    def process_new_frame(self, new_image):
        noise_level = self.estimate_background_noise(new_image)
        aligned_image, footprint = self.align_image_astroalign(new_image, self.reference_image)
        if self.is_color:
            image_weight = 1.0 / (np.mean(noise_level) ** 2)
            image_weight=1.0
            print("image weight",image_weight)
            for c in range(3):
                valid_mask =  (footprint == 0)
                new_values = aligned_image[c]
                current_values = self.stacked_image[c]
                current_weights = self.weight_map[c]
                current_noise = self.noise_map[c]

                total_weight = current_weights + image_weight
                updated_values = (current_values * current_weights + new_values * image_weight) / total_weight
                updated_noise = np.sqrt((current_noise**2 * current_weights +
                                        noise_level[c]**2 * image_weight) / total_weight)

                self.stacked_image[c][valid_mask] = updated_values[valid_mask]
                self.weight_map[c][valid_mask] = total_weight[valid_mask]
                self.noise_map[c][valid_mask] = updated_noise[valid_mask]
        else:
            image_weight = 1.0 / (noise_level ** 2)
            valid_mask = footprint > 0

            new_values = aligned_image
            current_values = self.stacked_image
            current_weights = self.weight_map
            current_noise = self.noise_map

            # S'assurer que noise_map est bien un tableau
            if np.isscalar(current_noise):
                current_noise = np.full_like(current_values, current_noise)
                self.noise_map = current_noise

            noise_map_frame = np.full_like(current_values, noise_level)

            total_weight = current_weights + image_weight
            updated_values = (current_values * current_weights + new_values * image_weight) / total_weight
            updated_noise = np.sqrt((current_noise**2 * current_weights +
                                    noise_map_frame**2 * image_weight) / total_weight)

            self.stacked_image[valid_mask] = updated_values[valid_mask]
            self.weight_map[valid_mask] = total_weight[valid_mask]
            self.noise_map[valid_mask] = updated_noise[valid_mask]

    
    def _update_pixel(self, channel, y, x, current_value, current_weight, 
                     new_value, image_weight, noise_level):
        """Met à jour un pixel individuel avec sigma clipping"""
        if current_weight > 0:
            # Récupère le bruit actuel
            if self.is_color:
                current_noise = self.noise_map[channel, y, x]
            else:
                current_noise = self.noise_map[y, x]
            
            # Vérifie le sigma clipping
            diff = abs(new_value - current_value)
            if diff <= self.sigma_clip * current_noise:
                # Combine les valeurs avec pondération
                total_weight = current_weight + image_weight
                combined_value = (current_value * current_weight + 
                                new_value * image_weight) / total_weight
                
                # Met à jour les valeurs
                if self.is_color:
                    self.stacked_image[channel, y, x] = combined_value
                    self.weight_map[channel, y, x] = total_weight
                    self.noise_map[channel, y, x] = np.sqrt(
                        (current_noise**2 * current_weight + 
                         noise_level**2 * image_weight) / total_weight
                    )
                else:
                    self.stacked_image[y, x] = combined_value
                    self.weight_map[y, x] = total_weight
                    self.noise_map[y, x] = np.sqrt(
                        (current_noise**2 * current_weight + 
                         noise_level**2 * image_weight) / total_weight
                    )
        else:
            # Premier pixel valide à cette position
            if self.is_color:
                self.stacked_image[channel, y, x] = new_value
                self.weight_map[channel, y, x] = image_weight
                self.noise_map[channel, y, x] = noise_level
            else:
                self.stacked_image[y, x] = new_value
                self.weight_map[y, x] = image_weight
                self.noise_map[y, x] = noise_level
    
    def add_frame(self, filepath):
        """Ajoute une nouvelle image au stack"""
        image = self.load_fits_image(filepath)
        if image is None:
            return False
        
        self.frame_count += 1
        
        # Phase de bootstrap
        if len(self.bootstrap_images) < self.bootstrap_frames:
            self.bootstrap_images.append(image)
            self.logger.info(f"Bootstrap: {len(self.bootstrap_images)}/{self.bootstrap_frames}")
            
            if len(self.bootstrap_images) == self.bootstrap_frames:
                return self.bootstrap_reference()
        else:
            # Phase de stacking en temps réel
            print("Process_new_frame")
            self.process_new_frame(image)
        
        return True
    
    def get_stacked_image(self):
        """Retourne l'image stackée actuelle"""
        return self.stacked_image.copy() if self.stacked_image is not None else None
    
    def get_weight_map(self):
        """Retourne la carte des poids"""
        return self.weight_map.copy() if self.weight_map is not None else None
    
    def save_stacked_image(self, filepath):
        """Sauvegarde l'image stackée en FITS"""
        if self.stacked_image is None:
            self.logger.error("Aucune image stackée à sauvegarder")
            return False
        
        try:
            hdu = fits.PrimaryHDU(self.stacked_image)
            hdu.header['FRAMES'] = self.frame_count
            hdu.header['SIGMA'] = self.sigma_clip
            hdu.header['BOOTSTR'] = self.bootstrap_frames
            hdu.header['ISCOLOR'] = self.is_color
            hdu.header['DETSIG'] = self.detection_sigma
            hdu.writeto(filepath, overwrite=True)
            self.logger.info(f"Image sauvegardée: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False

from processing import AstroImageProcessor

processor = AstroImageProcessor()
# Exemple d'utilisation
if __name__ == "__main__":
    # Initialise le stacker
    stacker = LiveStacker(
        sigma_clip=3.0, 
        bootstrap_frames=10,
        detection_sigma=3.0,
        max_control_points=50
    )
    
    # Simule l'arrivée d'images
    image_dir = Path("../../utils/01-observation-m16/01-images-initial")  # Dossier contenant les images FITS
    
    if image_dir.exists():
        fits_files = sorted(image_dir.glob("*.fits"))
        
        for i, fits_file in enumerate(fits_files):
            print(f"Traitement de {fits_file.name}...")
            success = stacker.add_frame(fits_file)
            
            if success and stacker.get_stacked_image() is not None:
                # Sauvegarde périodique
                if i % 10 == 0:
                    stacker.save_stacked_image(f"stacked_frame_{i:04d}.fits")
                    processed = processor.process_image(
                        f"stacked_frame_{i:04d}.fits",
                        stretch_method='midtones',
                        midtone=0.3,
                        enhance_contrast=True,
                        contrast_method='clahe'
                    )
        
                    if processed is not None:
                        # Sauvegarde en JPG
                        processor.save_jpg(processed, f"stacked_frame_{i:04d}.jpg", quality=95)

        
        # Sauvegarde finale
        stacker.save_stacked_image("final_stacked.fits")
        
        print(f"Stacking terminé avec {stacker.frame_count} images")
        if stacker.is_color:
            print(f"Images couleur traitées")
            print(f"Poids moyen par pixel et canal: {np.mean(stacker.get_weight_map()):.2f}")
        else:
            print(f"Images N&B traitées")
            print(f"Poids moyen par pixel: {np.mean(stacker.get_weight_map()):.2f}")
    else:
        print("Dossier 'images' non trouvé")
        print("Installez astroalign avec: pip install astroalign")