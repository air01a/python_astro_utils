import numpy as np
from astropy.stats import sigma_clip, mad_std
import cv2
from astropy.io import fits
import astroalign as aa
import logging
class WeightedLiveStacker:
    def __init__(self, shape, initial_stack=20, sigma=3.0, saturation_limit=60000, dead_threshold=0.9, detection_sigma=5.0, max_control_points=50):
        self.shape = shape
        self.initial_stack = initial_stack
        self.sigma = sigma
        self.saturation_limit = saturation_limit
        self.dead_threshold = dead_threshold

        self.stack_phase1 = []
        self.mean_image = np.zeros(shape, dtype=np.float32)
        self.weight_total = np.zeros(shape, dtype=np.float32)
        self.frame_count = 0

        self.bad_pixel_counter = np.zeros(shape, dtype=np.uint16)
        self.valid_mask = np.ones(shape, dtype=bool)
        self.detection_sigma = detection_sigma

        self.max_control_points = max_control_points

        self.phase = 'initial'
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def detect_bad_pixels(self, image):
        is_black = image <= 0
        is_saturated = image >= self.saturation_limit
        self.bad_pixel_counter += (is_black | is_saturated).astype(np.uint16)

        if self.frame_count >= self.initial_stack:
            bad_mask = (self.bad_pixel_counter / self.frame_count) >= self.dead_threshold
            self.valid_mask = ~bad_mask

    def estimate_noise(self, image):
        # Calcul global du bruit (ou par bloc si tu veux raffiner)
        return mad_std(image[self.valid_mask])  # plus robuste que std

    def add_frame(self, image: np.ndarray):
        print(f"[DEBUG] Frame {self.frame_count} shape: {image.shape}")

        image = image.astype(np.float32)
        self.detect_bad_pixels(image)
        if self.frame_count!=0:
            image = self.align_image_astroalign(image, self.mean_image)
        self.frame_count += 1

        if self.phase == 'initial':
            self.stack_phase1.append(image)

            try:
                stack = np.stack(self.stack_phase1, axis=0)
            except ValueError as e:
                print("Erreur d'empilement : les images n'ont pas toutes la même shape.")
                for i, img in enumerate(self.stack_phase1):
                    print(f" - Image {i} shape: {img.shape}")
                raise e

            self.mean_image = np.mean(stack, axis=0)

            if self.frame_count == self.initial_stack:
                self.phase = 'restart'
        
        elif self.phase == 'restart':
            self.stack_phase1.append(image)
            stack = np.stack(self.stack_phase1, axis=0)
            clipped = sigma_clip(stack, sigma=self.sigma, axis=0)
            clipped_masked = np.where(self.valid_mask, clipped, np.nan)
            self.mean_image = np.nanmean(clipped_masked, axis=0)
            self.weight_total[:] = 1.0  # Pondération de base uniforme
            self.stack_phase1 = []
            self.phase = 'incremental'

        else:  # incremental
            noise = self.estimate_noise(image)
            if noise == 0:
                weight = 1.0
            else:
                weight = 1.0 / (noise ** 2)

            stack = np.stack([self.mean_image, image], axis=0)
            clipped = sigma_clip(stack, sigma=self.sigma, axis=0)
            valid = ~np.isnan(clipped[1]) & self.valid_mask

            # Mise à jour pondérée
            self.mean_image[valid] = (
                self.mean_image[valid] * self.weight_total[valid] + image[valid] * weight
            ) / (self.weight_total[valid] + weight)

            self.weight_total[valid] += weight

        return self.mean_image


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
                data = hdul[0].data.astype(np.float64)

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
    
    def save_stacked_image(self, image, filepath):
        """Sauvegarde l'image stackée en FITS"""
        #if self.mean_image is None:
        #    self.logger.error("Aucune image stackée à sauvegarder")
        #    return False
        
        try:
            hdu = fits.PrimaryHDU(image)
            hdu.header['FRAMES'] = self.frame_count
            #hdu.header['SIGMA'] = self.sigma_clip
            #hdu.header['BOOTSTR'] = self.bootstrap_frames
            hdu.header['ISCOLOR'] = self.is_color
            hdu.header['DETSIG'] = self.detection_sigma
            hdu.writeto(filepath, overwrite=True)
            self.logger.info(f"Image sauvegardée: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False

if __name__ == "__main__":
    from pathlib import Path

    # Initialise le stacker
    stacker = WeightedLiveStacker((3, 1096, 1936))
    image = stacker.load_fits_image('../../utils/01-observation-m16/01-images-initial/TargetSet.M27.8.00.LIGHT.329.2023-10-01_21-39-23.fits.fits')
    stacker.save_stacked_image(image, 'test.fit')
    print(" eeeeee ")
    # Simule l'arrivée d'images
    image_dir = Path("../../utils/01-observation-m16/01-images-initial")  # Dossier contenant les images FITS
    
    if image_dir.exists():
        fits_files = sorted(image_dir.glob("*.fits"))
        
        for i, fits_file in enumerate(fits_files):
            print(f"Traitement de {fits_file.name}...")
            image = stacker.add_frame(stacker.load_fits_image(fits_file))
            
            if image is not None:
                # Sauvegarde périodique
                if i % 10 == 0:
                    stacker.save_stacked_image(f"stacked_frame_{i:04d}.fits")
                    

        
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