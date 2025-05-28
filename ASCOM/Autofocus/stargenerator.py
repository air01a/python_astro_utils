import numpy as np
import cv2
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import random
from astropy.stats import sigma_clipped_stats

POSITIONS = list(range(500, 950, 40))  

def generate_star_field(size=512, num_stars=20):
    img = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng()
    for _ in range(num_stars):
        x, y = rng.integers(0, size, size=2)
        cv2.circle(img, (x, y), radius=random.randint(1,3), color=random.randint(1,255), thickness=-1)
    return img

def apply_focus_blur(image, focus_position, best_position=1020, blur_scale=0.01):
    deviation = abs(focus_position - best_position)
    sigma = deviation * blur_scale + 0.8  # un peu de flou même au mieux
    #print(sigma)
    return cv2.GaussianBlur(image, (0, 0), sigma)


def image_contrast(image):
    image_8 = np.clip(image, 0, 255).astype(np.uint8)
    laplacian = cv2.Laplacian(image_8, cv2.CV_64F)
    return laplacian.var()

def compute_fwhm(image):
    #bkg_sigma = mad_std(image)
    mean, median, bkg_sigma = sigma_clipped_stats(image, sigma=3.0)

    daofind = DAOStarFinder(fwhm=3.0, threshold=5. * bkg_sigma)
    sources = daofind(image)
    print(sources)
    if sources is None or len(sources) == 0:
        return None

    return np.mean(sources['sharpness'])

def autofocus_scan(img_stars, method="sharpness", positions=POSITIONS):
    fwhm_values = []
    valid_positions = []
    #random.shuffle(POSITIONS)
    for pos in (positions):
        print(f"Focuser → {pos}")
        img =  apply_focus_blur(img_stars, pos)
        cv2.imshow("Simulated", img)
        cv2.waitKey(0)
        if method == "contrast":
            fwhm = image_contrast(img)
        else:
            fwhm = compute_fwhm(img)
        if fwhm is not None:
            print(f"FWHM = {fwhm:.2f}")
            fwhm_values.append(fwhm)
            valid_positions.append(pos)
        else:
            print("❌ Échec détection étoiles.")

    return valid_positions, fwhm_values

# Courbe modèle : parabole
def parabola(x, a, b, c):
    return a * (x - b)**2 + c

def fit_focus_curve(positions, fwhms):
    popt, _ = curve_fit(parabola, positions, fwhms)
    best_pos = popt[1]
    return popt, best_pos

def plot_focus_curve(positions, fwhms, fit_params):
    xfit = np.linspace(min(positions), max(positions), 200)
    yfit = parabola(xfit, *fit_params)
    plt.plot(positions, fwhms, 'bo', label="Données")
    plt.plot(xfit, yfit, 'r-', label="Ajustement parabole")
    plt.xlabel("Position du focuser")
    plt.ylabel("FWHM moyen")
    plt.title("Courbe de mise au point")
    plt.legend()
    plt.grid()
    plt.show()


def find_best_focus_position(img, positions, method="sharpness"):
    pos, fwhm = autofocus_scan(img,method, positions)
    if len(pos) >= 3:
        fit_params, best = fit_focus_curve(pos, fwhm)
        plot_focus_curve(pos, fwhm, fit_params)
        return int(round(best))
    return None

if __name__ == "__main__":
    img = generate_star_field(size=512, num_stars=20)
    print("✅ First loop to detect potentially good focus starting point.")
    positions = list(range(100, 2000, 100))
    best_position = find_best_focus_position(img, positions, "contrast")
    if best_position is not None:
        print(f"🔍 Best starting position found :  : {best_position}")
        positions=list(range(best_position-100, best_position+100, 20))
        print("✅ Second loop to refine the focus position.")
        best_position = find_best_focus_position(img, positions)
        if best_position is not None:
            print(f"🔍 Best focus position found : {best_position}")
        else:
            print("❌ No valid focus position found.")
    else:
            print("❌ No valid intial focus position found.")
