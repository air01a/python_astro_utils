import httpx
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from enum import Enum
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Générateur thread-safe pour les IDs de transaction
class TransactionIDGenerator:
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()

    def get_next_id(self) -> int:
        with self._lock:
            self._counter += 1
            if self._counter > 4294967295:
                self._counter = 1
            return self._counter

_transaction_id_generator = TransactionIDGenerator()




class ASCOMDeviceType(str, Enum):
    TELESCOPE = "telescope"
    CAMERA = "camera"
    FOCUSER = "focuser"
    FILTERWHEEL = "filterwheel"
    DOME = "dome"
    ROTATOR = "rotator"

class BaseDeviceInfo(BaseModel):
    name: str
    description: str
    driver_info: str
    driver_version: str
    interface_version: int
    supported_actions: List[str]
    connected: bool

class ASCOMAlpacaBaseClient:
    """Client synchrone ASCOM Alpaca"""
    def __init__(self, device_type: ASCOMDeviceType, host="localhost", port=11111, device_number=0, client_id=1001):
        self.device_type = device_type
        self.host = host
        self.port = port
        self.device_number = device_number
        self.client_id = client_id
        self.base_url = f"http://{host}:{port}/api/v1/{device_type.value}/{device_number}"
        self.client = httpx.Client(timeout=30.0)

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        if data is None:
            data = {}

        transaction_id = _transaction_id_generator.get_next_id()
        data.update({
            "ClientID": self.client_id,
            "ClientTransactionID": transaction_id
        })

        try:
            if method.upper() == "GET":
                response = self.client.get(url, params=data)
            else:
                response = self.client.put(url, data=data)

            response.raise_for_status()
            result = response.json()

            if result.get("ErrorNumber", 0) != 0:
                raise Exception(f"Erreur ASCOM {result['ErrorNumber']}: {result.get('ErrorMessage', 'Erreur inconnue')}")

            return result

        except httpx.RequestError as e:
            logger.error(f"Erreur de connexion {self.device_type.value}: {e}")
            raise Exception(f"Erreur de connexion au {self.device_type.value}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP {e.response.status_code} {self.device_type.value}: {e.response.text}")
            raise Exception(f"Erreur HTTP {self.device_type.value}: {e.response.status_code}")

    def connect(self) -> bool:
        result = self._make_request("PUT", "connected", {"Connected": True})
        return result.get("Value", False)

    def disconnect(self) -> bool:
        result = self._make_request("PUT", "connected", {"Connected": False})
        return not result.get("Value", True)

    def is_connected(self) -> bool:
        result = self._make_request("GET", "connected")
        return result.get("Value", False)

    def get_device_info(self) -> BaseDeviceInfo:
        results = [
            self._make_request("GET", "name"),
            self._make_request("GET", "description"),
            self._make_request("GET", "driverinfo"),
            self._make_request("GET", "driverversion"),
            self._make_request("GET", "interfaceversion"),
            self._make_request("GET", "supportedactions"),
            self.is_connected()
        ]

        return BaseDeviceInfo(
            name=results[0].get("Value", ""),
            description=results[1].get("Value", ""),
            driver_info=results[2].get("Value", ""),
            driver_version=results[3].get("Value", ""),
            interface_version=results[4].get("Value", 0),
            supported_actions=results[5].get("Value", []),
            connected=results[6]
        )

    def execute_action(self, action: str, parameters: str = "") -> str:
        result = self._make_request("PUT", "action", {
            "Action": action,
            "Parameters": parameters
        })
        return result.get("Value", "")
    

# ===== CLIENT TÉLESCOPE =====

class TelescopeState(str, Enum):
    PARKED = "parked"
    TRACKING = "tracking"
    SLEWING = "slewing"
    STOPPED = "stopped"

class AlignmentMode(int, Enum):
    ALT_AZ = 0
    POLAR = 1
    GERMAN_POLAR = 2

class GuideDirection(int, Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class TelescopeAxis(int, Enum):
    PRIMARY:0
    SECONDARY:1
    TERTIARY:2

class TelescopeInfo(BaseDeviceInfo):
    """Informations spécifiques au télescope"""
    alignment_mode: AlignmentMode
    aperture_area: float
    aperture_diameter: float
    can_find_home: bool
    can_park: bool
    can_pulse_guide: bool
    can_set_guide_rates: bool
    can_set_park: bool
    can_set_pier_side: bool
    can_set_tracking: bool
    can_slew: bool
    can_slew_altaz: bool
    can_slew_altaz_async: bool
    can_slew_async: bool
    can_sync: bool
    can_sync_altaz: bool
    can_unpark: bool
    declination_rate: float
    does_refraction: bool
    equatorial_system: int
    focal_length: float
    guide_rate_declination: float
    guide_rate_right_ascension: float
    has_pier_side: bool
    right_ascension_rate: float
    site_elevation: float
    site_latitude: float
    site_longitude: float
    slew_settle_time: int
    target_declination: float
    target_right_ascension: float
    tracking_rate: int
    tracking_rates: List[int]
    utc_date: str

class TelescopePosition(BaseModel):
    right_ascension: float
    declination: float
    altitude: float
    azimuth: float
    side_of_pier: int
    tracking: bool
    slewing: bool

class ASCOMAlpacaTelescopeClient(ASCOMAlpacaBaseClient):
    """Client ASCOM Alpaca synchrone pour télescope"""

    def __init__(self, host="localhost", port=11111, device_number=0):
        super().__init__(ASCOMDeviceType.TELESCOPE, host, port, device_number)

    def get_telescope_info(self) -> TelescopeInfo:
        """Récupère les informations complètes du télescope"""
        base_info = self.get_device_info()

        try:
            results = [
                self._make_request("GET", "alignmentmode"),
                self._make_request("GET", "aperturearea"),
                self._make_request("GET", "aperturediameter"),
                self._make_request("GET", "canfindhome"),
                self._make_request("GET", "canpark"),
                self._make_request("GET", "canpulseguide"),
            ]

            return TelescopeInfo(
                **base_info.dict(),
                alignment_mode=AlignmentMode(results[0].get("Value", 0)),
                aperture_area=results[1].get("Value", 0.0),
                aperture_diameter=results[2].get("Value", 0.0),
                can_find_home=results[3].get("Value", False),
                can_park=results[4].get("Value", False),
                can_pulse_guide=results[5].get("Value", False),
                # Valeurs par défaut pour les propriétés non récupérées
                can_set_guide_rates=False,
                can_set_park=False,
                can_set_pier_side=False,
                can_set_tracking=True,
                can_slew=True,
                can_slew_altaz=False,
                can_slew_altaz_async=False,
                can_slew_async=True,
                can_sync=True,
                can_sync_altaz=False,
                can_unpark=False,
                declination_rate=0.0,
                does_refraction=True,
                equatorial_system=0,
                focal_length=0.0,
                guide_rate_declination=0.0,
                guide_rate_right_ascension=0.0,
                has_pier_side=False,
                right_ascension_rate=0.0,
                site_elevation=0.0,
                site_latitude=0.0,
                site_longitude=0.0,
                slew_settle_time=0,
                target_declination=0.0,
                target_right_ascension=0.0,
                tracking_rate=0,
                tracking_rates=[],
                utc_date=""
            )
        except Exception as e:
            logger.warning(f"Impossible de récupérer toutes les propriétés du télescope: {e}")
            return TelescopeInfo(**base_info.dict(), **{
                attr: getattr(TelescopeInfo.__fields__[attr], 'default', None)
                for attr in TelescopeInfo.__fields__ if attr not in base_info.dict()
            })

    def get_position(self) -> TelescopePosition:
        """Récupère la position actuelle du télescope"""
        results = [
            self._make_request("GET", "rightascension"),
            self._make_request("GET", "declination"),
            self._make_request("GET", "altitude"),
            self._make_request("GET", "azimuth"),
            self._make_request("GET", "sideofpier"),
            self._make_request("GET", "tracking"),
            self._make_request("GET", "slewing")
        ]

        return TelescopePosition(
            right_ascension=results[0].get("Value", 0.0),
            declination=results[1].get("Value", 0.0),
            altitude=results[2].get("Value", 0.0),
            azimuth=results[3].get("Value", 0.0),
            side_of_pier=results[4].get("Value", 0),
            tracking=results[5].get("Value", False),
            slewing=results[6].get("Value", False)
        )

    def slew_to_coordinates(self, ra: float, dec: float) -> None:
        """Pointe le télescope vers les coordonnées RA/Dec (synchrone)"""
        self._make_request("PUT", "slewtocoordinates", {
            "RightAscension": ra,
            "Declination": dec
        })

    def slew_to_coordinates_async(self, ra: float, dec: float) -> None:
        """Pointe le télescope vers les coordonnées RA/Dec (asynchrone)"""
        self._make_request("PUT", "slewtocoordinatesasync", {
            "RightAscension": ra,
            "Declination": dec
        })

    def set_utc_date(self, date: str) -> None:
        self._make_request("PUT", "utcdate", {
            "UTCDate": date
        })

    def get_utc_date(self) -> str:
        result = self._make_request("GET", "utcdate")
        return result.get("Value", "")

    def sync_to_coordinates(self, ra: float, dec: float) -> None:
        self._make_request("PUT", "synctocoordinates", {
            "RightAscension": ra,
            "Declination": dec
        })

    def abort_slew(self) -> None:
        self._make_request("PUT", "abortslew")

    def set_tracking(self, enabled: bool) -> None:
        self._make_request("PUT", "tracking", {"Tracking": enabled})

    def is_tracking(self) -> bool:
        result = self._make_request("GET", "tracking")
        return result.get("Value", False)

    def park(self) -> None:
        self._make_request("PUT", "park")

    def unpark(self) -> None:
        self._make_request("PUT", "unpark")

    def is_parked(self) -> bool:
        result = self._make_request("GET", "atpark")
        return result.get("Value", False)

    def is_slewing(self) -> bool:
        result = self._make_request("GET", "slewing")
        return result.get("Value", False)

    def move_axis(self, axis: int, rate: float) -> None:
        self._make_request("PUT", "moveaxis", {"Axis": axis, "Rate": rate})

    def set_latitude(self, latitude: float) -> None:
        self._make_request("PUT", "sitelatitude", {"SiteLatitude": latitude})

    def set_longitude(self, longitude: float) -> None:
        self._make_request("PUT", "sitelongitude", {"SiteLongitude": longitude})

    def set_elevation(self, elevation: float) -> None:
        self._make_request("PUT", "siteelevation", {"SiteElevation": elevation})

    def get_altitude(self) -> float:
        result = self._make_request("GET", "altitude")
        return result.get("Value", 0.0)

# ===== CLIENT CAMÉRA =====

class CameraState(int, Enum):
    IDLE = 0
    WAITING = 1
    EXPOSING = 2
    READING = 3
    DOWNLOAD = 4
    ERROR = 5

class SensorType(int, Enum):
    MONOCHROME = 0
    COLOR = 1
    RGGB = 2
    CMYG = 3
    CMYG2 = 4
    LRGB = 5

class CameraInfo(BaseDeviceInfo):
    """Informations spécifiques à la caméra"""
    camera_x_size: int
    camera_y_size: int
    max_bin_x: int
    max_bin_y: int
    pixel_size_x: float
    pixel_size_y: float
    sensor_type: SensorType
    can_abort_exposure: bool
    can_asymmetric_bin: bool
    can_fast_readout: bool
    can_get_cooler_power: bool
    can_pulse_guide: bool
    can_set_ccd_temperature: bool
    can_stop_exposure: bool
    has_shutter: bool
    max_adu: int
    electrons_per_adu: float

class ExposureSettings(BaseModel):
    duration: float = 1.0
    bin_x: int = 1
    bin_y: int = 1
    start_x: int = 0
    start_y: int = 0
    num_x: Optional[int] = None
    num_y: Optional[int] = None
    light: bool = True  # True pour image, False pour dark

class ImageData(BaseModel):
    width: int
    height: int
    data: List[List[int]]  # Données d'image 2D
    exposure_duration: float
    timestamp: str

# ===== CLIENT CAMERA SYNCHRONE =====

class ASCOMAlpacaCameraClient(ASCOMAlpacaBaseClient):
    """Client ASCOM Alpaca synchrone pour caméra"""

    def __init__(self, host="localhost", port=11111, device_number=0):
        super().__init__(ASCOMDeviceType.CAMERA, host, port, device_number)

    def get_camera_info(self) -> CameraInfo:
        """Récupère les informations complètes de la caméra"""
        base_info = self.get_device_info()

        results = [
            self._make_request("GET", "cameraxsize"),
            self._make_request("GET", "cameraysize"),
            self._make_request("GET", "maxbinx"),
            self._make_request("GET", "maxbiny"),
            self._make_request("GET", "pixelsizex"),
            self._make_request("GET", "pixelsizey"),
            self._make_request("GET", "sensortype"),
            self._make_request("GET", "canabortexposure"),
            self._make_request("GET", "canasymmetricbin"),
            self._make_request("GET", "canfastreadout"),
            self._make_request("GET", "cangetcoolerpower"),
            self._make_request("GET", "canpulseguide"),
            self._make_request("GET", "cansetccdtemperature"),
            self._make_request("GET", "canstopexposure"),
            self._make_request("GET", "hasshutter"),
        ]

        return CameraInfo(
            **base_info.dict(),
            camera_x_size=results[0].get("Value", 0),
            camera_y_size=results[1].get("Value", 0),
            max_bin_x=results[2].get("Value", 1),
            max_bin_y=results[3].get("Value", 1),
            pixel_size_x=results[4].get("Value", 0.0),
            pixel_size_y=results[5].get("Value", 0.0),
            sensor_type=SensorType(results[6].get("Value", 0)),
            can_abort_exposure=results[7].get("Value", False),
            can_asymmetric_bin=results[8].get("Value", False),
            can_fast_readout=results[9].get("Value", False),
            can_get_cooler_power=results[10].get("Value", False),
            can_pulse_guide=results[11].get("Value", False),
            can_set_ccd_temperature=results[12].get("Value", False),
            can_stop_exposure=results[13].get("Value", False),
            has_shutter=results[14].get("Value", True),
            max_adu=65535,
            electrons_per_adu=1.0
        )

    def start_exposure(self, settings: ExposureSettings) -> None:
        """Démarre une exposition"""
        if settings.num_x is None or settings.num_y is None:
            info = self.get_camera_info()
            settings.num_x = settings.num_x or info.camera_x_size
            settings.num_y = settings.num_y or info.camera_y_size

        self._make_request("PUT", "binx", {"BinX": settings.bin_x})
        self._make_request("PUT", "biny", {"BinY": settings.bin_y})
        self._make_request("PUT", "startx", {"StartX": settings.start_x})
        self._make_request("PUT", "starty", {"StartY": settings.start_y})
        self._make_request("PUT", "numx", {"NumX": settings.num_x})
        self._make_request("PUT", "numy", {"NumY": settings.num_y})

        self._make_request("PUT", "startexposure", {
            "Duration": settings.duration,
            "Light": settings.light
        })

    def abort_exposure(self) -> None:
        """Annule l'exposition en cours"""
        self._make_request("PUT", "abortexposure")

    def stop_exposure(self) -> None:
        """Arrête l'exposition en cours"""
        self._make_request("PUT", "stopexposure")

    def get_camera_state(self) -> CameraState:
        """Récupère l'état de la caméra"""
        result = self._make_request("GET", "camerastate")
        return CameraState(result.get("Value", 0))

    def is_image_ready(self) -> bool:
        """Vérifie si une image est prête"""
        result = self._make_request("GET", "imageready")
        return result.get("Value", False)

    def get_image_array(self) -> ImageData:
        """Récupère les données d'image sous forme de tableau"""
        result = self._make_request("GET", "imagearray")
        image_data = result.get("Value", [])

        last_exposure_duration = self._make_request("GET", "lastexposureduration")
        last_exposure_start = self._make_request("GET", "lastexposurestarttime")

        return ImageData(
            width=len(image_data[0]) if image_data else 0,
            height=len(image_data) if image_data else 0,
            data=image_data,
            exposure_duration=last_exposure_duration.get("Value", 0.0),
            timestamp=last_exposure_start.get("Value", "")
        )

    def set_ccd_temperature(self, temperature: float) -> None:
        """Définit la température cible du CCD"""
        self._make_request("PUT", "setccdtemperature", {"SetCCDTemperature": temperature})

    def get_ccd_temperature(self) -> float:
        """Récupère la température actuelle du CCD"""
        result = self._make_request("GET", "ccdtemperature")
        return result.get("Value", 0.0)

    def set_cooler_on(self, enabled: bool) -> None:
        """Active/désactive le refroidisseur"""
        self._make_request("PUT", "cooleron", {"CoolerOn": enabled})

    def is_cooler_on(self) -> bool:
        """Vérifie si le refroidisseur est activé"""
        result = self._make_request("GET", "cooleron")
        return result.get("Value", False)




class FocuserInfo(BaseDeviceInfo):
    """Informations spécifiques au focuser"""
    absolute: bool
    is_moving: bool
    max_increment: int
    max_step: int
    position: int
    step_size: float
    temp_compensation: bool
    temp_compensation_available: bool
    temperature: float

class ASCOMAlpacaFocuserClient(ASCOMAlpacaBaseClient):
    """Client ASCOM Alpaca synchrone pour focuser"""

    def __init__(self, host="localhost", port=11111, device_number=0):
        super().__init__(ASCOMDeviceType.FOCUSER, host, port, device_number)

    def get_focuser_info(self) -> FocuserInfo:
        """Récupère les informations complètes du focuser"""
        base_info = self.get_device_info()

        results = [
            self._make_request("GET", "absolute"),
            self._make_request("GET", "ismoving"),
            self._make_request("GET", "maxincrement"),
            self._make_request("GET", "maxstep"),
            self._make_request("GET", "position"),
            self._make_request("GET", "stepsize"),
            self._make_request("GET", "tempcomp"),
            self._make_request("GET", "tempcompavailable"),
            self._make_request("GET", "temperature"),
        ]

        return FocuserInfo(
            **base_info.dict(),
            absolute=results[0].get("Value", False),
            is_moving=results[1].get("Value", False),
            max_increment=results[2].get("Value", 0),
            max_step=results[3].get("Value", 0),
            position=results[4].get("Value", 0),
            step_size=results[5].get("Value", 0.0),
            temp_compensation=results[6].get("Value", False),
            temp_compensation_available=results[7].get("Value", False),
            temperature=results[8].get("Value", 0.0)
        )

    def move_absolute(self, position: int) -> None:
        """Déplace le focuser à une position absolue"""
        self._make_request("PUT", "move", {"Position": position})

    def move_relative(self, steps: int) -> None:
        """Déplace le focuser d'un nombre de pas relatif"""
        current_pos = self.get_position()
        new_pos = current_pos + steps
        self.move_absolute(new_pos)

    def halt(self) -> None:
        """Arrête le mouvement du focuser"""
        self._make_request("PUT", "halt")

    def get_position(self) -> int:
        """Récupère la position actuelle"""
        result = self._make_request("GET", "position")
        return result.get("Value", 0)

    def is_moving(self) -> bool:
        """Vérifie si le focuser est en mouvement"""
        result = self._make_request("GET", "ismoving")
        return result.get("Value", False)

    def get_temperature(self) -> float:
        """Récupère la température"""
        result = self._make_request("GET", "temperature")
        return result.get("Value", 0.0)

    def set_temp_compensation(self, enabled: bool) -> None:
        """Active/désactive la compensation"""
        self._make_request("PUT", "tempcomp", {"TempComp": enabled})

    def is_temp_compensation_enabled(self) -> bool:
        """Vérifie si la compensation est active"""
        result = self._make_request("GET", "tempcomp")
        return result.get("Value", False)

    def wait_for_movement_complete(self, timeout: float = 60.0) -> bool:
        """Attend la fin du mouvement"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.is_moving():
                return True
            time.sleep(0.1)

        return False


# === Client Filter Wheel synchrone ===

class FilterWheelInfo(BaseDeviceInfo):
    focus_offsets: List[int]
    names: List[str]
    position: int

class ASCOMAlpacaFilterWheelClient(ASCOMAlpacaBaseClient):
    def __init__(self, host="localhost", port=11111, device_number=0):
        super().__init__(ASCOMDeviceType.FILTERWHEEL, host, port, device_number)

    def get_position(self) -> int:
        result = self._make_request("GET", "position")
        return result.get("Value", 0)

    def set_position(self, position: int) -> None:
        self._make_request("PUT", "position", {"Position": position})

    def get_names(self) -> List[str]:
        result = self._make_request("GET", "names")
        return result.get("Value", [])

    def get_filterwheel_info(self) -> FilterWheelInfo:
        base_info = self.get_device_info()

        focus_offsets = self._make_request("GET", "focusoffsets").get("Value", [])
        names = self._make_request("GET", "names").get("Value", [])
        position = self.get_position()

        return FilterWheelInfo(
            **base_info.dict(),
            focus_offsets=focus_offsets,
            names=names,
            position=position
        )

# === Exemple d'utilisation ===

if __name__ == "__main__":
    filter_client = ASCOMAlpacaFilterWheelClient("localhost", 11111, 0)

    # Connexion
    connected = filter_client.connect()
    print("Connecté:", connected)

    # Lire position
    pos = filter_client.get_position()
    print("Position actuelle:", pos)

    # Changer de position
    filter_client.set_position(2)
    print("Position changée vers 2")

    # Lire noms des filtres
    names = filter_client.get_names()
    print("Filtres:", names)

    # Infos complètes
    info = filter_client.get_filterwheel_info()
    print("Infos roue:", info)

    # Déconnexion
    disconnected = filter_client.disconnect()
    print("Déconnecté:", disconnected)



alpaca_telescope_client = ASCOMAlpacaTelescopeClient("localhost", 11111, 0)
alpaca_camera_client = ASCOMAlpacaCameraClient("localhost", 11111, 0)