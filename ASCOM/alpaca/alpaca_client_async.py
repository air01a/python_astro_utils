import asyncio
import httpx
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel
from enum import Enum
import logging
import threading
import base64
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Générateur thread-safe pour les IDs de transaction
class TransactionIDGenerator:
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
    
    def get_next_id(self) -> int:
        with self._lock:
            self._counter += 1
            if self._counter > 4294967295:  # 2^32 - 1
                self._counter = 1
            return self._counter

# Instance globale du générateur
_transaction_id_generator = TransactionIDGenerator()

class ASCOMDeviceType(str, Enum):
    """Types de dispositifs ASCOM supportés"""
    TELESCOPE = "telescope"
    CAMERA = "camera"
    FOCUSER = "focuser"
    FILTERWHEEL = "filterwheel"
    DOME = "dome"
    ROTATOR = "rotator"

class BaseDeviceInfo(BaseModel):
    """Informations de base communes à tous les dispositifs"""
    name: str
    description: str
    driver_info: str
    driver_version: str
    interface_version: int
    supported_actions: List[str]
    connected: bool

class ASCOMAlpacaBaseClient:
    """Classe de base pour tous les clients ASCOM Alpaca"""
    
    def __init__(self, device_type: ASCOMDeviceType, host: str = "localhost", 
                 port: int = 11111, device_number: int = 0, client_id: int = 1001):
        self.device_type = device_type
        self.host = host
        self.port = port
        self.device_number = device_number
        self.client_id = client_id
        self.base_url = f"http://{host}:{port}/api/v1/{device_type.value}/{device_number}"
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Effectue une requête vers l'API ASCOM Alpaca"""
        url = f"{self.base_url}/{endpoint}"
        
        # Ajouter ClientID et ClientTransactionID aux données
        if data is None:
            data = {}
        
        transaction_id = _transaction_id_generator.get_next_id()
        data.update({
            "ClientID": self.client_id,
            "ClientTransactionID": transaction_id
        })
        
        try:
            if method.upper() == "GET":
                response = await self.client.get(url, params=data)
            else:
                response = await self.client.put(url, data=data)
            
            response.raise_for_status()
            result = response.json()
            
            # Vérifier les erreurs ASCOM
            if result.get("ErrorNumber", 0) != 0:
                raise Exception(f"Erreur ASCOM {result['ErrorNumber']}: {result.get('ErrorMessage', 'Erreur inconnue')}")
            
            return result
            
        except httpx.RequestError as e:
            logger.error(f"Erreur de connexion {self.device_type.value}: {e}")
            raise Exception(f"Erreur de connexion au {self.device_type.value}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP {e.response.status_code} {self.device_type.value}: {e.response.text}")
            raise Exception(f"Erreur HTTP {self.device_type.value}: {e.response.status_code}")

    # Méthodes communes à tous les dispositifs ASCOM
    async def connect(self) -> bool:
        """Connecte le dispositif"""
        result = await self._make_request("PUT", "connected", {"Connected": True})
        return result.get("Value", False)
    
    async def disconnect(self) -> bool:
        """Déconnecte le dispositif"""
        result = await self._make_request("PUT", "connected", {"Connected": False})
        return not result.get("Value", True)
    
    async def is_connected(self) -> bool:
        """Vérifie si le dispositif est connecté"""
        result = await self._make_request("GET", "connected")
        return result.get("Value", False)
    
    async def get_device_info(self) -> BaseDeviceInfo:
        """Récupère les informations de base du dispositif"""
        info_tasks = [
            self._make_request("GET", "name"),
            self._make_request("GET", "description"),
            self._make_request("GET", "driverinfo"),
            self._make_request("GET", "driverversion"),
            self._make_request("GET", "interfaceversion"),
            self._make_request("GET", "supportedactions"),
            self.is_connected()
        ]
        
        results = await asyncio.gather(*info_tasks)
        
        return BaseDeviceInfo(
            name=results[0].get("Value", ""),
            description=results[1].get("Value", ""),
            driver_info=results[2].get("Value", ""),
            driver_version=results[3].get("Value", ""),
            interface_version=results[4].get("Value", 0),
            supported_actions=results[5].get("Value", []),
            connected=results[6]
        )

    async def execute_action(self, action: str, parameters: str = "") -> str:
        """Exécute une action personnalisée sur le dispositif"""
        result = await self._make_request("PUT", "action", {
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
    """Client ASCOM Alpaca pour télescope"""
    
    def __init__(self, host: str = "localhost", port: int = 11111, device_number: int = 0):
        super().__init__(ASCOMDeviceType.TELESCOPE, host, port, device_number)
    
    async def get_telescope_info(self) -> TelescopeInfo:
        """Récupère les informations complètes du télescope"""
        base_info = await self.get_device_info()
        
        # Récupérer les propriétés spécifiques au télescope
        telescope_tasks = [
            self._make_request("GET", "alignmentmode"),
            self._make_request("GET", "aperturearea"),
            self._make_request("GET", "aperturediameter"),
            self._make_request("GET", "canfindhome"),
            self._make_request("GET", "canpark"),
            self._make_request("GET", "canpulseguide"),
        ]
        
        try:
            results = await asyncio.gather(*telescope_tasks, return_exceptions=True)
            
            return TelescopeInfo(
                **base_info.dict(),
                alignment_mode=AlignmentMode(results[0].get("Value", 0) if not isinstance(results[0], Exception) else 0),
                aperture_area=results[1].get("Value", 0.0) if not isinstance(results[1], Exception) else 0.0,
                aperture_diameter=results[2].get("Value", 0.0) if not isinstance(results[2], Exception) else 0.0,
                can_find_home=results[3].get("Value", False) if not isinstance(results[3], Exception) else False,
                can_park=results[4].get("Value", False) if not isinstance(results[4], Exception) else False,
                can_pulse_guide=results[5].get("Value", False) if not isinstance(results[5], Exception) else False,
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
            return TelescopeInfo(**base_info.dict(), **{attr: getattr(TelescopeInfo.__fields__[attr], 'default', None) 
                                                       for attr in TelescopeInfo.__fields__ if attr not in base_info.dict()})
    
    async def get_position(self) -> TelescopePosition:
        """Récupère la position actuelle du télescope"""
        position_tasks = [
            self._make_request("GET", "rightascension"),
            self._make_request("GET", "declination"),
            self._make_request("GET", "altitude"),
            self._make_request("GET", "azimuth"),
            self._make_request("GET", "sideofpier"),
            self._make_request("GET", "tracking"),
            self._make_request("GET", "slewing")
        ]
        
        results = await asyncio.gather(*position_tasks)
        
        return TelescopePosition(
            right_ascension=results[0].get("Value", 0.0),
            declination=results[1].get("Value", 0.0),
            altitude=results[2].get("Value", 0.0),
            azimuth=results[3].get("Value", 0.0),
            side_of_pier=results[4].get("Value", 0),
            tracking=results[5].get("Value", False),
            slewing=results[6].get("Value", False)
        )
    
    async def slew_to_coordinates(self, ra: float, dec: float) -> None:
        """Pointe le télescope vers les coordonnées RA/Dec (synchrone)"""
        await self._make_request("PUT", "slewtocoordinates", {
            "RightAscension": ra,
            "Declination": dec
        })
    
    async def slew_to_coordinates_async(self, ra: float, dec: float) -> None:
        """Pointe le télescope vers les coordonnées RA/Dec (asynchrone)"""
        await self._make_request("PUT", "slewtocoordinatesasync", {
            "RightAscension": ra,
            "Declination": dec
        })

    async def set_utc_date(self, date:str) -> None:
        await self._make_request("PUT", "utcdate",{
            "UTCDate": date,
        })

    async def get_utc_date(self) -> str:
        result = await self._make_request("GET", "utcdate")
        return result.get("Value",'')
                
    async def sync_to_coordinates(self, ra: float, dec: float) -> None:
        await self._make_request("PUT", "synctocoordinates",{
            "RightAscension": ra,
            "Declination": dec
        })
    
    async def abort_slew(self) -> None:
        """Arrête le mouvement du télescope"""
        await self._make_request("PUT", "abortslew")
    
    async def set_tracking(self, enabled: bool) -> None:
        """Active/désactive le suivi"""
        await self._make_request("PUT", "tracking", {"Tracking": enabled})
    
    async def is_tracking(self) -> bool:
        """Vérifie si le suivi est activé"""
        result = await self._make_request("GET", "tracking")
        return result.get("Value", False)
    
    async def park(self) -> None:
        """Stationne le télescope"""
        await self._make_request("PUT", "park")
    
    async def unpark(self) -> None:
        """Déstationne le télescope"""
        await self._make_request("PUT", "unpark")
    
    async def is_parked(self) -> bool:
        """Vérifie si le télescope est stationné"""
        result = await self._make_request("GET", "atpark")
        return result.get("Value", False)
    
    async def is_slewing(self) -> bool:
        """Vérifie si le télescope est en mouvement"""
        result = await self._make_request("GET", "slewing")
        return result.get("Value", False)

    async def move_axis(self, axis:int, rate : float) -> None:
        """Vérifie si le télescope est en mouvement"""
        await self._make_request("PUT", "moveaxis",{"Axis":axis, "Rate":rate})

    async def set_latitude(self, latitude:float):
        await self._make_request("PUT", "sitelatitude", {"SiteLatitude":latitude})

    async def set_longitude(self, latitude:float):
        await self._make_request("PUT", "sitelongitude", {"SiteLongitude":latitude})

    async def set_elevation(self, elevation:float):
        await self._make_request("PUT", "siteelevation", {"SiteElevation":elevation})

    async def get_altitude(self) -> float:
        result = await self._make_request("GET", "altitude")
        return result.get("Value",0)

    
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

class ASCOMAlpacaCameraClient(ASCOMAlpacaBaseClient):
    """Client ASCOM Alpaca pour caméra"""
    
    def __init__(self, host: str = "localhost", port: int = 11111, device_number: int = 0):
        super().__init__(ASCOMDeviceType.CAMERA, host, port, device_number)
    
    async def get_camera_info(self) -> CameraInfo:
        """Récupère les informations complètes de la caméra"""
        base_info = await self.get_device_info()
        
        camera_tasks = [
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
        
        results = await asyncio.gather(*camera_tasks, return_exceptions=True)
        
        return CameraInfo(
            **base_info.dict(),
            camera_x_size=results[0].get("Value", 0) if not isinstance(results[0], Exception) else 0,
            camera_y_size=results[1].get("Value", 0) if not isinstance(results[1], Exception) else 0,
            max_bin_x=results[2].get("Value", 1) if not isinstance(results[2], Exception) else 1,
            max_bin_y=results[3].get("Value", 1) if not isinstance(results[3], Exception) else 1,
            pixel_size_x=results[4].get("Value", 0.0) if not isinstance(results[4], Exception) else 0.0,
            pixel_size_y=results[5].get("Value", 0.0) if not isinstance(results[5], Exception) else 0.0,
            sensor_type=SensorType(results[6].get("Value", 0) if not isinstance(results[6], Exception) else 0),
            can_abort_exposure=results[7].get("Value", False) if not isinstance(results[7], Exception) else False,
            can_asymmetric_bin=results[8].get("Value", False) if not isinstance(results[8], Exception) else False,
            can_fast_readout=results[9].get("Value", False) if not isinstance(results[9], Exception) else False,
            can_get_cooler_power=results[10].get("Value", False) if not isinstance(results[10], Exception) else False,
            can_pulse_guide=results[11].get("Value", False) if not isinstance(results[11], Exception) else False,
            can_set_ccd_temperature=results[12].get("Value", False) if not isinstance(results[12], Exception) else False,
            can_stop_exposure=results[13].get("Value", False) if not isinstance(results[13], Exception) else False,
            has_shutter=results[14].get("Value", True) if not isinstance(results[14], Exception) else True,
            max_adu=65535,  # Valeur par défaut pour 16-bit
            electrons_per_adu=1.0
        )
    
    async def start_exposure(self, settings: ExposureSettings) -> None:
        """Démarre une exposition"""
        # Définir les paramètres de binning et de région
        if settings.num_x is None or settings.num_y is None:
            info = await self.get_camera_info()
            settings.num_x = settings.num_x or info.camera_x_size
            settings.num_y = settings.num_y or info.camera_y_size
        
        # Configurer la caméra
        await self._make_request("PUT", "binx", {"BinX": settings.bin_x})
        await self._make_request("PUT", "biny", {"BinY": settings.bin_y})
        await self._make_request("PUT", "startx", {"StartX": settings.start_x})
        await self._make_request("PUT", "starty", {"StartY": settings.start_y})
        await self._make_request("PUT", "numx", {"NumX": settings.num_x})
        await self._make_request("PUT", "numy", {"NumY": settings.num_y})
        
        # Démarrer l'exposition
        await self._make_request("PUT", "startexposure", {
            "Duration": settings.duration,
            "Light": settings.light
        })
    
    async def abort_exposure(self) -> None:
        """Annule l'exposition en cours"""
        await self._make_request("PUT", "abortexposure")
    
    async def stop_exposure(self) -> None:
        """Arrête l'exposition en cours"""
        await self._make_request("PUT", "stopexposure")
    
    async def get_camera_state(self) -> CameraState:
        """Récupère l'état de la caméra"""
        result = await self._make_request("GET", "camerastate")
        return CameraState(result.get("Value", 0))
    
    async def is_image_ready(self) -> bool:
        """Vérifie si une image est prête"""
        result = await self._make_request("GET", "imageready")
        return result.get("Value", False)
    
    async def get_image_array(self) -> ImageData:
        """Récupère les données d'image sous forme de tableau"""
        result = await self._make_request("GET", "imagearray")
        image_data = result.get("Value", [])
        
        # Récupérer les métadonnées
        last_exposure_duration = await self._make_request("GET", "lastexposureduration")
        last_exposure_start = await self._make_request("GET", "lastexposurestarttime")
        
        return ImageData(
            width=len(image_data[0]) if image_data else 0,
            height=len(image_data) if image_data else 0,
            data=image_data,
            exposure_duration=last_exposure_duration.get("Value", 0.0),
            timestamp=last_exposure_start.get("Value", "")
        )
    
    async def set_ccd_temperature(self, temperature: float) -> None:
        """Définit la température cible du CCD"""
        await self._make_request("PUT", "setccdtemperature", {"SetCCDTemperature": temperature})
    
    async def get_ccd_temperature(self) -> float:
        """Récupère la température actuelle du CCD"""
        result = await self._make_request("GET", "ccdtemperature")
        return result.get("Value", 0.0)
    
    async def set_cooler_on(self, enabled: bool) -> None:
        """Active/désactive le refroidisseur"""
        await self._make_request("PUT", "cooleron", {"CoolerOn": enabled})
    
    async def is_cooler_on(self) -> bool:
        """Vérifie si le refroidisseur est activé"""
        result = await self._make_request("GET", "cooleron")
        return result.get("Value", False)


# ===== CLIENT FOCUSER =====

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
    """Client ASCOM Alpaca pour focuser"""
    
    def __init__(self, host: str = "localhost", port: int = 11111, device_number: int = 0):
        super().__init__(ASCOMDeviceType.FOCUSER, host, port, device_number)
    
    async def get_focuser_info(self) -> FocuserInfo:
        """Récupère les informations complètes du focuser"""
        base_info = await self.get_device_info()
        
        focuser_tasks = [
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
        
        results = await asyncio.gather(*focuser_tasks, return_exceptions=True)
        
        return FocuserInfo(
            **base_info.dict(),
            absolute=results[0].get("Value", False) if not isinstance(results[0], Exception) else False,
            is_moving=results[1].get("Value", False) if not isinstance(results[1], Exception) else False,
            max_increment=results[2].get("Value", 0) if not isinstance(results[2], Exception) else 0,
            max_step=results[3].get("Value", 0) if not isinstance(results[3], Exception) else 0,
            position=results[4].get("Value", 0) if not isinstance(results[4], Exception) else 0,
            step_size=results[5].get("Value", 0.0) if not isinstance(results[5], Exception) else 0.0,
            temp_compensation=results[6].get("Value", False) if not isinstance(results[6], Exception) else False,
            temp_compensation_available=results[7].get("Value", False) if not isinstance(results[7], Exception) else False,
            temperature=results[8].get("Value", 0.0) if not isinstance(results[8], Exception) else 0.0,
        )
    
    async def move_absolute(self, position: int) -> None:
        """Déplace le focuser à une position absolue"""
        await self._make_request("PUT", "move", {"Position": position})
    
    async def move_relative(self, steps: int) -> None:
        """Déplace le focuser d'un nombre de pas relatif"""
        current_pos = await self.get_position()
        new_pos = current_pos + steps
        await self.move_absolute(new_pos)
    
    async def halt(self) -> None:
        """Arrête le mouvement du focuser"""
        await self._make_request("PUT", "halt")
    
    async def get_position(self) -> int:
        """Récupère la position actuelle du focuser"""
        result = await self._make_request("GET", "position")
        return result.get("Value", 0)
    
    async def is_moving(self) -> bool:
        """Vérifie si le focuser est en mouvement"""
        result = await self._make_request("GET", "ismoving")
        return result.get("Value", False)
    
    async def get_temperature(self) -> float:
        """Récupère la température du focuser"""
        result = await self._make_request("GET", "temperature")
        return result.get("Value", 0.0)
    
    async def set_temp_compensation(self, enabled: bool) -> None:
        """Active/désactive la compensation de température"""
        await self._make_request("PUT", "tempcomp", {"TempComp": enabled})
    
    async def is_temp_compensation_enabled(self) -> bool:
        """Vérifie si la compensation de température est activée"""
        result = await self._make_request("GET", "tempcomp")
        return result.get("Value", False)
    
    async def wait_for_movement_complete(self, timeout: float = 60.0) -> bool:
        """Attend que le mouvement soit terminé"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if not await self.is_moving():
                return True
            await asyncio.sleep(0.1)
        
        return False


# ===== CLIENT ROUE À FILTRE =====

class FilterWheelInfo(BaseDeviceInfo):
    """Informations spécifiques à la roue à filtres"""
    focus_offsets: List[int]
    names: List[str]
    position: int

class ASCOMAlpacaFilterWheelClient(ASCOMAlpacaBaseClient):
    """Client ASCOM Alpaca pour roue à filtre"""
    
    def __init__(self, host: str = "localhost", port: int = 11111, device_number: int = 0):
        super().__init__(ASCOMDeviceType.FILTERWHEEL, host, port, device_number)
    
    async def get_filterwheel_info(self) -> FilterWheelInfo:
        """Récupère les informations complètes de la roue à filtres"""
        base_info = await self.get_device_info()
        
        tasks = [
            self._make_request("GET", "focusoffsets"),
            self._make_request("GET", "names"),
            self.get_position()
        ]
        
        results = await asyncio.gather(*tasks)
        
        return FilterWheelInfo(
            **base_info.dict(),
            focus_offsets=results[0].get("Value", []),
            names=results[1].get("Value", []),
            position=results[2]
        )
    
    async def get_position(self) -> int:
        """Récupère la position actuelle de la roue"""
        result = await self._make_request("GET", "position")
        return result.get("Value", 0)
    
    async def set_position(self, position: int) -> None:
        """Change la position de la roue"""
        await self._make_request("PUT", "position", {"Position": position})
    
    async def get_names(self) -> List[str]:
        """Récupère la liste des noms des filtres"""
        result = await self._make_request("GET", "names")
        return result.get("Value", [])

alpaca_telescope_client = ASCOMAlpacaTelescopeClient("localhost", 11111, 0)
alpaca_camera_client = ASCOMAlpacaCameraClient("localhost", 11111, 0)
async def main():
    async with ASCOMAlpacaFilterWheelClient("localhost", 11111, 0) as filter_client:
        # Connexion
        await filter_client.connect()

        # Lire la position
        pos = await filter_client.get_position()
        print(f"Position actuelle: {pos}")

        # Changer de position
        await filter_client.set_position(2)
        print("Position changée vers 2")

        # Lire les noms des filtres
        names = await filter_client.get_names()
        print(f"Noms des filtres: {names}")

        # Infos complètes
        info = await filter_client.get_filterwheel_info()
        print(f"Infos roue: {info}")

        # Déconnexion
        await filter_client.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
#http://localhost:11111//management/v1/description pour liste des devices
#http://localhost:11111/management/v1/configureddevices pour la liste des tous les devices configurés