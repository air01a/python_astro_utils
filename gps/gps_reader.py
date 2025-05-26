import serial
import pynmea2
import serial.tools.list_ports

class GpsDevice:
    def __init__(self):

        self.port_name = None
        self.port = None

        self.latitude = None
        self.longitude = None
        self.latitude = None
        self.longitude = None
        self.num_sats = None
        self.altitude = None
        self.units = None
        self.gps_qual = None
        self.datestamp = None
        self.timestamp = None
        self.spd_over_grnd = None
        self.true_course = None
        self.isConnected = False
        self.is_fixed = False
        self.last_message = None
        self.num_sv_in_view = None

    def connect(self, port=None, baud_rate=9600, timeout=1):
        if port==None:
            if self.port_name==None:
                return False
            port = self.port_name
        else:
            self.port_name = port

        self.port = serial.Serial(port, baudrate = baud_rate, timeout = timeout)

    def update_gga(self, timestamp, latitude, lat_dir, longitude, lon_dir, num_sats, altitude, units, gps_qual):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.lat_dir = lat_dir
        self.lon_dir = lon_dir
        self.num_sats = num_sats
        self.altitude =  altitude
        self.units = units
        self.gps_qual = gps_qual
        self.is_connected = True
        self.is_fixed = True

        self.last_message = "GGA"


    def set_fix(self, fix):
        self.is_fixed = fix

    def update_rmc(self, timestamp, datestamp, spd_over_grnd, true_course, status):
        self.timestamp = timestamp
        self.datestamp = datestamp
        self.spd_over_grnd = spd_over_grnd
        self.true_course = true_course
        if status != 'A':
            self.is_fixed = False
        self.last_message = "RMC"
        

    def update_gsv(self, num_sats):
        self.num_sv_in_view = num_sats
        self.last_message = "GSV"

    def other_message(self):
        self.last_message = "OTHER"


    def detect_gps_port(self, baudrate=9600, timeout=1):


        ports = serial.tools.list_ports.comports()
        
        for port_info in ports:
            port_name = port_info.device
            try:
                with serial.Serial(port_name, baudrate=baudrate, timeout=timeout) as ser:
                    for _ in range(5):  
                        line = ser.readline().decode('ascii', errors='ignore').strip()
                        if line.startswith('$GP') or line.startswith('$GN'):
                            #
                            self.port_name = port_name
                            return port_name
            except (OSError, serial.SerialException):
                continue

        print("❌ Aucun GPS détecté.")
        return None

    def read_data(self):

        line = self.port.readline().decode('ascii', errors='replace').strip()
        if not line.startswith('$'):
            self.other_message()
            return False

        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            self.other_message()
            return False

        # Trame GGA : Fix GPS, position, satellites, altitude
        if isinstance(msg, pynmea2.GGA):
            if msg.latitude and msg.longitude:
                self.update_gga(msg.timestamp, msg.latitude, msg.lat_dir, msg.longitude, msg.lon_dir, msg.num_sats, msg.altitude, msg.altitude_units, msg.gps_qual)

            else:
                self.set_fix(False)

        # Trame RMC : Date, vitesse, cap
        elif isinstance(msg, pynmea2.RMC):
            self.update_rmc(msg.timestamp, msg.datestamp, msg.spd_over_grnd, msg.true_course, msg.status)

        # Trame GSV : Satellites en vue 
        elif isinstance(msg, pynmea2.GSV):
            self.update_gsv(msg.num_sv_in_view)

        else:
            self.other_message()
        return True

    def close(self):
        if self.port:
            self.port.close()
        self.is_connected = False
        self.is_fixed = False

if __name__ == '__main__':
    print("🔍 Recherche d'un GPS sur les ports série...")
    gps_data = GpsDevice()
    gps_port = gps_data.detect_gps_port()
    GSV = 0
    print(f"✅ GPS détecté sur {gps_port} : {gps_port}")
    if not gps_port is None:
        gps_data.connect(None)
        try:
            print("Lecture des données GPS sur COM5...\n(Interrompez avec Ctrl+C)")
            while True:
                gps_data.read_data()
                if gps_data.last_message=="GGA":
                    if gps_data.is_fixed:
                        print(f"🛰️  Fix GPS obtenu")
                        print(f" - Heure UTC         : {gps_data.timestamp}")
                        print(f" - Latitude          : {gps_data.latitude} {gps_data.lat_dir}")
                        print(f" - Longitude         : {gps_data.longitude} {gps_data.lon_dir}")
                        print(f" - Nombre satellites : {gps_data.num_sats}")
                        print(f" - Altitude          : {gps_data.altitude} {gps_data.units}")
                        print(f" - Qualité du fix    : {gps_data.gps_qual} (0=No fix, 1=GPS fix, 2=DGPS fix)")
                    else:
                        print(f"📡 En attente de fix GPS... Satellites visibles : {msg.num_sats}")
                elif gps_data.last_message=='RMC':
                    print(f"📅 Trame RMC reçue")
                    print(f" - Heure UTC         : {gps_data.timestamp}")
                    print(f" - Date              : {gps_data.datestamp}")
                    print(f" - Vitesse (noeuds)  : {gps_data.spd_over_grnd}")
                    print(f" - Cap (degrés)      : {gps_data.true_course}")
                    if gps_data.is_fixed:
                        print(f" - Statut            : Actif")
                    else:
                        print(f" - Statut            : Inactif (pas de position)")
                elif gps_data.last_message == "GSV":
                    if GSV % 10==0:
                        print(f"🔭 Satellites en vue : {gps_data.num_sv_in_view}")
                    GSV+=1

            else:
                print("❌ Pas de satellites détéctés")
        except KeyboardInterrupt:
            print("\nArrêt du programme.")
            gps_data.close()

