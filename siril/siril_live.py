from pysiril.siril import Siril
from pysiril.wrapper import Wrapper
import os

app = Siril()
cmd = Wrapper(app)
app.Open()  

# Define directories
workdir = "./lights"
process_dir = '.\\process'

# Set up Siril
#app.Execute("set16bits")
#app.Execute("setext fit")

def light(light_dir, process_dir):
    fits_file = [f for f in os.listdir(light_dir) if f.lower().endswith(('.fits', '.fit'))]
    cmd.cd("C:/Users/eniquet/dev/siril")
    #cmd.Execute("start_ls '-rotate'", False)
    cmd.start_ls(rotate=True)
    #cmd.Execute("livestack 'm27.8.00.LIGHT.329.2023-10-01_21-39-23_1.fits'")
    for file in fits_file:
        print(f"livestack {file}")
        cmd.livestack(f"{light_dir}/{file}")
    cmd.stop_ls()
    cmd.load("live_stack_00001.fit")
    cmd.autostretch()
    cmd.Execute("close")
    cmd.close()

#try:
    #master_bias(workdir + '/biases', process_dir)
    #master_flat(workdir + '/flats', process_dir)
    #master_dark(workdir + '/darks', process_dir)
print(workdir)
print(process_dir)

#light(workdir,process_dir)
cmd.cd("..")

cmd.load("live_stack_00001.fit")
cmd.autostretch()
cmd.savejpg("test.jpg")
cmd.Execute("close")
#except Exception as e:
#    print("\n**** ERROR *** " + str(e) + "\n")
#cmd.cd("1/lights")
#cmd.Execute("start_ls -rotate", False)
#cmd.start_ls(rotate=True)
#cmd.Execute("livestack m27.8.00.LIGHT.329.2023-10-01_21-39-23_1.fits")
#cmd.livestack("m27.8.00.LIGHT.329.2023-10-01_21-39-23_1.fits")
#for file in fits_file:
#    print(f"livestack {file}")

#cmd.close()

app.Close()
del app