from pysiril.siril import Siril
from pysiril.wrapper import Wrapper
import os
from os.path import dirname, realpath, join

app = Siril()
cmd = Wrapper(app)
app.Open()  

dir_path = dirname(realpath(__file__))
# Define directories
workdir = join(dir_path, "dark")
process_dir = join(dir_path,'process')
# Set up Siril
app.Execute("set16bits")
app.Execute("setext fit")

def master_dark(dark_dir, process_dir):
    cmd.cd(dark_dir )
    cmd.convert( 'dark', out=process_dir, fitseq=True )
    cmd.cd( process_dir )
    cmd.stack( 'dark', type='rej', sigma_low=3, sigma_high=3, norm='no')




master_dark(workdir,process_dir)

app.Close()
del app