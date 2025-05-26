from pysiril.siril import Siril
from pysiril.wrapper import Wrapper
from os.path import dirname, realpath, join
app = Siril()
cmd = Wrapper(app)
app.Open()  


dir_path = dirname(realpath(__file__))
# Define directories
workdir = join(dir_path, "lights")
process_dir = join(dir_path,'process')

# Set up Siril
app.Execute("set16bits")
app.Execute("setext fit")

def light(light_dir, process_dir):
    cmd.cd(light_dir)
    cmd.convert( 'light', out=process_dir, debayer=True)
    cmd.cd( process_dir )
    #cmd.preprocess('light', dark='dark_stacked', flat='pp_flat_stacked', cfa=True, equalize_cfa=True, debayer=True )
    cmd.register('light')
    cmd.stack('r_light', type='rej', sigma_low=3, sigma_high=3, norm='addscale', output_norm=True, out='../test')
    cmd.close()

#try:
    #master_bias(workdir + '/biases', process_dir)
    #master_flat(workdir + '/flats', process_dir)
    #master_dark(workdir + '/darks', process_dir)
print(workdir)
print(process_dir)
light(workdir,process_dir)
#except Exception as e:
#    print("\n**** ERROR *** " + str(e) + "\n")

app.Close()
del app