Repository for the code doing background estimation with neural networks.

The main code is at 'bkgnet' directory. 

Examples contain a python notebook with explaining how to run BKGnet over an image.
It also contains a python file 'libexample.py' with some functions needed to run the 
notebook outside a pipeline.

When integrating this to a larger pipeline, libexample.py should not be needed. 

For instane, running BKGnet requires to know the  CCD coordinates of the target objects. 
This can be given directly or use the header and a list of target objects with RA, DEC. 
libexample.py deals with the header and RA,DEC butif thecoordinates are already proportionated 
to the network, this extra step is no longer needed.




