Repository for the code doing background estimation with neural networks.

Examples contain a python notebook with explaining how to run BKGnet over an image.
It also contains a python file 'libexample.py' with some functions needed to run the 
notebook outside a pipeline. This ile should not be needed when integrating BKGnet 
into a larger pipeline.

For instance, running BKGnet requires to know the  CCD coordinates of the target objects. 
This can be given directly or use the header and a list of target objects with RA, DEC. 
libexample.py deals with the header and RA,DEC but if the coordinates are already given 
to the network, this extra step is no longer needed.




