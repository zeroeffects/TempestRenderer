This directory contains the blue scarf dataset from the paper

"A radiative transfer framework for rendering materials with anisotropic structure"
Wenzel Jakob, Adam Arbree, Jonathan T. Moon, Kavita Bala and Steve Marschner
ACM Transactions on Graphics (Proceedings of SIGGRAPH 2010)
(http://www.cs.cornell.edu/projects/diffusion-sg10/)

If you use this dataset in your own research, please remember
 to cite the above reference.

The scarf data was produced from a yarn-level spline representation
made by a physical simulation tool for knits, and this was then 
turned into a 3D voxelization. The original spline data is courtesy
of Jonathan Kaldor, and the voxelization was done by Manuel Vargas
Escalante and Manolis Savva.

To understand how the scene works, please take a look at "scarf.xml",
which contains many XML comments. By default, the settings are 
intentionally reduced to produce a rendering within a few minutes,
but at the cost of quality (the nice multiple scattering "glow" will
be missing) -- this produces a rendering similar to the file 
scarf-lowquality.jpg.

The XML file also explains how to change the settings so that the
image looks more appealing, but at the cost of significantly longer
render times (see the file scarf-highquality.jpg for a preview).
