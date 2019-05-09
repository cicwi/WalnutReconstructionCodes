# WalnutReconstructionCodes
This is a collection of Python and MATLAB scripts for loading, pre-processing and 
reconstructing the X-ray CT projection data as described in

"A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning" by
Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten,
Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg

* `FDKReconstruction.m` and `FDKReconstruction.py` compute FDK reconstructions for data from a single source-detector orbit, which leads to high cone angle artifacts.
* `GroundTruthReconstruction.m` and `GroundTruthReconstruction.py` compute an iterative reconstructions using the data from all three source-detector orbits, which leads to a reconstruction free of high cone angle artifacts.

## Requirements

* All scripts make use of the [ASTRA toolbox](www.astra-toolbox.com). For obtaining a comparable scaling of the image intensities between FDK and iterative reconstructions, it is required to use a development version of the ASTRA toolbox more recent than 1.9.0dev.
* `GroundTruthReconstruction.m` makes use of the [SPOT toolbox](http://www.cs.ubc.ca/labs/scl/spot/).

## Contributors

Henri Der Sarkissian (henri.dersarkissian@gmail.com), Felix Lucka (Felix.Lucka@cwi.nl), CWI, Amsterdam
