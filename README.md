# WalnutReconstructionCodes
This is a collection of Python and MATLAB scripts for loading, pre-processing and 
reconstructing X-ray CT projection data of 42 walnuts as described in

[Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten, Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg, "A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning", Sci Data 6, 215 (2019)](https://doi.org/10.1038/s41597-019-0235-y) or [arXiv:1905.04787](https://arxiv.org/abs/1905.04787) (2019)

Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten, Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg, "A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning",  or arXiv:1905.04787 (2019)


* `FDKReconstruction.m` and `FDKReconstruction.py` compute FDK reconstructions for data from a single source-detector orbit, which leads to high cone angle artifacts.
* `GroundTruthReconstruction.m` and `GroundTruthReconstruction.py` compute an iterative reconstructions using the data from all three source-detector orbits, which leads to a reconstruction free of high cone angle artifacts.
* The complete data set can be found via the following links: [1-8](https://doi.org/10.5281/zenodo.2686725), [9-16](https://doi.org/10.5281/zenodo.2686970), [17-24](https://doi.org/10.5281/zenodo.2687386), [25-32](https://doi.org/10.5281/zenodo.2687634), [33-37](https://doi.org/10.5281/zenodo.2687896), [38-42](https://doi.org/10.5281/zenodo.2688111).

## Requirements

* All scripts make use of the [ASTRA toolbox](https://www.astra-toolbox.com/). For obtaining a comparable scaling of the image intensities between FDK and iterative reconstructions, it is required to use a development version of the ASTRA toolbox more recent than 1.9.0dev.
* `GroundTruthReconstruction.m` makes use of the [SPOT toolbox](http://www.cs.ubc.ca/labs/scl/spot/).

## Contributors

Henri Der Sarkissian (henri.dersarkissian@gmail.com), Felix Lucka (Felix.Lucka@cwi.nl), CWI, Amsterdam
