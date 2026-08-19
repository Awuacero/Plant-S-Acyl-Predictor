# Credits and Attributions

## 1. Scientific Background & Citation

The methodology, biophysical filtering logic, and benchmarking underlying **Plant-S-Acyl-Predictor** were developed and described in the following research:

> **Integrative Computational and Experimental S-Acylation Profiling Reveals a Conserved Pollen S-Acylome in Angiosperms**
> *Andrea Román Mateo* <sup>1</sup>, *Fernando Gallego* <sup>3,4</sup>, *Javier Santos* <sup>2</sup>, *Juan de Dios Alché* <sup>1</sup>, *Gonzalo Claros* <sup>2\*</sup>, *Francisco J. Veredas* <sup>3,4</sup> and *Antonio Castro* <sup>1\*</sup>
> 
> <sup>1</sup> Department of Stress, Development and Signaling of Plants, Plant Reproductive Biology and Advanced Microscopy Laboratory (BReMAP), Estación Experimental del Zaidín, CSIC, Granada, Spain.
> <sup>2</sup> Department of Biochemistry and Molecular Biology, University of Málaga, Málaga, Spain.
> <sup>3</sup> Department of Computer Science and Programming Languages, University of Málaga, Málaga, Spain.
> <sup>4</sup> Research Institute of Multilingual Language Technologies, University of Málaga, Málaga, Spain.

If you use this pipeline in your research, please cite the paper above.

## 2. Biophysical Filters (Surface Accessibility)

The structural accessibility analysis implemented in this tool relies on the following established biophysical scales:

*   **Unified Hydrophobicity Scale (UHS):** Koehler, J., Woetzel, N., Staritzbichler, R., Sanders, C.R. and Meiler, J. (2009), A unified hydrophobicity scale for multispan membrane proteins. *Proteins*, 76: 13-29. https://doi.org/10.1002/prot.22315
*   **Kyte-Doolittle Hydropathy Scale:** Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*, 157(1), 105–132. https://doi.org/10.1016/0022-2836(82)90515-0

## 3. Third-Party Software: MusiteDeep

This application integrates and utilizes the deep-learning framework **MusiteDeep** to perform the final predictive step for S-palmitoylation modifications on cysteine residues.

*   **Original MusiteDeep paper:** Wang, D., Guan, Y., Shu, Q., Song, B., & Xu, D. (2017). MusiteDeep: A deep-learning framework for general and kinase-specific phosphorylation site prediction. *Bioinformatics*, 33(24), 3909–3916. https://doi.org/10.1093/bioinformatics/btx496
*   **Repository:** [https://github.com/duolinwang/MusiteDeep_web/tree/master/MusiteDeep](https://github.com/duolinwang/MusiteDeep_web/tree/master/MusiteDeep)

### MusiteDeep License (MIT)

Copyright (c) 2019 duolinwang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
