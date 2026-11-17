# Citing utrcalling

If you use utrcalling in your research, please cite it. Releases are archived
on Zenodo under the DOI
[10.5281/zenodo.23686176](https://doi.org/10.5281/zenodo.23686176), which
always resolves to the latest version; each version also has its own DOI,
listed on that page. The citation metadata is in
[`CITATION.cff`](https://github.com/AndreMacedo88/utrcalling/blob/main/CITATION.cff),
and GitHub's "Cite this repository" button formats it as APA or BibTeX. The
count matrices record the version that made them (`.uns["utrcalling"]`).

## Acknowledgements

Peak calling for 5' data builds on the master's thesis of Eugenia Guerrero
Aruffo: *Interferon regulates transcription and translation in neural stem
cells by targeting regulatory elements on the 5' UTR* (Heidelberg University,
carried out at the German Cancer Research Center, 2026). She chose the linear
kernel, half of Scott's bandwidth per UTR, the filters per treatment and by
molecules; found and fixed peak calling on flat densities; and proposed the
initiator as a measure of how well peaks find start sites, and bootstrap
resampling as a measure of how stable peaks are.

## License

utrcalling is free software, under the
[BSD 3-Clause license](https://github.com/AndreMacedo88/utrcalling/blob/main/LICENSE).
Its author is André Lopes Martins Macedo.
