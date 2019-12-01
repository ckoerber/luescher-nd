![](https://github.com/ckoerber/luescher-nd/workflows/Run%20tests/badge.svg)

# luescher-nd

This repository supports the publication [Renormalization of a Contact Interaction on a Lattice](https://www.arxiv.org).
It contains the scripts which generated the data, the data itself, scripts which generates the plots and the tex files for the paper.


## Install
The modules can be installed via pip:
```bash
pip install [--user] -r requirements.txt
pip install [--user] [-e] .
```

For the exact module versions see `requirements-freeze.txt`.

## Tests
After installing run
```bash
pytest .
```

## Content

The repository contains the following directories

Directory | Content
---|---
`data` | Data files which are used to generate the figures in the paper.
`docs` | The documentation configuration for this repository ([`sphinx`](https://www.sphinx-doc.org/en/master/)).
`luescher_nd` | The module which was used to compute and plot.
`notebooks` | Development and interface files for the repo. Go here to see how it can be used.
`notes` | Notes created during the development process. We do not give warranty for correctness here.
`scripts` | Export scripts for tuning the contact interaction and computing the spectrum + effective range expansion.
`tests` | Test files for the `luescher_nd` module.


## More questions?

Feel free to take a look at [the docs](https://ckoerber.github.io/luescher-nd/).

## Authors
* [Evan Berkowitz](https://github.com/evanberkowitz)
* [Christopher Körber](https://github.com/ckoerber)
* [Tom Luu](https://github.com/luutom)


## Contributing
Feel free to write the authors or file issues on the repository page.

## License
See [`LICENSE`](LICENSE)
