# General

Fork of the original code created by <a href="https://github.com/amirhk" target="_blank">`Amir-Hossein Karimi`</a> and his co-authors in order to provide Benchmark algorithm for OCEAN.

This code is provided only to reproduce the numerical results for OCEAN. We will not update it. Users are therefore encouraged to use the code from the initial branch of <a href="https://github.com/amirhk/mace">mace</a> for any other use.


# How to build the numerical results

First,

```console
pip install virtualenv
cd mace
virtualenv -p python3 _venv
source _venv/bin/activate
pip install -r requirements.txt
pysmt-install --z3 --confirm-agreement
```

The numerical experiments of OCEAN can be run using 
```console
python runExperimentsOfOCEANpaper.py
```

Finally, view the results under the `_experiments` folder.
