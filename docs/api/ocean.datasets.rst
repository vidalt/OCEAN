Datasets
========

The :mod:`ocean.datasets` module exposes three convenience loaders backed by
``ocean.datasets.Loader``.

.. currentmodule:: ocean.datasets

.. function:: load_adult(*, scale=False, return_mapper=True)

   Load the Adult income dataset.

   When ``return_mapper`` is ``True`` the function returns
   ``((data, target), mapper)``. Otherwise it returns ``(data, target)``.

.. function:: load_compas(*, scale=False, return_mapper=True)

   Load the COMPAS dataset.

   The return convention matches :func:`load_adult`.

.. function:: load_credit(*, scale=False, return_mapper=True)

   Load the Credit dataset.

   The return convention matches :func:`load_adult`.
