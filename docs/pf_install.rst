Installation
==============

``noisefinder`` is not yet published on PyPI. Until it is, install it directly
from the GitHub repository.

Install with `pip`
-------------------

.. code:: bash

    pip install git+https://github.com/lorsala/noisefinder.git

This requires Python 3.10 or later. Dependencies (``numpy``, ``scipy``,
``mpmath``) are installed automatically.

Install for development
------------------------

To contribute to ``noisefinder``, clone the repository and install it with
`Poetry <https://python-poetry.org/>`_, including the test and docs
dependency groups:

.. code:: bash

    git clone https://github.com/lorsala/noisefinder.git
    cd noisefinder
    poetry install --with test,docs

See :doc:`pf_contributing` for more on contributing to the project.
