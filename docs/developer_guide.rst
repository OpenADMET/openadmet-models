Developer Guide
===============

Welcome to the developer documentation for OpenADMET Models!

Contributing
------------

OpenADMET Models is an open-source project, and we welcome contributions from the community. Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated!
We also welcome feedback and suggestions for improving the package. Please feel free to open issues on our GitHub repository or join our community discussions to discuss ideas and ask questions.

We require that all contributions adhere to our coding standards and pass our test suite. Additionally we ask that you follow our [code of conduct](https://omsf.io/resources/conduct/) to ensure a welcoming and inclusive environment for all contributors.
Additionally we require all contributors to agree to a developer certificate of origin (DCO). This is a simple statement that you have the right to submit the code you are contributing and that you agree to have it included in the project under the project's license. You can indicate your agreement by ticking the DCO box when submitting a pull request on GitHub.
Learn more about DCOs here: `Developer Certificate of Origin <https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin>`__.

License
-------

OpenADMET Models is distributed under the **MIT License** — see the
`LICENSE <https://github.com/OpenADMET/openadmet_models/blob/main/LICENSE>`__
for full details.


Developing the Package
----------------------

Follow the steps in the `Installation <installation>`__ guide to set up your development environment. Remember to install the package in editable mode using:

```bash
pip install -e .
```

So that changes to the source code are reflected immediately.


Testing
-------

We require that new features and bug fixes include appropriate tests. Tests are located in the `tests/` directory. We use `pytest` as our testing framework.
Tests are separated into `unit` and `integration` tests. Unit tests focus on individual components, while integration tests ensure that different parts of the package work together as expected.
In particular the `Anvil` workflows are tested extensively in the integration test suite. 

Tests are automatically run on each pull request using GitHub Actions to ensure code quality and functionality. You can run the tests locally using:

```bash
# from the root of the repository

# integration tests
pytest openadmet/models/tests/integration

# unit tests
pytest openadmet/models/tests/unit
```

You can also run a GPU version of the integration tests if you have a compatible GPU and the necessary drivers installed:

```bash
pytest -v -m gpu openadmet/models/tests/integration
```


Documentation
-------------

The documentation is built using Sphinx and is located in the `docs/` directory.

To build the documentation locally you will need to create a new conda environment with the dependencies listed in `docs/environment.yml`:

```bash
mamba env create -f docs/environment.yml
mamba activate openadmet_models_docs
cd docs
make html
```

You can then view the documentation by opening `_build/html/index.html` in your web browser.



Code Style
----------

We use pre-commit hooks to enforce code style and quality. These should run automatically when you submit a PR to the repository. 



Tips and Tricks
----------------

- Use descriptive commit messages to make it easier to understand the history of changes.
- Break large changes into smaller, manageable chunks to simplify the review process.
- Keep your branches up to date with the main branch to avoid merge conflicts.
- Write tests for new features and bug fixes to ensure code quality and prevent regressions.

Getting Help
------------

We are a very friendly bunch! If you have any questions or need assistance, please don't hesitate to reach out. You can open an issue on our GitHub repository or join our community discussions on [Github Discussions](https://github.com/orgs/OpenADMET/discussions).
We look forward to your contributions and hope you enjoy working with OpenADMET Models!
