# Developer's Guide

1. Start by forking the main repository: [https://github.com/weecology/MillionTrees](https://github.com/weecology/MillionTrees).
2. Clone your fork of the repository:

    - Using HTTPS: `git clone https://github.com/myUserName/MillionTrees.git`
    - Using SSH: `git clone git@github.com:myUserName/MillionTrees.git`

3. Link your cloned repository to the main repository (typically named `upstream`):

    - `git remote add upstream https://github.com/weecology/MillionTrees.git`

5. Verify your remote settings with:

    ```bash
    git remote -v
    ```

    You should see output similar to:

    ```
    origin    git@github.com:myUserName/MillionTrees.git (fetch)
    origin    git@github.com:myUserName/MillionTrees.git (push)
    upstream  https://github.com/weecology/MillionTrees.git (fetch)
    upstream  https://github.com/weecology/MillionTrees.git (push)
    ```

6. Install the package from the main directory. Use the `-U` or `--upgrade` flag to update or overwrite any previously installed versions:

    ```bash
    pip install . -U
    ```

## Documentation

We use [Sphinx](http://www.sphinx-doc.org/en/stable/) and [Read the Docs](https://readthedocs.org/) for our documentation. Sphinx supports both reStructuredText and markdown as markup languages. 

Source code documentation is automatically included after committing to the main repository. To add additional supporting documentation, create new reStructuredText or markdown files in the `docs` folder.

If you need to reorganize the documentation, refer to the [Sphinx documentation](http://www.sphinx-doc.org/en/stable/).

### Update Documentation

The documentation is automatically updated for changes within modules. **However, it is essential to update the documentation after adding new modules** in the `engines` or `lib` directories.

1. Navigate to the `docs` directory and create a temporary directory, e.g., `source`.
2. Run the following command to generate documentation:

    ```bash
    cd docs
    mkdir source
    sphinx-apidoc -f -o ./source /Users/..../MillionTrees/milliontrees/
    ```

   In this example, `source` is the destination folder for the generated `.rst` files, and `/Users/..../MillionTrees/milliontrees/` is the path to the `milliontrees` source code.

3. Review the generated files, make necessary edits to ensure accuracy, and then commit and push the changes.