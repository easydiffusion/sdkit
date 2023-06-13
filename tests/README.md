# Instructions for testing

**For Easy Diffusion developers:**
1. Setup a working installation of Easy Diffusion.
2. Open the `Developer Console`.
3. Run: `python -m pip install pytest`
4. `python -m pip uninstall sdkit` to remove the installed sdkit.
5. `python -m pip install -e git+git@github.com:easydiffusion/sdkit.git#egg=sdkit`
6. Delete the newly created `src/sdkit` directory inside Easy Diffusion:
* Windows: `rmdir src\sdkit`
* Linux/Mac: `rm -r src/sdkit`
7. Create a link to your local sdkit repo:
* Windows: `mklink /J src\sdkit D:\path\to\your\sdkit\repo`
* Linux/Mac: `ln -s /path/to/your/sdkit/repo src/sdkit`
8. Run the tests with: `pytest src\sdkit\tests`

**pytest tip:** You can also run specific test files by running `pytest src\sdkit\tests\test_some_file.py`. For verbose logging, use the `-v` flag.

**For regular sdkit developers:**
1. Setup a working installation of sdkit.
2. Run `python -m pip install pytest`
3. Run `pytest tests` inside the project folder.
