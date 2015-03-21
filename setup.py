from setuptools import setup

setup(name='oscillators',
      version='0.1',
      description="Oscillators",
      long_description="",
      author='Ludovic Charleux',
      author_email='ludovic.charleux@univ-savoie.fr',
      license='GPL v2',
      packages=['ocillators'],
      zip_safe=False,
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib"
          ],
      )
