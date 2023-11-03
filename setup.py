import setuptools

setuptools.setup(
    name='rl_course_pnp',
    version='0.2',
    author='ArgentumWalker',
    description='Deep RL HSE Course 2023',
    long_description_content_type="text/markdown",
    url='https://github.com/gvolsky/RL_course_Predators_and_Preys',
    license='MIT',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=['opencv-python',
                      'Pillow',
                      'numpy',
                      'torchrl'
                     ],
)