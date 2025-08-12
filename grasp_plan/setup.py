from setuptools import setup, find_packages

setup(
    name="grasp_plan",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line for line in open('requirements.txt').readlines()
        if "@" not in line
    ],
    description="Multi-Object Reasoning and Task-Specific Grasping",
    author="Chrisantus Eze",
    author_email="chrisantus.eze@okstate.edu",
    license="MIT Software License",
    url="",
    keywords="robotics manipulation learning computer-vision",
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
)
