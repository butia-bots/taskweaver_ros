import os
import re

import setuptools
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

def get_package_version():
    import datetime
    import json

    version_file = os.path.join(os.path.dirname(__file__), "version.json")
    with open(version_file, "r") as f:
        version_spec = json.load(f)
    base_version = version_spec["prod"]
    main_suffix = version_spec["main"]
    dev_suffix = version_spec["dev"]

    version = base_version
    return version
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", None)
    build_number = os.environ.get("BUILD_BUILDNUMBER", None)

    if branch_name == "production":
        return version

    version += main_suffix if main_suffix is not None else ""
    if branch_name == "main":
        return version

    version += dev_suffix if dev_suffix is not None else ""
    if build_number is not None:
        version += f"+{build_number}"
    else:
        version += f"+local.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    return version


def update_version_file(version: str):
    # Extract the version from the init file.
    VERSIONFILE = "taskweaver/__init__.py"
    with open(VERSIONFILE, "rt") as f:
        raw_content = f.read()

    content = re.sub(r"__version__ = [\"'][^']*[\"']", f'__version__ = "{version}"', raw_content)
    with open(VERSIONFILE, "wt") as f:
        f.write(content)

    def revert():
        with open(VERSIONFILE, "wt") as f:
            f.write(raw_content)

    return revert


version_str = get_package_version()
revert_version_file = update_version_file(version_str)

# Configurations
with open("README.md", "r") as fh:
    long_description = fh.read()


cur_dir = os.path.dirname(
    os.path.abspath(
        __file__,
    ),
)

required_packages = []
with open(os.path.join(cur_dir, "requirements.txt"), "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        else:
            package = line.strip()
            if "whl" in package:
                continue
            required_packages.append(package)
# print(required_packages)

packages = [
    *setuptools.find_packages(),
]

try:
    d = generate_distutils_setup(
        install_requires=required_packages,  # Dependencies
        extras_require={},
        # Searches throughout all dirs for files to include
        packages=packages,
        # Must be true to include files depicted in MANIFEST.in
        # include_package_data=True
    )
    setup(**d)
finally:
    revert_version_file()
