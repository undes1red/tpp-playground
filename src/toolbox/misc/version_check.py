from packaging.specifiers import SpecifierSet
from packaging.version import Version


def version_check(version_str, requirement_str):
    spec = SpecifierSet(requirement_str)
    v = Version(version_str)
    return v in spec
