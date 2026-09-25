"""fieldMapping: conservative transfer of field data between non-matching meshes.

- VTU / PVD input and output with arbitrary polyhedra (numpy + standard library)
- finite-element integration on any VTK cell (linear, quadratic, polygon, polyhedron)
- conservative, sign-preserving surface flux transfer (common refinement)
- energy-conserving mapping of layered 1-D through-thickness profiles to 3-D meshes
"""
from pythonLibs.fieldMapping.io import readPvd, readVtu, writePvd, writeVtu
from pythonLibs.fieldMapping.mesh import (Quadrature, UnstructuredMesh, planeSurface, sphereSurface, sphericalShell,
                                          structuredBox)
from pythonLibs.fieldMapping.SurfaceFluxMapper import FluxMapResult, SurfaceFluxMapper
from pythonLibs.fieldMapping.LayeredProfileMapper import LayeredMapResult, LayeredProfileMapper, Station
from pythonLibs.fieldMapping.Material import Material
from pythonLibs.fieldMapping.FieldMappingHandler import FieldMappingHandler

__all__ = ["FieldMappingHandler", "LayeredProfileMapper", "LayeredMapResult", "Station", "Material", "UnstructuredMesh", "Quadrature", "readVtu", "writeVtu", "readPvd", "writePvd", "SurfaceFluxMapper",
           "FluxMapResult", "structuredBox", "planeSurface", "sphereSurface", "sphericalShell"]
