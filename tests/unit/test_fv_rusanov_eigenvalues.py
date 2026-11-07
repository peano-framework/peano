import importlib
import inspect
import re

# exahype2.api.fv re-exports the classes under their module names, so we
# fetch the modules (which host the DSL kernels) explicitly
explicit = importlib.import_module("exahype2.api.fv.RusanovWithExplicitEuler")
optimistic = importlib.import_module(
    "exahype2.api.fv.RusanovWithOptimisticExplicitEuler"
)


KERNELS = [
    explicit.timeStepWithRusanov2D,
    explicit.timeStepWithRusanov3D,
    optimistic.timeStepWithRusanov2DOnPatchInterior,
    optimistic.timeStepWithRusanov2DOnShell,
    optimistic.timeStepWithRusanov3DOnPatchInterior,
    optimistic.timeStepWithRusanov3DOnShell,
]


def _evaluated_regions(kernel):
    """The argument eval("maxEigenvalue", ...) is applied to, per call"""
    return re.findall(
        r'eval\("maxEigenvalue", (\w+(?:\[[^\]]*\])?)', inspect.getsource(kernel)
    )


def test_eigenvalues_are_never_evaluated_on_the_whole_halo():
    for kernel in KERNELS:
        for region in _evaluated_regions(kernel):
            # input without a slice is the patch including all of its halo,
            # whereas output without a slice is the patch without halo
            assert region != "input", kernel.__name__
            if "[" not in region:
                continue
            # Along each axis, we may step at most one volume into the halo,
            # i.e. a slice starts at -1 at the lowest and ends at NV+1 at the
            # highest
            for axis in region[region.index("[") + 1 : -1].split(","):
                axis = axis.strip().replace(" ", "")
                if axis == ":":
                    continue
                start, end = axis.split(":")
                assert start not in ("-H", "0-H"), (kernel.__name__, region)
                assert end not in ("NV+H",), (kernel.__name__, region)


def test_2d_and_3d_kernels_test_for_the_same_pde_terms():
    def guards(kernel):
        return set(re.findall(r"^\s*if (\w+):", inspect.getsource(kernel), re.M))

    expected = {"volumetricSource", "flux", "nonconservativeProduct"}
    assert guards(explicit.timeStepWithRusanov2D) == expected
    assert guards(explicit.timeStepWithRusanov3D) == expected
    assert guards(explicit.maxEigenvalueOverPatch2D) == {"maxEigenvalue"}
    assert guards(explicit.maxEigenvalueOverPatch3D) == {"maxEigenvalue"}


def test_time_step_kernels_do_not_return_an_eigenvalue():
    for kernel in KERNELS:
        assert "patchData.maxEigenvalue" not in inspect.getsource(kernel), (
            kernel.__name__
        )
        assert "if maxEigenvalue:" not in inspect.getsource(kernel), kernel.__name__


def test_eigenvalue_kernels_read_the_patch_without_halo():
    for kernel in [explicit.maxEigenvalueOverPatch2D, explicit.maxEigenvalueOverPatch3D]:
        source = inspect.getsource(kernel)
        assert "patchData.maxEigenvalue" in source
        assert "H" not in re.findall(r"\b[NH]V?\b", source), kernel.__name__
        assert len(re.findall(r'eval\("maxEigenvalue", input, x', source)) >= 2
