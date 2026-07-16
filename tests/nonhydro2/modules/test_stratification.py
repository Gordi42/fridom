"""ConstantStratification: the ramped-n2 AR-D7 report.

A time-dependent ``n2`` (a spun-up stratification, ``fr.Ramp``) is a
scalar parameter that advances correctly under AdamBashforth (R1's
scalar path — it is NOT field-materialized, unlike the sw2 ``csqr`` or
the beta-plane ``f(y)``, so it never bare-crashes). But it feeds the
module's ``linear=True`` restoring term, so a frozen-``L`` (exponential)
stepper must refuse it (AR-D7), exactly as R1 does for ``coriolis.f0``.
"""
import fridom as fr
import fridom.nonhydro2 as nh
from fridom.nonhydro2.modules.stratification import ConstantStratification


def test_static_n2_reports_no_time_dependent_linear_parameter():
    assert ConstantStratification(n2=1.0).time_dependent_linear_parameters() \
        == ()


def test_ramped_n2_reports_the_stratification_parameter():
    ramp = fr.model.Ramp(0.0, 1.0, period=1.0)
    assert ConstantStratification(
        n2=ramp).time_dependent_linear_parameters() == (
        str(fr.model.params.STRATIFICATION_N2),)


def test_ramped_n2_assembles_and_advances_under_adam_bashforth():
    # the scalar Ramp path (spun-up stratification) is untouched: it
    # assembles cleanly and is NOT rejected at construction.
    grid = fr.spatial.Grid((
        fr.spatial.meshes.IntervalMesh(6, (0.0, 1.0), periodic=True,
                                       name="x"),
        fr.spatial.meshes.IntervalMesh(6, (0.0, 1.0), periodic=True,
                                       name="y"),
        fr.spatial.meshes.IntervalMesh(6, (0.0, 1.0), periodic=False,
                                       name="z")))
    model = nh.Model(
        grid=grid, dsqr=1.0,
        stratification=ConstantStratification(
            n2=fr.model.Ramp(0.0, 1.0, period=1.0)),
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3, order=1))
    assert isinstance(
        model.module(ConstantStratification).n2, fr.model.Ramp)
    model.advance(2)
